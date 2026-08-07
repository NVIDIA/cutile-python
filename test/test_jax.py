# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import export
    from jax.sharding import NamedSharding, PartitionSpec as P
    import ml_dtypes
except ImportError:
    pytest.skip("JAX module not found", allow_module_level=True)

import cuda.tile as ct
from cuda.tile.jax import OutputPlaceholder, InputOutput, cutile_call


@ct.kernel
def _scale(x, y, c: ct.Constant):
    bid = ct.bid(0)
    ct.store(y, bid, ct.load(x, bid, 1) * c)


@ct.kernel
def _copy(x, y):
    bid = ct.bid(0)
    ct.store(y, bid, ct.load(x, bid, 1))


@ct.kernel
def _scale_non_const(x, y, c):
    bid = ct.bid(0)
    ct.store(y, bid, ct.load(x, bid, 1) * c)


@ct.kernel
def _scale_non_const_i64(x, y, c: ct.ScalarInt64):
    bid = ct.bid(0)
    ct.store(y, bid, ct.load(x, bid, 1) * c)


@ct.kernel
def _negate(x, y):
    bid = ct.bid(0)
    ct.store(y, bid, -ct.load(x, bid, 1))


@ct.kernel
def _interleaved(c: ct.Constant, x, y):
    bid = ct.bid(0)
    ct.store(y, bid, ct.load(x, bid, 1) * c)


@ct.kernel
def _split(x, y, z):
    bid = ct.bid(0)
    val = ct.load(x, bid, 1)
    ct.store(y, bid, val)
    ct.store(z, bid, -val)


@ct.kernel
def _double_inplace(x):
    bid = ct.bid(0)
    ct.store(x, bid, ct.load(x, bid, 1) * 2)


@ct.kernel
def _scale_i64(x: ct.IndexedWithInt64, y: ct.IndexedWithInt64, c: ct.Constant):
    bid = ct.bid(0)
    ct.store(y, bid, ct.load(x, bid, 1) * c)


@ct.kernel
def _mixed_scalars(x, y,
                   gate: ct.Constant[bool],
                   mult: ct.Constant[int],
                   offset: ct.Constant[float]):
    bid = ct.bid(0)
    v = ct.load(x, bid, 1) * mult + offset
    ct.store(y, bid, gate * v)


def _f32(*shape):
    return jnp.arange(int(np.prod(shape) or 1), dtype=jnp.float32).reshape(shape)


def _array(*shape, dtype):
    return jnp.arange(int(np.prod(shape) or 1), dtype=dtype).reshape(shape)


def test_eager_call():
    x = _f32(10)
    y = cutile_call((10,), _scale, (x, OutputPlaceholder(x.shape, x.dtype), 3))
    np.testing.assert_array_equal(y, 3 * x)


def test_jit_call():
    @jax.jit
    def graph(x):
        return cutile_call((10,), _scale,
                           (x, OutputPlaceholder(x.shape, x.dtype), 3))

    x = _f32(10)
    np.testing.assert_array_equal(graph(x), 3 * x)


@pytest.mark.parametrize("dtype", [ml_dtypes.bfloat16,
                                   ml_dtypes.float8_e4m3fn,
                                   ml_dtypes.float8_e5m2,
                                   ml_dtypes.float8_e8m0fnu])
def test_dtype_support(dtype):
    @jax.jit
    def graph(x):
        return cutile_call((10,), _copy, (x, OutputPlaceholder(x.shape, x.dtype)))

    # float8_e8m0fnu encodes 2^x and has no representation of 0, so arange
    # would round most values to NaN. Use exact powers of two instead.
    if dtype == ml_dtypes.float8_e8m0fnu:
        x = jnp.asarray(2.0 ** np.arange(10), dtype=dtype)
    else:
        x = _array(10, dtype=dtype)
    try:
        np.testing.assert_array_equal(graph(x), x)
    except ct.TileUnsupportedFeatureError:
        pass


def test_multiple_calls():
    @jax.jit
    def graph(x):
        ph = OutputPlaceholder(x.shape, x.dtype)
        y1 = cutile_call((10,), _scale, (x,  ph, 1))
        y2 = cutile_call((10,), _scale, (y1, ph, 1))
        y3 = cutile_call((10,), _scale, (y2, ph, 1))
        return y1 + y2 + y3

    x = _f32(10)
    np.testing.assert_array_equal(graph(x), 3 * x)

    text = jax.jit(graph).lower(_f32(10)).as_text()
    assert text.count('@cutile_launch') == 3


def test_distinct_kernels_in_one_graph():
    """Two distinct kernels in the same graph each emit a launch op."""
    def graph(x):
        ph = OutputPlaceholder(x.shape, x.dtype)
        y = cutile_call((10,), _scale,  (x, ph, 2))
        z = cutile_call((10,), _negate, (y, ph))
        return z

    text = jax.jit(graph).lower(_f32(10)).as_text()
    assert text.count('@cutile_launch') == 2


@pytest.mark.parametrize(
    "kernel,value,expected_factor",
    [
        # bool -> packed as int32 (0 or 1)
        (_scale_non_const, True,  1),
        (_scale_non_const, False, 0),
        # python int -> packed as int32 (default)
        (_scale_non_const, 4,     4),
        (_scale_non_const, -3,    -3),
        # python float -> packed as float32
        (_scale_non_const, 0.5,   0.5),
        (_scale_non_const, -1.25, -1.25),
        # ct.ScalarInt64 annotation -> packed as int64; value > 2^31 OK
        (_scale_non_const_i64, 1 << 32, 1 << 32),
    ],
    ids=["bool_true", "bool_false", "int32_pos", "int32_neg",
         "float_half", "float_neg", "int64_large"],
)
def test_runtime_scalar_for_non_constant_param(kernel, value, expected_factor):
    """A Python scalar passed for a non-Constant kernel parameter is forwarded
    as a runtime scalar (packed bit-for-bit into the launch arg vector) and
    does not bake the value into the cubin."""
    x = _f32(10)

    @jax.jit(static_argnums=1)
    def f(x, c):
        return cutile_call((10,), kernel,
                           (x, OutputPlaceholder(x.shape, x.dtype), c))

    expected = np.float32(expected_factor) * np.asarray(x)
    np.testing.assert_array_equal(np.asarray(f(x, value)), expected)


def test_args_in_non_standard_order():
    x = _f32(10)

    def graph(x):
        args = (3, x, OutputPlaceholder(x.shape, x.dtype))
        return cutile_call((10, 1, 1), _interleaved, args)

    y = jax.jit(graph)(x)
    np.testing.assert_array_equal(y, 3 * x)


def test_multiple_outputs():
    """Kernel writing to two output buffers returns a tuple of both."""
    x = _f32(10)

    def graph(x):
        ph = OutputPlaceholder(x.shape, x.dtype)
        return cutile_call((10, 1, 1), _split, (x, ph, ph))

    y, z = jax.jit(graph)(x)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(z, -x)


def test_int64_indexed_array():
    """ct.IndexedWithInt64 annotation flows through to the array constraint
    so the kernel is compiled and called with i64 shape/stride."""
    @jax.jit
    def graph(x):
        return cutile_call((10,), _scale_i64,
                           (x, OutputPlaceholder(x.shape, x.dtype), 3))

    x = _f32(10)
    np.testing.assert_array_equal(graph(x), 3 * x)


def test_mixed_scalar_constant_types():
    """Bool, int, and float scalars all reach the kernel as baked-in
    constants. Each (type, value) tuple compiles to a distinct cubin."""
    x = _f32(10)

    def run(gate, mult, offset):
        @jax.jit
        def graph(x):
            return cutile_call(
                (10,), _mixed_scalars,
                (x, OutputPlaceholder(x.shape, x.dtype), gate, mult, offset))
        return graph(x)

    # bool=True path
    y = run(True, 3, 0.5)
    np.testing.assert_array_equal(y, 3 * x + 0.5)

    # bool=False path: gate * v == 0
    y = run(False, 3, 0.5)
    np.testing.assert_array_equal(y, jnp.zeros_like(x))

    # different int / float values reach the kernel
    y = run(True, 2, -1.0)
    np.testing.assert_array_equal(y, 2 * x - 1.0)


def test_list_constraint_with_static_shape_uses_v2():
    from cuda.tile.jax import _jax as _jax_mod

    element = ct.compilation.ArrayConstraint(
        dtype=ct.float32,
        ndim=1,
        index_dtype=ct.int32,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
        shape_constant=(1,),
    )
    constraint = ct.compilation.ListConstraint(
        element, alias_groups=(), elements_may_alias=False)

    assert _jax_mod._calling_convention_for((constraint,)).version == 2


def test_cubin_id_round_trip(monkeypatch):
    """The 32-byte cubin_id digest round-trips through the FFI u8 array
    attribute, including byte values with the high bit set."""
    from cuda.tile.jax import _jax as _jax_mod

    real_compile = _jax_mod.compile_kernel_cached

    def force_high_bytes(kernel, constraints):
        function_name, cubin_code, _ = real_compile(kernel, constraints)
        return function_name, cubin_code, b"\xff" * 32

    # Clear the python-side compile cache so the forced id is actually used
    # (otherwise a prior test may have populated it with the real id).
    monkeypatch.setattr(_jax_mod, "_COMPILE_CACHE", {})
    monkeypatch.setattr(_jax_mod, "compile_kernel_cached", force_high_bytes)

    @jax.jit
    def graph(x):
        return cutile_call((10,), _scale,
                           (x, OutputPlaceholder(x.shape, x.dtype), 3))

    x = _f32(10)
    np.testing.assert_array_equal(graph(x), 3 * x)


def test_inplace_update():
    """InputOutput arg aliases its input and output buffers; the kernel
    reads and writes the same buffer."""
    @jax.jit
    def graph(x):
        return cutile_call((10,), _double_inplace, (InputOutput(x),))

    x = _f32(10)
    np.testing.assert_array_equal(graph(x), 2 * x)


def _concurrent_worker(args):
    """Worker for test_concurrent_executions; runs in a thread or a fresh
    spawn'd process. Defined at module scope so ProcessPoolExecutor can
    pickle it."""
    factor, iterations = args

    @jax.jit
    def graph(x):
        return cutile_call((10,), _scale,
                           (x, OutputPlaceholder(x.shape, x.dtype), factor))

    x = _f32(10)
    for _ in range(iterations):
        y = np.asarray(graph(x))
        if not np.array_equal(y, factor * np.asarray(x)):
            return f"factor={factor}: mismatch"
    return None


def test_concurrent_executions():
    """Multiple workers compiling and running jitted cuTile graphs in
    parallel."""
    from concurrent.futures import ThreadPoolExecutor

    factors = [1, 2, 3, 4, 42, 42, 42, 42]   # 4 unique + 4 shared
    iterations = 4
    args = [(f, iterations) for f in factors]

    executor = ThreadPoolExecutor(max_workers=len(factors))
    with executor as ex:
        results = list(ex.map(_concurrent_worker, args))

    failures = [r for r in results if r is not None]
    assert not failures, f"worker failures: {failures}"


def test_export_run_in_new_process(tmp_path):
    """Export a jitted cuTile graph, then load and run it in a fresh
    Python process. Exercises FFI re-registration and the cubin embedded
    in the export, with no shared state from the producer process."""
    import os
    import subprocess
    import sys
    import textwrap

    @jax.jit
    def graph(x):
        return cutile_call((10,), _scale,
                           (x, OutputPlaceholder(x.shape, x.dtype), 3))

    disabled = (
        export.DisabledSafetyCheck.custom_call("cutile_launch"),
    )
    x = _f32(10)
    exported = export.export(graph, disabled_checks=disabled)(
        jax.ShapeDtypeStruct(x.shape, x.dtype))
    blob = exported.serialize()

    blob_path = tmp_path / "exp.bin"
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    blob_path.write_bytes(blob)
    np.save(x_path, np.asarray(x))

    runner = textwrap.dedent("""
        import sys
        import numpy as np
        import cuda.tile.jax
        from jax import export
        blob = open(sys.argv[1], 'rb').read()
        x = np.load(sys.argv[2])
        y = export.deserialize(blob).call(x)
        np.save(sys.argv[3], np.asarray(y))
    """)
    subprocess.run(
        [sys.executable, "-c", runner, str(blob_path), str(x_path), str(y_path)],
        check=True, env=os.environ.copy(),
    )

    np.testing.assert_array_equal(np.load(y_path), 3 * np.asarray(x))


# ============== Multigpu Sharding Test ===============
def per_block(scale_a, scale_b, TILE):
    """Build a per-shard cutile_call closure for a 1-D block of size TILE*K."""

    def block(a, b):
        out = OutputPlaceholder(a.shape, a.dtype)
        grid = (a.shape[0] // TILE, 1, 1)
        return cutile_call(grid, _scaled_add, (a, b, out, scale_a, scale_b, TILE))

    return block


@ct.kernel
def _scaled_add(A, B, C, *,
                scale_a: ct.Constant[float],
                scale_b: ct.Constant[float],
                TILE: ct.Constant[int]):
    """Per-tile elementwise: C = A * scale_a + B * scale_b."""
    bid = ct.bid(0)
    a = ct.load(A, index=(bid,), shape=(TILE,))
    b = ct.load(B, index=(bid,), shape=(TILE,))
    ct.store(C, index=(bid,), tile=a * scale_a + b * scale_b)


def _make_inputs(shape, dtype, key):
    a_key, b_key = jax.random.split(key, 2)
    a = jax.random.normal(a_key, shape, dtype=dtype)
    b = jax.random.normal(b_key, shape, dtype=dtype)
    return a, b


@pytest.mark.parametrize("n", range(3))
def test_jit_sharding(n):
    TILE = 128
    ngpu = jax.device_count()
    mesh = jax.make_mesh((ngpu,), ("d",))
    sharding = NamedSharding(mesh, P("d"))

    N = TILE * 8 * ngpu
    a, b = _make_inputs((N,), jnp.float32, jax.random.key(1123 + n))
    a = jax.device_put(a, sharding)
    b = jax.device_put(b, sharding)

    @jax.jit(static_argnums=[2, 3])
    def compute(a, b, sa, sb):
        block = jax.shard_map(per_block(sa, sb, TILE),
                              mesh=mesh,
                              in_specs=(P("d"), P("d")),
                              out_specs=P("d"))
        return block(a, b), a * sa + b * sb

    c, c_ref = compute(a, b, 1.0, 2.0)
    assert jnp.allclose(c, c_ref, atol=1e-5)

    c, c_ref = compute(a, b, 3.0, 4.0)
    assert jnp.allclose(c, c_ref, atol=1e-5)


@ct.kernel
def _tuple_param_variadic_scalars(x, y, addends: tuple[int, ...]):
    bid = ct.bid(0)
    ct.store(y, bid, ct.load(x, bid, 1) + addends[0])


@pytest.mark.parametrize("addends", [(3,), (3, 5)])
def test_tuple_variadic_scalars(addends):
    x = _f32(10)
    ph = OutputPlaceholder(x.shape, x.dtype)
    y = cutile_call((10,), _tuple_param_variadic_scalars, (x, ph, addends))
    np.testing.assert_array_equal(y, x + addends[0])


@ct.kernel
def _tuple_param_arrays(out, pair: tuple[jax.Array, jax.Array]):
    bid = ct.bid(0)
    ct.store(out, bid, ct.load(pair[0], bid, 1) + ct.load(pair[1], bid, 1))


def test_tuple_of_arrays():
    a = _f32(10)
    b = _f32(10) * 2
    ph = OutputPlaceholder(a.shape, a.dtype)
    y = cutile_call((10,), _tuple_param_arrays, (ph, (a, b)))
    np.testing.assert_array_equal(y, a + b)


@ct.kernel
def _tuple_param_nested(x, y, params: tuple[tuple[ct.Constant[int], int], int]):
    bid = ct.bid(0)
    v = ct.load(x, bid, 1) + params[0][0] + params[0][1] + params[1]
    ct.store(y, bid, v)


def test_tuple_nested():
    x = _f32(10)
    ph = OutputPlaceholder(x.shape, x.dtype)
    y = cutile_call((10,), _tuple_param_nested, (x, ph, ((3, 4), 5)))
    np.testing.assert_array_equal(y, x + 12)


@ct.kernel
def _tuple_param_const_scalar(x, y, addends: tuple[ct.Constant[int], int]):
    bid = ct.bid(0)
    ct.store(y, bid, ct.load(x, bid, 1) + addends[0] + addends[1])


def test_tuple_constant_element_drives_recompilation():
    """The constant element of a tuple is part of the compile cache key while a
    runtime-scalar element is not: changing the constant recompiles, changing
    the runtime scalar reuses the cubin."""
    from cuda.tile.jax import _jax as _jax_mod

    _jax_mod._COMPILE_CACHE.clear()
    x = _f32(10)
    ph = OutputPlaceholder(x.shape, x.dtype)

    # Same constant (3), different runtime scalar -> one cubin.
    np.testing.assert_array_equal(
        cutile_call((10,), _tuple_param_const_scalar, (x, ph, (3, 4))), x + 7)
    np.testing.assert_array_equal(
        cutile_call((10,), _tuple_param_const_scalar, (x, ph, (3, 9))), x + 12)
    assert len(_jax_mod._COMPILE_CACHE) == 1

    # Different constant (7) -> distinct cubin.
    np.testing.assert_array_equal(
        cutile_call((10,), _tuple_param_const_scalar, (x, ph, (7, 4))), x + 11)
    assert len(_jax_mod._COMPILE_CACHE) == 2


def test_tuple_wrong_length_rejected():
    x = _f32(10)
    ph = OutputPlaceholder(x.shape, x.dtype)
    with pytest.raises(TypeError, match="expects a tuple of length 2, got 3"):
        cutile_call((10,), _tuple_param_const_scalar, (x, ph, (3, 4, 5)))
