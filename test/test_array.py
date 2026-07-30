# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from typing import Annotated
from unittest.mock import patch

import cuda.tile
import cuda.tile as ct
import torch
import math
from cuda.tile import TileTypeError
from cuda.tile._bytecode import BytecodeVersion
from util import assert_equal
from conftest import requires_tileiras


@ct.kernel
def array_attr_kernel(X, out):
    ndim = X.ndim
    shape = X.shape
    strides = X.strides
    ct.static_assert(ndim == 3)
    ct.static_assert(len(shape) == ndim)
    ct.static_assert(len(strides) == ndim)

    ct.store(out, (0,), shape[0])
    ct.store(out, (1,), shape[1])
    ct.store(out, (2,), shape[2])
    ct.store(out, (3,), strides[0])
    ct.store(out, (4,), strides[1])
    ct.store(out, (5,), strides[2])


def test_array_attr():
    x = torch.zeros((2, 3, 4), device='cuda')
    out = torch.zeros(6, device='cuda', dtype=torch.int64)
    ct.launch(torch.cuda.current_stream(),
              (1,),
              array_attr_kernel, (x, out))
    assert list(out[0:3]) == list(x.shape)
    assert list(out[3:6]) == list(x.stride())


def test_array_getitem():
    @ct.kernel
    def kernel(x):
        x[0]

    x = torch.zeros((10,), device='cuda')
    with pytest.raises(TileTypeError, match="Arrays are not directly subscriptable"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_array_setitem():
    @ct.kernel
    def kernel(x):
        x[0] = 3.0

    x = torch.zeros((10,), device='cuda')
    with pytest.raises(TileTypeError, match="Arrays do not support item assignment. Use store()"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_array_aug_setitem():
    @ct.kernel
    def kernel(x):
        x[0] += 3

    x = torch.zeros((10,), device='cuda')
    with pytest.raises(TileTypeError, match="Arrays are not directly subscriptable"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


@ct.kernel
def int64_index_inc1(x: ct.IndexedWithInt64, y: ct.IndexedWithInt64, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid, 0), shape=(TILE, 1))
    ct.store(y, index=(bid, 0), tile=tx + 1)


@requires_tileiras(BytecodeVersion.V_13_3)
def test_int64_index_inc1():
    """
    This test may be excluded from selected CI jobs with
    ``-k "not int64_index"`` because it requires a very large allocation.
    Keep ``int64_index`` in the test name unless those CI filters are updated.
    """
    n = (1 << 32) + 5

    x = torch.randint(-128, 127, (n, 1), device='cuda', dtype=torch.int8)
    y = torch.zeros(n, 1, device='cuda', dtype=torch.int8)

    TILE = 2048
    grid = (math.ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, int64_index_inc1, (x, y, TILE))
    assert_equal(y, x + 1)


def test_int64_index_overflow_without_annotation():
    # Stride > INT32_MAX triggers OverflowError without allocating 6 GiB.
    # dim-0 stride 2**32 exceeds INT32_MAX; dim-1 stride 0 keeps storage at 128 elements.
    base = torch.zeros(128, device='cuda', dtype=torch.bfloat16)
    x = torch.as_strided(base, (1, 25165824, 1, 128), (2**32, 0, 0, 1))
    out = torch.as_strided(base, (1, 25165824, 1, 128), (2**32, 0, 0, 1))

    @ct.kernel
    def kernel(value, out_):
        pass

    with pytest.raises(OverflowError):
        ct.launch(torch.cuda.current_stream(),
                  (1,),
                  kernel, (x, out))


@ct.kernel
def load_static_shaped(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_shape_dims=(0, 1))], out):
    t = ct.load(x, (0, 0), (16, 16))
    ct.store(out, (0, 0), t)


def test_static_shape_standalone_recompile():
    k = cuda.tile.kernel(load_static_shaped._pyfunc)
    shapes = [(16, 16), (32, 32), (16, 16)]
    with patch('cuda.tile._compile.compile_tile',
               side_effect=cuda.tile._compile.compile_tile) as mock_compile:
        for shape in shapes:
            x = torch.randint(0, 100, shape, dtype=torch.int32, device='cuda')
            out = torch.zeros((16, 16), dtype=torch.int32, device='cuda')
            ct.launch(torch.cuda.current_stream(), (1,), k, (x, out))
            assert_equal(out, x[:16, :16])
    assert mock_compile.call_count == 2


def test_static_shape_constraint_values():
    x = torch.zeros((48, 32), dtype=torch.float16, device='cuda')
    out = torch.zeros((16, 16), dtype=torch.float16, device='cuda')
    sig = ct.compilation.KernelSignature.from_kernel_args(
        load_static_shaped, (x, out),
        ct.compilation.CallingConvention.cutile_python_v2())

    constraint = sig.parameters[0]
    assert constraint.shape_constant == (48, 32)
    assert constraint.stride_constant == (None, 1)
    assert constraint.stride_divisible_by == (8, 1)
    assert constraint.may_alias_internally is False
    assert constraint.base_addr_divisible_by == 16


def test_static_shape_annotation_validation():
    with pytest.raises(TypeError, match="must be int"):
        ct.ArrayAnnotation(static_shape_dims=(False,))


def test_static_shape_out_of_range_axis():
    @ct.kernel
    def bad(x: Annotated[ct.Array, ct.ArrayAnnotation(static_shape_dims=(5,))], out):
        pass

    x = torch.zeros((4, 8), device='cuda')
    out = torch.zeros((4, 8), device='cuda')
    with pytest.raises(ValueError, match="static_shape_dims contains axis 5"):
        ct.launch(torch.cuda.current_stream(), (1,), bad, (x, out))


def test_static_shape_and_stride_annotation_together():
    # finalize() writes shape values then stride values; make sure parsing them in that same order.
    @ct.kernel
    def k(x: Annotated[ct.Array, ct.ArrayAnnotation(
              static_shape_dims=(0,), static_stride_dims=(0,))], out):
        pass

    x = torch.zeros((4, 10), device='cuda')[:, :8]   # shape (4, 8), strides (10, 1)
    assert x.shape == (4, 8) and x.stride() == (10, 1)
    out = torch.zeros((4, 4), device='cuda')
    sig = ct.compilation.KernelSignature.from_kernel_args(
        k, (x, out), ct.compilation.CallingConvention.cutile_python_v2())
    constraint = sig.parameters[0]
    assert constraint.shape_constant == (4, None)
    assert constraint.stride_constant == (10, None)


@ct.kernel
def copy_static_strided(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(0,))], out):
    row_stride = ct.static_eval(x.strides[0])
    ct.static_assert(row_stride == 10)
    t = ct.load(x, (0, 0), (4, 8))
    ct.store(out, (0, 0), t)


def test_stride_is_static():
    buf = torch.arange(40, dtype=torch.float16, device='cuda').reshape(4, 10)
    x = buf[:, :8]
    out = torch.zeros((4, 8), dtype=torch.float16, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), copy_static_strided, (x, out))
    assert_equal(out, x)


@ct.kernel
def assert_annotated_inner_stride_one(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(1,))], out):
    ct.static_assert(x.strides[1] == 1)
    ct.store(out, (0,), x.strides[1])


def test_static_stride_annotated_stride_one_is_observable():
    x = torch.zeros((4, 8), dtype=torch.int64, device='cuda')  # contiguous, strides (8, 1)
    out = torch.zeros(1, dtype=torch.int64, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), assert_annotated_inner_stride_one, (x, out))
    assert_equal(out, torch.ones(1, dtype=torch.int64, device='cuda'))


@ct.kernel
def stride_dim0_only(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(0,))], out):
    pass


def test_static_stride_annotation_drops_inferred_stride_one():
    x = torch.zeros((4, 8), dtype=torch.float16, device='cuda')  # contiguous, strides (8, 1)
    out = torch.zeros((4, 8), dtype=torch.float16, device='cuda')
    sig = ct.compilation.KernelSignature.from_kernel_args(
        stride_dim0_only, (x, out),
        ct.compilation.CallingConvention.cutile_python_v2())
    assert sig.parameters[0].stride_constant == (8, None)


@ct.kernel
def load_stride_dim0(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(0,))], out):
    t = ct.load(x, (0, 0), (4, 4))
    ct.store(out, (0, 0), t)


def test_static_stride_annotation_ignores_nonannotated_contiguity():
    # Annotating dim 0 drops the dispatcher's inferred stride==1 (contiguity) for the
    # non-annotated inner dim from the cache key, so arrays that differ only in that dim's
    # contiguity share one compiled kernel. (Inner strides 1/2/3 are all non-16-byte-divisible,
    # so only the stride==1 bit differs -- alignment, which is orthogonal, is left specialized.)
    k = cuda.tile.kernel(load_stride_dim0._pyfunc)
    a = torch.zeros((4, 20), dtype=torch.float32, device='cuda')[:, :4]      # strides (20, 1)
    b = torch.zeros((4, 20), dtype=torch.float32, device='cuda')[:, 0:8:2]   # strides (20, 2)
    c = torch.zeros((4, 20), dtype=torch.float32, device='cuda')[:, 0:12:3]  # strides (20, 3)
    assert (a.stride(), b.stride(), c.stride()) == ((20, 1), (20, 2), (20, 3))
    with patch('cuda.tile._compile.compile_tile',
               side_effect=cuda.tile._compile.compile_tile) as mock_compile:
        for arr in (a, b, c):
            out = torch.zeros((4, 4), dtype=torch.float32, device='cuda')
            ct.launch(torch.cuda.current_stream(), (1,), k, (arr, out))
    assert mock_compile.call_count == 1


def test_static_stride_annotation_validation():
    with pytest.raises(TypeError, match="must be int"):
        ct.ArrayAnnotation(static_stride_dims=(False,))


def test_static_stride_out_of_range_axis():
    @ct.kernel
    def bad(x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(5,))], out):
        pass

    x = torch.zeros((4, 8), device='cuda')
    out = torch.zeros((4, 8), device='cuda')
    # A single (non-list) array must raise cleanly, not abort the interpreter.
    with pytest.raises(ValueError, match="static_stride_dims contains axis 5"):
        ct.launch(torch.cuda.current_stream(), (1,), bad, (x, out))


def test_static_stride_duplicate_axis():
    @ct.kernel
    def bad(x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(0, 0))], out):
        pass

    x = torch.zeros((4, 8), device='cuda')
    out = torch.zeros((4, 8), device='cuda')
    with pytest.raises(ValueError, match="duplicate axis"):
        ct.compilation.KernelSignature.from_kernel_args(
            bad, (x, out), ct.compilation.CallingConvention.cutile_python_v2())


@ct.kernel
def copy_transposed(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(1,))], out):
    # The annotated non-contiguous stride is a compile-time constant and is used for addressing.
    ct.static_assert(x.strides[1] == 4)
    t = ct.load(x, (0, 0), (4, 8))
    ct.store(out, (0, 0), t)


def test_static_stride_transposed_column_major():
    base = torch.arange(32, dtype=torch.float32, device='cuda').reshape(8, 4)  # (4, 1)
    x = base.T  # (4, 8) column-major view, strides (1, 4)
    assert x.stride() == (1, 4)
    out = torch.zeros((4, 8), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), copy_transposed, (x, out))
    assert_equal(out, x)


@ct.kernel
def copy_3d_permuted(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(0, 1, 2))], out):
    ct.static_assert(x.strides[0] == 8)
    ct.static_assert(x.strides[1] == 16)
    t = ct.load(x, (0, 0, 0), (2, 4, 8))
    ct.store(out, (0, 0, 0), t)


def test_static_stride_3d_permuted():
    base = torch.arange(2 * 4 * 8, dtype=torch.float32, device='cuda').reshape(4, 2, 8)
    x = base.permute(1, 0, 2)  # (2, 4, 8), strides (8, 16, 1)
    assert x.stride() == (8, 16, 1)
    out = torch.zeros((2, 4, 8), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), copy_3d_permuted, (x, out))
    assert_equal(out, x)


@ct.kernel
def copy_singleton_axis(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(1,))], out):
    ct.static_assert(x.strides[1] == 1)
    t = ct.load(x, (0, 0), (8, 1))
    ct.store(out, (0, 0), t)


def test_static_stride_singleton_axis():
    # A size-one singleton reports 16-byte stride divisibility whatever its physical stride is;
    # pinning that stride to its exact value must not turn into a contradictory constraint.
    x = torch.arange(8, dtype=torch.float32, device='cuda').unsqueeze(1)
    assert x.shape == (8, 1) and x.stride() == (1, 1)
    out = torch.zeros((8, 1), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), copy_singleton_axis, (x, out))
    assert_equal(out, x)


@ct.kernel
def copy_broadcast_axis(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(1,))], out):
    # A zero stride stays a compile-time constant in the front end; only the tensor-view type
    # keeps that axis dynamic, since TileIR requires strictly positive static strides.
    ct.static_assert(x.strides[1] == 0)
    stride = ct.static_eval(x.strides[1])
    ct.static_assert(stride == 0)
    t = ct.load(x, (0, 0), (4, 8))
    ct.store(out, (0, 0), t)


def test_static_stride_broadcast_axis():
    base = torch.arange(4, dtype=torch.float32, device='cuda').reshape(4, 1)
    x = base.expand(4, 8)  # strides (1, 0)
    assert x.stride() == (1, 0)
    out = torch.zeros((4, 8), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), copy_broadcast_axis, (x, out))
    assert_equal(out, x)
