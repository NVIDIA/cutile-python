# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import operator

import pytest
import torch

from torch.testing import make_tensor
from conftest import float_dtypes, int_dtypes, dtype_id
from util import filecheck, get_bytecode
import cuda.tile as ct
from cuda.tile._exception import TileTypeError, TileSyntaxError
from cuda.tile._numeric_semantics import RoundingMode as RMd


@ct.kernel
def cumsum_axis1(input, output, reverse: ct.Constant[bool],
                 T: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile = ct.load(input, index=(px, 0), shape=(T, N))
    out = ct.cumsum(tile, axis=1, reverse=reverse)
    ct.store(output, index=(px, 0), tile=out)


@pytest.mark.parametrize("shape", [(32, 32)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsumf(shape, dtype, reverse):
    x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumsum_axis1, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumsum(x.flip(1), 1).flip(1) if reverse else torch.cumsum(x, 1)
    atol, rtol = (1e-5, 1e-6) if dtype is torch.float32 else (5e-1, 1e-1)
    torch.testing.assert_close(y, ref_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(32, 32)])
@pytest.mark.parametrize("dtype", int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("low", [-100])
@pytest.mark.parametrize("high", [-20, 100])
def test_cumsumi(shape, dtype, reverse, low, high):
    x = torch.randint(low, high + 1, shape, dtype=dtype, device="cuda")
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumsum_axis1, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumsum(x.flip(1), 1).flip(1) if reverse else torch.cumsum(x, 1)
    ref_result = ref_result.to(dtype)
    torch.testing.assert_close(y, ref_result)


@pytest.mark.parametrize("shape", [(32, 32)])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsumb(shape, reverse):
    x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    y = torch.zeros_like(x, dtype=torch.int32)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumsum_axis1, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumsum(x.flip(1), 1).flip(1) if reverse else torch.cumsum(x, 1)
    ref_result = ref_result.to(torch.int32)
    torch.testing.assert_close(y, ref_result)


@ct.kernel
def cumprod_axis0(input, output, reverse: ct.Constant[bool],
                  T: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile = ct.load(input, index=(px, 0), shape=(T, N))
    out = ct.cumprod(tile, axis=0, reverse=reverse)
    ct.store(output, index=(px, 0), tile=out)


@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
def test_cumprodf(shape, dtype, reverse):
    x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumprod_axis0, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumprod(x.flip(0), 0).flip(0) if reverse else torch.cumprod(x, 0)
    atol, rtol = (1e-5, 1e-6) if dtype is torch.float32 else (1e-4, 1e-1)
    torch.testing.assert_close(y, ref_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("dtype", int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("low", [-100])
@pytest.mark.parametrize("high", [-20, 100])
def test_cumprodi(shape, dtype, reverse, low, high):
    x = torch.randint(low, high + 1, shape, dtype=dtype, device="cuda")
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumprod_axis0, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumprod(x.flip(0), 0).flip(0) if reverse else torch.cumprod(x, 0)
    ref_result = ref_result.to(dtype)
    torch.testing.assert_close(y, ref_result)


@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumprodb(shape, reverse):
    x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    y = torch.zeros_like(x).to(torch.int32)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumprod_axis0,
              (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumprod(x.flip(0), 0).flip(0) if reverse else torch.cumprod(x, 0)
    ref_result = ref_result.to(torch.int32)
    torch.testing.assert_close(y, ref_result)


def make_scan_rounding_mode(scan_op, rounding_mode):
    @ct.kernel
    def kernel(input, output, T: ct.Constant[int], N: ct.Constant[int]):
        px = ct.bid(0)
        tile = ct.load(input, index=(px, 0), shape=(T, N))
        out = scan_op(tile, axis=0, reverse=False, rounding_mode=rounding_mode)
        ct.store(output, index=(px, 0), tile=out)
    return kernel


@pytest.mark.use_mlir
@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op_func, tile_op", [(ct.cumsum, "addf"), (ct.cumprod, "mulf")])
@pytest.mark.parametrize("rounding_mode",
                         [RMd.RN, RMd.RZ, RMd.RM, RMd.RP, RMd.FULL, RMd.APPROX, RMd.RZI])
def test_scan_rounding_mode(
    shape, dtype, op_func, tile_op, rounding_mode
):
    should_raise_rounding_mode = rounding_mode in [RMd.RZI, RMd.APPROX, RMd.FULL]
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    kernel = make_scan_rounding_mode(op_func, rounding_mode)
    if should_raise_rounding_mode:
        with pytest.raises(TileTypeError,
                           match=fr"Rounding mode {rounding_mode.value} is not supported"):
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1]))
    else:
        bytecode = get_bytecode(kernel, (x, y, shape[0], shape[1]))
        if rounding_mode is RMd.RN:
            # Rmd.RN as the default rounding mode is not included in the mlir text
            check_directive = "// CHECK-NOT: rounding<{{[^>]*}}>"
        else:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]] rounding<{rounding_mode.value}>"
            )
        filecheck(bytecode, check_directive)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1]))


def make_scan_flush_to_zero(scan_op, flush_to_zero):
    @ct.kernel
    def kernel(input, output, T: ct.Constant[int], N: ct.Constant[int]):
        px = ct.bid(0)
        tile = ct.load(input, index=(px, 0), shape=(T, N))
        out = scan_op(tile, axis=0, reverse=False, flush_to_zero=flush_to_zero)
        ct.store(output, index=(px, 0), tile=out)
    return kernel


@pytest.mark.use_mlir
@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op_func, tile_op", [(ct.cumsum, "addf"), (ct.cumprod, "mulf")])
@pytest.mark.parametrize("flush_to_zero", [True, False])
def test_scan_flush_to_zero(shape, dtype, op_func, tile_op, flush_to_zero):
    should_raise = flush_to_zero and (dtype != torch.float32)
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    kernel = make_scan_flush_to_zero(op_func, flush_to_zero)
    if should_raise:
        with pytest.raises(TileTypeError,
                           match=r"Flush to zero can only be used for float32 type"):
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1]))
    else:
        bytecode = get_bytecode(kernel, (x, y, shape[0], shape[1]))
        if flush_to_zero:
            check_directive = f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]] flush_to_zero :"
        else:
            check_directive = f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]]{{{{[[:space:]]*}}}}:"
        filecheck(bytecode, check_directive)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1]))


# ========== Custom ct.scan (user-defined lambda) ==========

@pytest.mark.parametrize("shape", [(16, 16)])
@pytest.mark.parametrize("dtype", float_dtypes + int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("flavor", ["lambda", "def", "operator"])
def test_scan_custom_cumsum(shape, dtype, reverse, flavor):
    @ct.kernel
    def kernel_lambda(x, y, reverse: ct.Constant[bool]):
        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=1, func=lambda a, b: a + b, identity=0, reverse=reverse)
        ct.store(y, (0, 0), yt)

    @ct.kernel
    def kernel_def(x, y, reverse: ct.Constant[bool]):
        def f(a, b):
            return a + b

        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=1, func=f, identity=0, reverse=reverse)
        ct.store(y, (0, 0), yt)

    @ct.kernel
    def kernel_operator(x, y, reverse: ct.Constant[bool]):
        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=1, func=operator.add, identity=0, reverse=reverse)
        ct.store(y, (0, 0), yt)

    kernel = locals()[f"kernel_{flavor}"]

    # Prepare input and reference depending on dtype
    if dtype in float_dtypes:
        x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
        ref = torch.cumsum(x.flip(1), 1).flip(1) if reverse else torch.cumsum(x, 1)
        atol, rtol = (1e-5, 1e-6) if dtype is torch.float32 else (5e-1, 1e-1)
    else:
        numel = shape[0] * shape[1]
        base = torch.arange(numel, dtype=torch.int32, device="cuda").reshape(shape)
        x = base.to(dtype)
        ref = torch.cumsum(x.flip(1), 1).flip(1) if reverse else torch.cumsum(x, 1)
        ref = ref.to(x.dtype)

    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, reverse))

    if dtype in float_dtypes:
        torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)
    else:
        torch.testing.assert_close(y, ref)


@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_custom_cumprod(shape, dtype, reverse):
    @ct.kernel
    def scan_cumprod_axis0(input, output, reverse: ct.Constant[bool],
                           T: ct.Constant[int], N: ct.Constant[int]):
        px = ct.bid(0)
        tile = ct.load(input, index=(px, 0), shape=(T, N))
        out = ct.scan(tile, axis=0, func=lambda a, b: a * b, identity=1, reverse=reverse)
        ct.store(output, index=(px, 0), tile=out)

    x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, scan_cumprod_axis0,
              (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumprod(x.flip(0), 0).flip(0) if reverse else torch.cumprod(x, 0)
    atol, rtol = (1e-5, 1e-6) if dtype is torch.float32 else (1e-4, 1e-1)
    torch.testing.assert_close(y, ref_result, atol=atol, rtol=rtol)


def test_custom_scan_last_axis():
    @ct.kernel
    def kernel(x, y):
        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=-1, func=lambda a, b: a + b, identity=0)
        ct.store(y, (0, 0), yt)

    x = torch.arange(256, dtype=torch.int32, device="cuda").reshape(16, 16)
    ref = torch.cumsum(x, -1, dtype=torch.int32)
    y = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    torch.testing.assert_close(y, ref)


def test_custom_scan_none_axis():
    @ct.kernel
    def kernel(x, y):
        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=None, func=lambda a, b: a + b, identity=0)
        ct.store(y, (0, 0), yt)

    x = torch.arange(256, dtype=torch.int32, device="cuda").reshape(16, 16)
    y = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    with pytest.raises(
        TileTypeError, match="Expected an integer constant, but given value has type None"
    ):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_custom_scan_ifelse_not_supported():
    @ct.kernel
    def kernel(x, y):
        def f(a, b):
            if ct.bid(0) == 0:
                # In case of type mismatch, compiler will complain about nested branching
                return a + b.type()
            else:
                return (a + b) % 5

        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=1, func=f, identity=0)
        ct.store(y, (0, 0), yt)

    x = torch.arange(256, dtype=torch.int32, device="cuda").reshape(16, 16)
    y = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    with pytest.raises(TileSyntaxError, match="Branching inside scan body is not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_custom_scan_loop_not_supported():
    @ct.kernel
    def kernel(x, y):
        def f(a, b):
            res = ct.zeros((), dtype=ct.int32)
            for i in range(ct.bid(0) + 1):
                res += (a + b) * i
            return res

        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=1, func=f, identity=0)
        ct.store(y, (0, 0), yt)

    x = torch.arange(256, dtype=torch.int32, device="cuda").reshape(16, 16)
    y = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    with pytest.raises(TileSyntaxError, match="Loops inside scan body are not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_custom_scan_printf_not_supported():
    @ct.kernel
    def kernel(x, y):
        def f(a, b):
            ct.printf("%d %d", a, b)
            return a + b

        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=1, func=f, identity=0)
        ct.store(y, (0, 0), yt)

    x = torch.arange(256, dtype=torch.int32, device="cuda").reshape(16, 16)
    y = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    with pytest.raises(TileSyntaxError, match="Operations with memory effects"
                                              " are not supported inside scan body"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_custom_scan_load_not_supported():
    @ct.kernel
    def kernel(x, y):
        def f(a, b):
            return ct.load(x, (a, b), (1, 1)).item()

        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=1, func=f, identity=0)
        ct.store(y, (0, 0), yt)

    x = torch.arange(256, dtype=torch.int32, device="cuda").reshape(16, 16)
    y = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    with pytest.raises(TileSyntaxError, match="Operations with memory effects"
                                              " are not supported inside scan body"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_custom_scan_nested_not_supported():
    @ct.kernel
    def kernel(x, y):
        def f(a, b):
            inner = ct.scan(
                ct.full((1,), 1, dtype=ct.int32), axis=0,
                func=lambda u, v: u + v, identity=0)
            return a + b + inner.item()

        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.scan(xt, axis=1, func=f, identity=0)
        ct.store(y, (0, 0), yt)

    x = torch.arange(256, dtype=torch.int32, device="cuda").reshape(16, 16)
    y = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    with pytest.raises(TileSyntaxError, match="Nested scan/reduction is not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_scan_two_element_tuple_broadcastable_shapes():
    """Scan with 2-element tuple of broadcastable shapes; combine is add (cumsum)."""
    shape_a = (16,)
    shape_b = (64, 16)
    shape_broadcasted = (64, 16)

    @ct.kernel
    def kernel(x, w, out_a, out_b):
        xt = ct.load(x, (0,), shape_a)
        wt = ct.load(w, (0, 0), shape_b)

        def combine(prev_a, prev_b, curr_a, curr_b):
            return (prev_a + curr_a, prev_b + curr_b)

        cumsum_a, cumsum_b = ct.scan(
            (xt, wt), axis=-1, func=combine, identity=(0, 0)
        )
        ct.store(out_a, (0, 0), cumsum_a)
        ct.store(out_b, (0, 0), cumsum_b)

    x = torch.randint(-10, 10, shape_a, dtype=torch.int32, device="cuda")
    w = torch.randint(-10, 10, shape_b, dtype=torch.int32, device="cuda")
    out_a = torch.zeros(shape_broadcasted, dtype=torch.int32, device="cuda")
    out_b = torch.zeros(shape_broadcasted, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, w, out_a, out_b))

    ref_a = x.expand(shape_broadcasted).cumsum(dim=-1, dtype=torch.int32)
    ref_b = w.expand(shape_broadcasted).cumsum(dim=-1, dtype=torch.int32)
    torch.testing.assert_close(out_a, ref_a)
    torch.testing.assert_close(out_b, ref_b)


def test_scan_four_element_tuple():
    """Scan with a 4-element tuple: cumsum, cumprod, cummax, and accumulated xor."""
    shape = (64, 64)

    @ct.kernel
    def kernel(x, y, z, w, out_sum, out_prod, out_max, out_xor):
        xt = ct.load(x, (0, 0), shape)
        yt = ct.load(y, (0, 0), shape)
        zt = ct.load(z, (0, 0), shape)
        wt = ct.load(w, (0, 0), shape)

        def combine(prev_sum, prev_prod, prev_max, prev_xor, curr_a, curr_b, curr_c, curr_d):
            return (
                prev_sum + curr_a,
                prev_prod * curr_b,
                ct.maximum(prev_max, curr_c),
                ct.bitwise_xor(prev_xor, curr_d),
            )

        cumsum_out, cumprod_out, cummax_out, cumxor_out = ct.scan(
            (xt, yt, zt, wt), axis=1, func=combine,
            identity=(0.0, 1.0, float("-inf"), 0),
        )
        ct.store(out_sum, (0, 0), cumsum_out)
        ct.store(out_prod, (0, 0), cumprod_out)
        ct.store(out_max, (0, 0), cummax_out)
        ct.store(out_xor, (0, 0), cumxor_out)

    x = torch.rand(shape, dtype=torch.float32, device="cuda") * 2 - 1
    y = torch.rand(shape, dtype=torch.float32, device="cuda") * 0.5 + 0.75  # positive for cumprod
    z = torch.rand(shape, dtype=torch.float32, device="cuda") * 2 - 1
    w = torch.randint(0, 256, shape, dtype=torch.int32, device="cuda")
    out_sum = torch.zeros(shape, dtype=torch.float32, device="cuda")
    out_prod = torch.zeros(shape, dtype=torch.float32, device="cuda")
    out_max = torch.zeros(shape, dtype=torch.float32, device="cuda")
    out_xor = torch.zeros(shape, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel,
              (x, y, z, w, out_sum, out_prod, out_max, out_xor))

    ref_sum = torch.cumsum(x, dim=1)
    ref_prod = torch.cumprod(y, dim=1)
    ref_max = torch.cummax(z, dim=1).values
    ref_xor = torch.empty_like(w)
    ref_xor[:, 0] = w[:, 0]
    for j in range(1, shape[1]):
        ref_xor[:, j] = ref_xor[:, j - 1] ^ w[:, j]
    torch.testing.assert_close(out_sum, ref_sum, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(out_prod, ref_prod, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(out_max, ref_max, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(out_xor, ref_xor)
