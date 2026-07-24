# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from math import ceil
from io import BytesIO

import torch
from torch.testing import make_tensor

import cuda.tile as ct
from util import assert_close, assert_equal, torch_to_tf32
from cuda.tile._exception import TileTypeError, TileUnsupportedFeatureError
from cuda.tile._cext import CallingConvention
from conftest import float_dtypes, int_dtypes, bool_dtypes, dtype_id
from cuda.tile._bytecode.version import BytecodeVersion
from conftest import requires_tileiras


@pytest.fixture
def shape():
    return (512, )


@pytest.fixture
def tile():
    return 64


@ct.kernel
def array_astype_to_float32(x, y, TILE: ct.Constant[int], use_method: ct.Constant[bool]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    if use_method:
        ty = tx.astype(np.float32)
    else:
        ty = ct.astype(tx, np.float32)
    ct.store(y, index=(bid,), tile=ty)


@pytest.mark.parametrize("use_method", [True, False])
def test_astype(shape, tile, use_method):
    x = make_tensor(shape, dtype=torch.int32, device='cuda')
    ref = x.to(torch.float32)
    y = torch.zeros_like(ref)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, array_astype_to_float32, (x, y, tile, use_method))
    assert_equal(y, ref)


def make_astype_to_f32_kernel(rounding_mode):
    @ct.kernel
    def kernel(x, y, TILE: ct.Constant[int], use_method: ct.Constant[bool]):
        bid = ct.bid(0)
        tx = ct.load(x, index=(bid,), shape=(TILE,))
        if use_method:
            ty = tx.astype(np.float32, rounding_mode=rounding_mode)
        else:
            ty = ct.astype(tx, np.float32, rounding_mode=rounding_mode)
        ct.store(y, index=(bid,), tile=ty)
    return kernel


@pytest.mark.parametrize("use_method", [True, False])
@pytest.mark.parametrize("rounding_mode", [None,
                                           ct.RoundingMode.RN,
                                           pytest.param(
                                                ct.RoundingMode.RM,
                                                marks=requires_tileiras(BytecodeVersion.V_13_4)),
                                           pytest.param(
                                                ct.RoundingMode.RP,
                                                marks=requires_tileiras(BytecodeVersion.V_13_4)),
                                           pytest.param(
                                                ct.RoundingMode.RZ,
                                                marks=requires_tileiras(BytecodeVersion.V_13_4))])
def test_astype_rounding_mode_f64_f32(use_method, rounding_mode):
    low = np.float32(1)
    high = np.nextafter(low, np.float32(2))
    val = np.float64(low) + np.float64(high - low) * 0.6
    x = torch.tensor([-val, val], dtype=torch.float64, device='cuda')

    match rounding_mode:
        case ct.RoundingMode.RN | None: ref = [-high, high]
        case ct.RoundingMode.RM: ref = [-high, low]
        case ct.RoundingMode.RP: ref = [-low, high]
        case ct.RoundingMode.RZ: ref = [-low, low]
    ref = torch.tensor(ref, dtype=torch.float32, device='cuda')

    y = torch.zeros_like(ref)
    grid = (1,)
    kernel = make_astype_to_f32_kernel(rounding_mode)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, 2, use_method))
    assert_equal(y, ref)


def make_astype_to_tf32_kernel(rounding_mode):
    @ct.kernel
    def kernel(x, y, TILE: ct.Constant[int], use_method: ct.Constant[bool]):
        bid = ct.bid(0)
        tx = ct.load(x, index=(bid,), shape=(TILE,))
        if use_method:
            ty = tx.astype(ct.tfloat32, rounding_mode=rounding_mode)
        else:
            ty = ct.astype(tx, ct.tfloat32, rounding_mode=rounding_mode)
        ty = ct.astype(ty, y.dtype)  # because we cannot implicitly cast tfloat32 to float32
        ct.store(y, index=(bid,), tile=ty)
    return kernel


@pytest.mark.parametrize("use_method", [True, False])
@pytest.mark.parametrize("rounding_mode",
                         [None,
                          ct.RoundingMode.RN,
                          pytest.param(
                              ct.RoundingMode.RA,
                              marks=requires_tileiras(BytecodeVersion.V_13_4)),
                          pytest.param(
                              ct.RoundingMode.RZ,
                              marks=requires_tileiras(BytecodeVersion.V_13_4))])
def test_astype_rounding_mode_f32_tf32(use_method, rounding_mode):
    low = np.float32(1)
    high = np.float32(1 + 2**-10)
    val = np.float32(1 + 2**-11)
    x = torch.tensor([-val, val], dtype=torch.float32, device='cuda')

    match rounding_mode:
        case ct.RoundingMode.RN | None: ref = [-low, low]
        case ct.RoundingMode.RA: ref = [-high, high]
        case ct.RoundingMode.RZ: ref = [-low, low]
    ref = torch.tensor(ref, dtype=torch.float32, device='cuda')

    y = torch.zeros_like(ref)
    grid = (1,)
    kernel = make_astype_to_tf32_kernel(rounding_mode)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, 2, use_method))
    assert_equal(y, ref)


def make_astype_to_kernel(rounding_mode, from_dtype, to_dtype):
    @ct.kernel
    def kernel(y):
        ty = ct.ones((2,), dtype=from_dtype)
        ty = ty.astype(to_dtype, rounding_mode=rounding_mode)
        ty = ty.astype(y.dtype)
        ct.store(y, index=(0,), tile=ty)
    return kernel


@pytest.mark.parametrize("rounding_mode", [ct.RoundingMode.RN,
                                           ct.RoundingMode.RA,
                                           ct.RoundingMode.RM,
                                           ct.RoundingMode.RP,
                                           ct.RoundingMode.RZ])
def test_reject_astype_rounding_mode_i32_f32(rounding_mode):
    y = torch.zeros((2,), dtype=torch.int32, device='cuda')
    kernel = make_astype_to_kernel(rounding_mode, y.dtype, ct.float32)

    with pytest.raises(TileTypeError, match="rounding_mode is only valid for float "
                                            "to float conversions"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))


def compile_with(kernel, args, arch: str, version: str):
    sig = ct.compilation.KernelSignature.from_kernel_args(
            kernel, args, CallingConvention.cutile_python_v1())
    ct.compilation.export_kernel(kernel, [sig], output_file=BytesIO(), gpu_code=arch,
                                 output_format="cubin", bytecode_version=version)


@pytest.mark.parametrize("to_dtype", [ct.float32, ct.float16])
@pytest.mark.parametrize("rounding_mode", [ct.RoundingMode.RA,
                                           ct.RoundingMode.RM,
                                           ct.RoundingMode.RP])
def test_reject_astype_rounding_mode_from_float8_e8m0fnu(to_dtype, rounding_mode):
    from_dtype = ct.float8_e8m0fnu
    y = torch.zeros((2,), dtype=torch.float32, device='cuda')
    kernel = make_astype_to_kernel(rounding_mode, from_dtype, to_dtype)

    with pytest.raises(TileTypeError, match=f"rounding_mode={rounding_mode} is "
                                            f"not supported for conversion "
                                            f"from {from_dtype} to {to_dtype}"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))


@requires_tileiras(BytecodeVersion.V_13_4)
@pytest.mark.parametrize("from_dtype", [ct.float32, ct.tfloat32, ct.float8_e5m2,
                                        ct.float16, ct.bfloat16, ct.float8_e4m3fn,
                                        ct.float8_e8m0fnu, ct.float4_e2m1fn])
@pytest.mark.parametrize("rounding_mode", [ct.RoundingMode.RA,
                                           ct.RoundingMode.RM,
                                           ct.RoundingMode.RP,
                                           ct.RoundingMode.RZ])
def test_reject_astype_rounding_mode_bc_version(from_dtype, rounding_mode):
    to_dtype = ct.float64
    y = torch.zeros((2,), dtype=torch.float32, device='cuda')
    kernel = make_astype_to_kernel(rounding_mode, from_dtype, to_dtype)

    with pytest.raises(TileUnsupportedFeatureError,
                       match="The requested conversion and rounding_mode "
                             "require tileiras 13.4 or later. Current version is 13.3."):
        compile_with(kernel, (y,), "sm_100", "13.3")


@ct.kernel
def array_bitcast(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ty = ct.bitcast(tx, y.dtype)
    ct.store(y, index=(bid,), tile=ty)


@ct.kernel
def kernel_astype_tf32(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ty = ct.astype(tx, ct.tfloat32)
    ty = ct.astype(ty, y.dtype)
    ct.store(y, index=(bid,), tile=ty)


@pytest.mark.parametrize("dtype", [torch.float16,
                                   torch.float32,
                                   torch.bfloat16,
                                   torch.float64])
def test_cast_tf32(dtype):
    # Test that tf32 is casted to float32
    x = make_tensor((32, 32), dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    ref = torch_to_tf32(x).view(-1)
    x = x.view(-1)
    y = y.view(-1)
    grid = (ceil(x.numel() / 32), 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel_astype_tf32, (x, y, 32))
    assert_close(y, ref, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("dtype_x, dtype_y", [
    # identities
    (torch.int32, torch.int32),
    (torch.float32, torch.float32),
    (torch.int64, torch.int64),
    (torch.float64, torch.float64),
    (torch.float16, torch.float16),
    # float/int pairs
    (torch.int32, torch.float32),
    (torch.float32, torch.int32),
    (torch.float64, torch.int64),
    (torch.int64, torch.float64),
    # failing pairs with different bitwidths
    (torch.int32, torch.int64),
    (torch.int64, torch.float32),
    (torch.float16, torch.int32),
    # failing pairs with bool
    (torch.bool, torch.int8),
    (torch.uint8, torch.bool),
    (torch.bool, torch.bool),
])
def test_array_bitcast(shape, tile, dtype_x, dtype_y):
    # avoid inputs that could produce nans of infs to not break assert
    if dtype_x == torch.bool:
        x = torch.randint(0, 2, shape, dtype=dtype_x, device='cuda')
    elif dtype_x in (torch.int32, torch.int64, torch.int8, torch.uint8):
        x = torch.randint(0, 100, shape, dtype=dtype_x, device='cuda')
    else:
        x = torch.randn(shape, dtype=dtype_x, device='cuda')
    ref = x.view(dtype=dtype_y)
    y = torch.zeros_like(ref)
    grid = (ceil(shape[0] / tile), 1, 1)
    if (dtype_x == torch.bool or dtype_y == torch.bool
            or dtype_x.itemsize != dtype_y.itemsize):
        with pytest.raises(TileTypeError):
            ct.launch(torch.cuda.current_stream(), grid, array_bitcast, (x, y, tile))

    else:
        ct.launch(torch.cuda.current_stream(), grid, array_bitcast, (x, y, tile))
        assert_equal(y, ref)


@ct.kernel
def array_astype_bool_to_float(y):
    tx = ct.full((1,), True, dtype=ct.bool_)
    ty = ct.astype(tx, np.float32)
    ct.store(y, index=(0,), tile=ty)


def test_astype_bool_to_float():
    x = torch.zeros((1,), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), array_astype_bool_to_float, (x,))
    ref = torch.ones((1,), dtype=torch.float32, device='cuda')
    assert_equal(x, ref)


@ct.kernel
def scalar_astype(scalar, array_out):
    x = ct.astype(scalar, array_out.dtype)
    ct.store(array_out, (0,), x)


def test_astype_scalar():
    x = torch.zeros((1,), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,),
              scalar_astype, (5, x,))
    ref = torch.full((1,), 5, dtype=torch.float32, device='cuda')
    assert_equal(x, ref)


def make_array_astype_kernel(to_dtype):
    @ct.kernel
    def kernel(x, y, TILE: ct.Constant[int]):
        bid = ct.bid(0)
        tx = ct.load(x, index=(bid,), shape=(TILE,))
        ty = ct.astype(tx, to_dtype)
        ct.store(y, index=(bid,), tile=ty)
    return kernel


@pytest.mark.parametrize("from_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
@pytest.mark.parametrize("to_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
def test_array_astype(shape, tile, from_dtype, to_dtype):
    x = make_tensor(shape, dtype=from_dtype, device='cuda') * 5
    # Make the second half of the array 0 to test truncation
    x[x.numel()//2:] = 0
    y = torch.zeros_like(x, dtype=to_dtype)
    grid = (ceil(x.numel() / tile), 1, 1)

    array_astype = make_array_astype_kernel(to_dtype)
    ct.launch(torch.cuda.current_stream(), grid, array_astype, (x, y, tile))
    assert_equal(y, x.to(y.dtype))
