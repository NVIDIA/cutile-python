# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._bytecode.version import BytecodeVersion
import pytest
import torch

from math import ceil
import cuda.tile as ct
from util import assert_equal, make_test_tensor, require_hopper_or_newer, require_blackwell_or_newer
from conftest import (
    float_dtypes, bool_dtypes, int_dtypes, dtype_id, requires_tileiras, uint_dtypes
)
from torch.testing import make_tensor
from cuda.tile import PaddingMode
from typing import Optional

float8_dtypes = [
    pytest.param(torch.float8_e5m2, marks=require_hopper_or_newer()),
    pytest.param(torch.float8_e4m3fn, marks=require_hopper_or_newer()),
    pytest.param(torch.float8_e8m0fnu, marks=(require_blackwell_or_newer(),
                                              requires_tileiras(BytecodeVersion.V_13_2))),
]

test_dtypes = (float_dtypes + bool_dtypes + int_dtypes + uint_dtypes +
               [torch.float64] + float8_dtypes)


def assert_tensors_contiguity(tensors, predicate):
    is_contiguous = all(t.is_contiguous() for t in tensors)
    assert is_contiguous if predicate else not is_contiguous


@ct.kernel
def array_copy_1d(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(y, index=(bid,), tile=tx)


@pytest.mark.parametrize("shape", [(128,), (225,)])
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("stride_step", [1, 2])
@pytest.mark.parametrize("dtype", test_dtypes, ids=dtype_id)
def test_array_copy_1d(shape, stride_step, dtype, tile):
    x = make_test_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x, device='cuda')
    xx = x[::stride_step]
    assert xx.stride() == (stride_step,)
    yy = y[::stride_step]
    assert yy.stride() == (stride_step,)
    assert_tensors_contiguity((xx, yy), stride_step == 1)
    grid = (ceil(len(xx) / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, array_copy_1d, (xx, yy, tile))
    y_expected = torch.zeros_like(x)
    y_expected[::stride_step] = x[::stride_step]
    assert_equal(y, y_expected)


@ct.kernel
def array_copy_2d(x, y, TILE_X: ct.Constant[int], TILE_Y: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    tx = ct.load(x, index=(bidx, bidy), shape=(TILE_X, TILE_Y))
    ct.store(y, index=(bidx, bidy), tile=tx)


@pytest.mark.parametrize("shape", [(128, 128), (192, 192)])
@pytest.mark.parametrize("tile", [(64, 64), (128, 128)])
@pytest.mark.parametrize("permute", [(0, 1), (1, 0)])
@pytest.mark.parametrize("stride_step", [1, 2])
@pytest.mark.parametrize("dtype", test_dtypes, ids=dtype_id)
def test_array_copy_2d(shape, stride_step, permute, dtype, tile):
    x = make_test_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    xx = x[::stride_step, ::stride_step].permute(permute)
    yy = y[::stride_step, ::stride_step].permute(permute)
    assert_tensors_contiguity((xx, yy), stride_step == 1 and permute == (0, 1))
    grid = (*(ceil(i / j) for i, j in zip(xx.shape, tile)), 1)
    ct.launch(torch.cuda.current_stream(), grid, array_copy_2d, (xx, yy, tile[0], tile[1]))
    y_expected = torch.zeros_like(x)
    y_expected[::stride_step, ::stride_step] = x[::stride_step, ::stride_step]
    assert_equal(y, y_expected)


@ct.kernel
def array_copy_3d(x, y,
                  TILE_BATCH: ct.Constant[int],
                  TILE_X: ct.Constant[int],
                  TILE_Y: ct.Constant[int]):
    bidb = ct.bid(0)
    bidx = ct.bid(1)
    bidy = ct.bid(2)
    tx = ct.load(x, index=(bidb, bidx, bidy), shape=(TILE_BATCH, TILE_X, TILE_Y))
    ct.store(y, index=(bidb, bidx, bidy), tile=tx)


@pytest.mark.parametrize("shape", [(4, 128, 128), (2, 192, 192)])
@pytest.mark.parametrize("tile", [(2, 64, 64), (1, 128, 128)])
@pytest.mark.parametrize("permute", [(0, 1, 2), (2, 1, 0), (2, 0, 1)])
@pytest.mark.parametrize("stride_step", [1, 2])
@pytest.mark.parametrize("dtype", test_dtypes, ids=dtype_id)
def test_array_copy_3d(shape, stride_step, permute, dtype, tile):
    x = make_test_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x, device='cuda')
    xx = x[:, :, ::stride_step].permute(permute)
    yy = y[:, :, ::stride_step].permute(permute)
    assert_tensors_contiguity((xx, yy), stride_step == 1 and permute == (0, 1, 2))
    permuted_tile = tile[permute[0]], tile[permute[1]], tile[permute[2]]
    grid = tuple(ceil(i / j) for i, j in zip(xx.shape, permuted_tile))
    ct.launch(torch.cuda.current_stream(), grid, array_copy_3d,
              (xx, yy, permuted_tile[0], permuted_tile[1], permuted_tile[2]))
    y_expected = torch.zeros_like(x)
    y_expected[:, :, ::stride_step] = x[:, :, ::stride_step]
    assert_equal(y, y_expected)


def make_array_copy_2d_with_padding_kernel(padding_mode: Optional[PaddingMode]):

    @ct.kernel
    def kernel(x, y, TILE_X: ct.Constant[int], TILE_Y: ct.Constant[int]):
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tx = ct.load(x, index=(bidx, bidy), shape=(TILE_X, TILE_Y), padding_mode=padding_mode)
        ct.store(y, index=(bidx, bidy), tile=tx)
    return kernel


@pytest.mark.parametrize("padding_value, float_padding_value", [
    (PaddingMode.UNDETERMINED, None),
    (PaddingMode.ZERO, 0.0),
    (PaddingMode.NEG_ZERO, -0.0),
    (PaddingMode.NAN, float('nan')),
    (PaddingMode.POS_INF, float('inf')),
    (PaddingMode.NEG_INF, float('-inf'))
])
def test_array_copy_2d_with_padding(padding_value, float_padding_value):
    shape = (63, 63)
    tile = (64, 64)
    x = make_tensor(shape, dtype=torch.float32, device='cuda')
    y = make_tensor(tile, dtype=torch.float32, device='cuda')
    grid = (ceil(shape[0] / tile[0]), ceil(shape[1] / tile[1]), 1)
    ct.launch(torch.cuda.current_stream(), grid,
              make_array_copy_2d_with_padding_kernel(padding_value),
              (x, y, tile[0], tile[1]))
    if float_padding_value is None:
        assert_equal(y[:shape[0], :shape[1]], x)
    else:
        y_expected = torch.ones(tile, dtype=torch.float32, device='cuda') * float_padding_value
        y_expected[:shape[0], :shape[1]] = x
        assert_equal(y, y_expected)
