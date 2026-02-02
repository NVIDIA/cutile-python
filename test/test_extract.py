# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest
import torch

from math import ceil
import cuda.tile as ct
from util import assert_equal
from conftest import float_dtypes, dtype_id
from torch.testing import make_tensor
from cuda.tile._exception import TileTypeError


@ct.kernel
def extract_1d(x, y, TILE: ct.Constant[int], use_method: ct.Constant[bool]):
    tx = ct.load(x, index=0, shape=TILE)
    if use_method:
        tx = tx.extract(index=TILE//2, shape=1) + 5
    else:
        tx = ct.extract(tx, index=TILE//2, shape=()) + 5
    tx = ct.full(TILE, tx.item(), dtype=y.dtype)
    ct.store(y, index=0, tile=tx)


@pytest.mark.parametrize("shape", [(128,)])
@pytest.mark.parametrize("tile", [128])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("use_method", [True, False])
def test_extract_1d(shape, dtype, tile, use_method):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, extract_1d, (x, y, tile, use_method))
    ref = torch.full((tile,), x[tile//2] + 5, dtype=dtype, device=x.device)
    assert_equal(y, ref)


@ct.kernel
def extract_2d(x, y,
               TILE_X: ct.Constant[int],
               TILE_Y: ct.Constant[int]):
    tx = ct.load(x, index=(0, 0), shape=(TILE_X, TILE_Y))
    tx_1 = ct.extract(tx, index=(0, 0), shape=(TILE_X//2, TILE_Y))
    tx_2 = ct.extract(tx, index=(1, 0), shape=(TILE_X//2, TILE_Y))
    tx_1 += 5.0
    tx_2 -= 3.0
    tx = ct.cat((tx_1, tx_2), axis=0)
    ct.store(y, index=(0, 0), tile=tx)


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("tile", [(128, 128)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_extract_2d(shape, dtype, tile):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (*(ceil(i / j) for i, j in zip(shape, tile)), 1)
    ct.launch(torch.cuda.current_stream(), grid, extract_2d, (x, y, tile[0], tile[1]))
    ref1 = x[:x.shape[0]//2] + 5.0
    ref2 = x[x.shape[0]//2:] - 3.0
    ref = torch.concatenate((ref1, ref2), 0)
    assert_equal(y, ref)


@ct.kernel
def extract_1d_non_scalar_item(x, y, TILE: ct.Constant[int]):
    tx = ct.load(x, index=(0,), shape=(TILE,))
    tx = ct.extract(tx, index=(TILE//2,), shape=(2,)) + 5
    tx = ct.full((TILE,), tx.item(), dtype=y.dtype)
    ct.store(y, index=(0,), tile=tx)


@pytest.mark.parametrize("shape", [(128,)])
@pytest.mark.parametrize("tile", [128])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_extract_1d_non_scalar_item(shape, dtype, tile):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    with pytest.raises(TileTypeError, match=re.escape("Cannot reshape (2,) to ()")):
        ct.launch(torch.cuda.current_stream(), grid, extract_1d_non_scalar_item, (x, y, tile))
