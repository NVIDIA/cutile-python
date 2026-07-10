# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest
import torch

from math import ceil
import cuda.tile as ct
from util import assert_equal
from conftest import float_dtypes, dtype_id, requires_tileiras
from torch.testing import make_tensor
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._exception import TileTypeError


pytestmark = requires_tileiras(BytecodeVersion.V_13_4)


@ct.kernel
def insert_1d(x, y, TILE: ct.Constant[int], SUB: ct.Constant[int], IDX: ct.Constant[int],
              use_method: ct.Constant[bool]):
    tx = ct.load(x, index=0, shape=TILE)
    sub = ct.extract(tx, index=IDX, shape=SUB) + 5.0
    if use_method:
        tx = tx.insert(IDX, sub)
    else:
        tx = ct.insert(tx, IDX, sub)
    ct.store(y, index=0, tile=tx)


@pytest.mark.parametrize("shape", [(128,)])
@pytest.mark.parametrize("tile", [128])
@pytest.mark.parametrize("sub", [32])
@pytest.mark.parametrize("idx", [0, 3])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("use_method", [True, False])
def test_insert_1d(shape, dtype, tile, sub, idx, use_method):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, insert_1d, (x, y, tile, sub, idx, use_method))
    ref = x.clone()
    ref[idx * sub:(idx + 1) * sub] += 5.0
    assert_equal(y, ref)


@ct.kernel
def insert_2d(x, y,
              TILE_X: ct.Constant[int], TILE_Y: ct.Constant[int],
              SUB_X: ct.Constant[int], SUB_Y: ct.Constant[int],
              IX: ct.Constant[int], IY: ct.Constant[int]):
    tx = ct.load(x, index=(0, 0), shape=(TILE_X, TILE_Y))
    sub = ct.extract(tx, index=(IX, IY), shape=(SUB_X, SUB_Y)) + 5.0
    tx = ct.insert(tx, (IX, IY), sub)
    ct.store(y, index=(0, 0), tile=tx)


@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_insert_2d(dtype):
    tile = (128, 128)
    sub = (64, 32)
    ix, iy = 1, 2
    x = make_tensor(tile, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, insert_2d,
              (x, y, tile[0], tile[1], sub[0], sub[1], ix, iy))
    ref = x.clone()
    ref[ix * sub[0]:(ix + 1) * sub[0], iy * sub[1]:(iy + 1) * sub[1]] += 5.0
    assert_equal(y, ref)


@ct.kernel
def insert_0d(x, y, TILE: ct.Constant[int], IDX: ct.Constant[int]):
    tx = ct.load(x, index=0, shape=TILE)
    s = ct.extract(tx, index=IDX, shape=()) + 5.0
    tx = ct.insert(tx, IDX, s)
    ct.store(y, index=0, tile=tx)


@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_insert_0d(dtype):
    tile = 128
    idx = 7
    x = make_tensor((tile,), dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1, 1, 1), insert_0d, (x, y, tile, idx))
    ref = x.clone()
    ref[idx] += 5.0
    assert_equal(y, ref)


@ct.kernel
def insert_dynamic_index(x, idx, y, TILE: ct.Constant[int], SUB: ct.Constant[int]):
    tx = ct.load(x, index=0, shape=TILE)
    i = ct.load(idx, index=0, shape=())
    sub = ct.extract(tx, index=i, shape=SUB) + 5.0
    tx = ct.insert(tx, i, sub)
    ct.store(y, index=0, tile=tx)


@pytest.mark.parametrize("idx_val", [0, 2, 3])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_insert_dynamic_index(dtype, idx_val):
    tile, sub = 128, 32
    x = make_tensor((tile,), dtype=dtype, device='cuda')
    idx = torch.tensor([idx_val], dtype=torch.int32, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1, 1, 1), insert_dynamic_index,
              (x, idx, y, tile, sub))
    ref = x.clone()
    ref[idx_val * sub:(idx_val + 1) * sub] += 5.0
    assert_equal(y, ref)


@ct.kernel
def insert_dtype_mismatch(x, y, TILE: ct.Constant[int]):
    tx = ct.load(x, index=0, shape=TILE)
    sub = ct.full((TILE // 2,), 1, dtype=ct.int32)
    tx = ct.insert(tx, 0, sub)
    ct.store(y, index=0, tile=tx)


def test_insert_dtype_mismatch():
    x = make_tensor((128,), dtype=torch.float16, device='cuda')
    y = torch.zeros_like(x)
    with pytest.raises(TileTypeError, match="Cannot insert a tile of dtype"):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), insert_dtype_mismatch, (x, y, 128))


@ct.kernel
def insert_non_divisible(x, y, TILE: ct.Constant[int]):
    tx = ct.load(x, index=0, shape=TILE)
    # A value larger than the destination cannot divide it: TILE % (2 * TILE) != 0.
    sub = ct.full((2 * TILE,), 1.0, dtype=x.dtype)
    tx = ct.insert(tx, 0, sub)
    ct.store(y, index=0, tile=tx)


def test_insert_non_divisible():
    x = make_tensor((128,), dtype=torch.float16, device='cuda')
    y = torch.zeros_like(x)
    with pytest.raises(TileTypeError, match="not divisible by"):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), insert_non_divisible, (x, y, 128))


@ct.kernel
def insert_oob(x, y, TILE: ct.Constant[int]):
    tx = ct.load(x, index=0, shape=TILE)
    sub = ct.full((TILE // 2,), 1.0, dtype=x.dtype)
    # only 2 subtiles; index 2 is out of bounds
    tx = ct.insert(tx, 2, sub)
    ct.store(y, index=0, tile=tx)


def test_insert_oob():
    x = make_tensor((128,), dtype=torch.float16, device='cuda')
    y = torch.zeros_like(x)
    with pytest.raises(TileTypeError, match="out of bounds"):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), insert_oob, (x, y, 128))


@ct.kernel
def insert_rank_mismatch(x, y, TILE_X: ct.Constant[int], TILE_Y: ct.Constant[int]):
    tx = ct.load(x, index=(0, 0), shape=(TILE_X, TILE_Y))
    sub = ct.full((TILE_X,), 1.0, dtype=x.dtype)
    tx = ct.insert(tx, (0, 0), sub)
    ct.store(y, index=(0, 0), tile=tx)


def test_insert_rank_mismatch():
    x = make_tensor((128, 128), dtype=torch.float16, device='cuda')
    y = torch.zeros_like(x)
    with pytest.raises(TileTypeError, match=re.escape("does not match the tile rank")):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), insert_rank_mismatch, (x, y, 128, 128))
