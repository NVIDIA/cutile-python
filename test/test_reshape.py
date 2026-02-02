# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from math import ceil
import cuda.tile as ct
from util import assert_equal, jit_kernel
from conftest import float_dtypes, dtype_id
from torch.testing import make_tensor
from cuda.tile._exception import TileTypeError


@ct.kernel
def expanded_copy(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    expanded = ct.expand_dims(tx, -1)
    ct.store(y, index=(bid, 0), tile=expanded)


@pytest.mark.parametrize("shape", [(128,)])
@pytest.mark.parametrize("tile", [128])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_expand_dims(shape, dtype, tile):
    x = make_tensor(shape, dtype=dtype, device="cuda")
    y = torch.zeros_like(x).unsqueeze(-1)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, expanded_copy, (x, y, tile))
    assert_equal(y, x.unsqueeze(-1))


# === Helpers ===
new_axis_indexing_kernel_template = """
def {name}(x, y, TILE_0: ct.Constant[int], TILE_1: ct.Constant[int]):
    bid0 = ct.bid(0)
    bid1 = ct.bid(1)
    tx = ct.load(x, index=(bid0, bid1), shape=(TILE_0, TILE_1))
    expanded = {expand_expr}
    ct.store(y, index={index_expr}, tile=expanded)"""


@pytest.mark.parametrize("kernel_axes", [
    ("tx[None, :, :]", "(0, bid0, bid1)", lambda x: x.unsqueeze(0)),
    ("tx[:, None, :]", "(bid0, 0, bid1)", lambda x: x.unsqueeze(1)),
    ("tx[:, :, np.newaxis]", "(bid0, bid1, 0)", lambda x: x.unsqueeze(-1)),
    ("tx[np.newaxis, np.newaxis, :, :]", "(0, 0, bid0, bid1)",
     lambda x: x.unsqueeze(0).unsqueeze(0)),
    ("tx[:, None, None, :]", "(bid0, 0, 0, bid1)", lambda x: x.unsqueeze(1).unsqueeze(1)),
    ("tx[:, :]", "(bid0, bid1)", lambda x: x),
    ("tx[..., None]", "(bid0, 0, bid1)", lambda x: x.unsqueeze(-1)),
    ("tx[None, ...]", "(0, bid0, bid1)", lambda x: x.unsqueeze(0)),
    ("tx[None, ..., None]", "(0, bid0, bid1, 0)", lambda x: x.unsqueeze(0).unsqueeze(-1)),
    ("tx[None]", "(0, bid0, bid1)", lambda x: x.unsqueeze(0)),
])
def test_new_axis_indexing(kernel_axes, tmp_path):
    expand_expr, index_expr, expected_fn = kernel_axes
    shape = (128, 128)
    tile = (128, 128)
    x = make_tensor(shape, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(expected_fn(x))
    grid = (ceil(shape[0] / tile[0]), ceil(shape[1] / tile[1]), 1)
    source = new_axis_indexing_kernel_template.format(name="new_axis_indexing",
                                                      expand_expr=expand_expr,
                                                      index_expr=index_expr)
    kernel = jit_kernel("new_axis_indexing", source, tmp_path, {"np": np})
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile[0], tile[1]))
    assert_equal(y, expected_fn(x))


@pytest.mark.parametrize("expand_expr", ["tx[0]", "tx[0, 1]", "tx[:]", "tx[::2]"])
def test_invalid_new_axis_indexing(expand_expr, tmp_path):
    shape = (128, 128)
    tile = (128, 128)
    x = make_tensor(shape, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile[0]), ceil(shape[1] / tile[1]), 1)
    source = new_axis_indexing_kernel_template.format(name="new_axis_indexing",
                                                      expand_expr=expand_expr,
                                                      index_expr="(bid0, bid1)")
    kernel = jit_kernel("new_axis_indexing", source, tmp_path, {"np": np})
    error_type = TileTypeError
    with pytest.raises(error_type, match=".*Directly indexing a tile is not supported.*"):
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile[0], tile[1]))


@ct.kernel
def reshape_copy(x, y,
                 R_TILE: ct.Constant[int],
                 C_TILE: ct.Constant[int],
                 use_method: ct.Constant[bool]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(R_TILE * C_TILE,))
    if use_method:
        reshaped = tx.reshape(shape=(R_TILE, C_TILE))
    else:
        reshaped = ct.reshape(tx, (R_TILE, C_TILE))
    ct.store(y, index=(bid, 0), tile=reshaped)


@pytest.mark.parametrize("r_tile", [128])
@pytest.mark.parametrize("c_tile", [128])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("use_method", [True, False])
def test_reshape_copy(dtype, r_tile, c_tile, use_method):
    x = make_tensor((r_tile * c_tile,), dtype=dtype, device="cuda")
    y = torch.zeros((r_tile, c_tile), dtype=dtype, device=x.device)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, reshape_copy, (x, y, r_tile, c_tile, use_method))
    assert_equal(y, torch.reshape(x, (r_tile, c_tile)))


@ct.kernel
def reshape_implicit_dim(x, y,
                         R_TILE: ct.Constant[int],
                         C_TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(R_TILE * C_TILE,))
    reshaped = ct.reshape(tx, (R_TILE, -1))
    ct.store(y, index=(bid, 0), tile=reshaped)


@pytest.mark.parametrize("r_tile", [128])
@pytest.mark.parametrize("c_tile", [128])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_reshape_implicit_dim(dtype, r_tile, c_tile):
    x = make_tensor((r_tile * c_tile,), dtype=dtype, device="cuda")
    y = torch.zeros((r_tile, c_tile), dtype=dtype, device=x.device)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, reshape_implicit_dim, (x, y, r_tile, c_tile))
    assert_equal(y, torch.reshape(x, (r_tile, c_tile)))


@ct.kernel
def reshape_more_than_one_dim_negative_one(x, y,
                                           R_TILE: ct.Constant[int],
                                           C_TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(R_TILE * C_TILE,))
    reshaped = ct.reshape(tx, (-1, -1))
    ct.store(y, index=(bid, 0), tile=reshaped)


def test_reshape_more_than_one_negative_one():
    r_tile, c_tile = 128, 128
    dtype = torch.float32
    x = make_tensor((r_tile * c_tile,), dtype=dtype, device="cuda")
    y = torch.zeros((r_tile, c_tile), dtype=dtype, device=x.device)
    grid = (1, 1, 1)
    with pytest.raises(TileTypeError, match="Only one dimension can be -1"):
        ct.launch(torch.cuda.current_stream(), grid, reshape_more_than_one_dim_negative_one,
                  (x, y, r_tile, c_tile))


def test_reshape_scalar():
    @ct.kernel
    def kernel(x):
        tx = ct.reshape(4, (1, 1, 1))
        ct.store(x, (0, 0, 0), tx)

    x = torch.zeros((1, 1, 1), device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 4
