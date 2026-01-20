# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from cuda.tile._exception import TileTypeError
from util import assert_equal
from conftest import arithmetic_dtypes, dtype_id
from torch.testing import make_tensor


ConstInt = ct.Constant[int]


@ct.kernel
def slice_copy_1d(x, y, start: int, stop: int, TILE: ConstInt):
    sub_x = x.slice(0, start, stop)
    sub_y = y.slice(-1, start, stop)
    tile = ct.load(sub_x, index=(0,), shape=(TILE,))
    ct.store(sub_y, index=(0,), tile=tile)


@pytest.mark.parametrize("shape", [(16,), (64,)])
@pytest.mark.parametrize("dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("start,stop", [(2, 7), (4, 4), (4, 5)],
                         ids=["small_slice", "empty_slice", "single_element_slice"])
def test_slice_1d(shape, dtype, start, stop):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), slice_copy_1d, (x, y, start, stop, 8))
    expected = torch.zeros_like(x)
    expected[start:stop] = x[start:stop]
    assert_equal(y, expected)


@ct.kernel
def slice_copy_2d(x, y, start: int, stop: int, TILE_M: ConstInt, TILE_N: ConstInt):
    sub_x = x.slice(axis=1, start=start, stop=stop)
    sub_y = y.slice(axis=1, start=start, stop=stop)
    m, n = sub_x.shape
    for i in range(ct.cdiv(m, TILE_M)):
        for j in range(ct.cdiv(n, TILE_N)):
            tile = ct.load(sub_x, index=(i, j), shape=(TILE_M, TILE_N))
            ct.store(sub_y, index=(i, j), tile=tile)


@pytest.mark.parametrize("shape", [(12, 16), (64, 128)])
@pytest.mark.parametrize("dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_slice_2d(shape, dtype, noncontiguous):
    x = make_tensor(shape, dtype=dtype, device='cuda', noncontiguous=noncontiguous)
    y = torch.zeros_like(x)
    start, stop = 4, 15
    tile_m, tile_n = 4, 8
    ct.launch(torch.cuda.current_stream(), (1,), slice_copy_2d,
              (x, y, start, stop, tile_m, tile_n))
    expected = torch.zeros_like(x)
    expected[:, start:stop] = x[:, start:stop]
    assert_equal(y, expected)


@ct.kernel
def slice_copy_3d(x, y, start: int, stop: int,
                  TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    sub_x = x.slice(axis=1, start=start, stop=stop)
    sub_y = y.slice(axis=1, start=start, stop=stop)
    m, n, k = sub_x.shape
    for i in range(ct.cdiv(m, TILE_M)):
        for j in range(ct.cdiv(n, TILE_N)):
            for kk in range(ct.cdiv(k, TILE_K)):
                tile = ct.load(sub_x, (i, j, kk), (TILE_M, TILE_N, TILE_K))
                ct.store(sub_y, (i, j, kk), tile)


@pytest.mark.parametrize("shape", [(8, 16, 32), (16, 32, 64)])
@pytest.mark.parametrize("dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_slice_3d(shape, dtype, noncontiguous):
    x = make_tensor(shape, dtype=dtype, device='cuda', noncontiguous=noncontiguous)
    y = torch.zeros_like(x)
    start, stop = 4, 14
    tile_m, tile_n, tile_k = 4, 4, 8
    ct.launch(torch.cuda.current_stream(), (1,), slice_copy_3d,
              (x, y, start, stop, tile_m, tile_n, tile_k))
    expected = torch.zeros_like(x)
    expected[:, start:stop, :] = x[:, start:stop, :]
    assert_equal(y, expected)


@ct.kernel
def ragged_copy_2d(A, B, indptr, TILE_M: ConstInt, TILE_N: ConstInt):
    seg_id = ct.bid(0)

    start = ct.load(indptr, index=seg_id, shape=())
    end = ct.load(indptr, index=seg_id + 1, shape=())

    sub_A = A.slice(axis=0, start=start, stop=end)
    sub_B = B.slice(axis=0, start=start, stop=end)

    m = end - start
    j = ct.bid(1)
    for i in range(ct.cdiv(m, TILE_M)):
        tile = ct.load(sub_A, (i, j), (TILE_M, TILE_N))
        ct.store(sub_B, (i, j), tile)


@pytest.mark.parametrize("dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_ragged_copy_2d(dtype, noncontiguous):
    # 2D array with ragged segments along axis 0
    # Segments: [0,4), [4,7), [7,12)
    M, N = 12, 16
    A = make_tensor((M, N), dtype=dtype, device='cuda', noncontiguous=noncontiguous)
    B = torch.zeros_like(A)
    indptr = torch.tensor([0, 4, 7, 12], dtype=torch.int32, device="cuda")

    tile_m, tile_n = 4, 8
    num_segments = 3
    grid = (num_segments, ct.cdiv(N, tile_n), 1)
    ct.launch(torch.cuda.current_stream(), grid, ragged_copy_2d, (A, B, indptr, tile_m, tile_n))
    assert_equal(B, A)


@ct.kernel
def chained_slice_copy(x, y, TILE: ConstInt):
    # x[4:16][2:8] -> y[6:12]
    sub1 = x.slice(axis=0, start=4, stop=16)
    sub2 = sub1.slice(axis=0, start=2, stop=ct.astype(8, ct.int64))
    sub_y = y.slice(axis=0, start=ct.astype(6, ct.int8), stop=12)
    tile = ct.load(sub2, (0,), (TILE,))
    ct.store(sub_y, (0,), tile)


@pytest.mark.parametrize("dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_chained_slice(dtype, noncontiguous):
    x = make_tensor((20,), dtype=dtype, device='cuda', noncontiguous=noncontiguous)
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), chained_slice_copy, (x, y, 8))
    expected = torch.zeros_like(x)
    expected[6:12] = x[6:12]
    assert_equal(y, expected)


@ct.kernel
def slice_unsigned_index(A, start: int):
    u_start = ct.astype(start, ct.uint32)
    A.slice(axis=0, start=u_start, stop=5)


@ct.kernel
def slice_float_index(A):
    A.slice(axis=0, start=1.0, stop=5.0)


@pytest.mark.parametrize("kernel,args", [
    (slice_unsigned_index, lambda A: (A, 0)),
    (slice_float_index, lambda A: (A,)),
], ids=["unsigned_index", "float_index"])
def test_invalid_index_type(kernel, args):
    A = torch.zeros((10,), dtype=torch.float32, device="cuda")
    match = "Expected a signed integer scalar or a 0D signed integer tile"
    with pytest.raises(TileTypeError, match=match):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, args(A))


@ct.kernel
def slice_axis_oob(A, axis: ConstInt):
    A.slice(axis=axis, start=0, stop=5)


@pytest.mark.parametrize("axis", [1, -2])
def test_axis_out_of_bounds(axis):
    A = torch.zeros((10,), dtype=torch.float32, device="cuda")
    with pytest.raises(TileTypeError, match=f"Axis {axis} is out of range for rank 1'"):
        ct.launch(torch.cuda.current_stream(), (1,), slice_axis_oob, (A, axis))


@ct.kernel
def slice_negative_start(A):
    A.slice(axis=0, start=-1, stop=5)


@ct.kernel
def slice_negative_stop(A):
    A.slice(axis=0, start=0, stop=-1)


@ct.kernel
def slice_stop_less_than_start(A):
    A.slice(axis=0, start=5, stop=3)


@pytest.mark.parametrize("kernel,match", [
    (slice_negative_start, "Slice start must be non-negative"),
    (slice_negative_stop, "Slice stop must be non-negative"),
    (slice_stop_less_than_start, "Slice stop must be greater than or equal to start"),
], ids=["negative_start", "negative_stop", "stop_less_than_start"])
def test_invalid_literal_slice_bounds(kernel, match):
    A = torch.zeros((10,), dtype=torch.float32, device="cuda")
    with pytest.raises(TileTypeError, match=match):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (A,))
