# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import pytest
import torch

__doc__ = """
Port of the transpose examples from the cuda-samples repository:
https://github.com/NVIDIA/cuda-samples/blob/master/Samples/6_Performance/transpose/transpose.cu

The original test program seems to be from here:
https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu

but slightly cleaned up and with more documentation.

Note this comment in the updated source:

// --------------------------------------------------------------------
// Partial transposes
// NB: the coarse- and fine-grained routines only perform part of a
//     transpose and will fail the test against the reference solution
//
//     They are used to assess performance characteristics of different
//     components of a full transpose
// --------------------------------------------------------------------

For this reason, we do not validate the results of those kernels, just
ensure they compile and execute.
"""


TILE_DIM = 32
BLOCK_ROWS = 16
MATRIX_SIZE_X = 1024
MATRIX_SIZE_Y = 1024
NUM_REPS = 100


@cl.kernel
def copy(odata, idata, width: cl.Constant[int], height: cl.Constant[int]):
    bx, by = cl.block_index(0), cl.block_index(1)
    tx, ty = cl.thread_index(0), cl.thread_index(1)
    xIndex = bx * TILE_DIM + tx
    yIndex = by * TILE_DIM + ty
    index = xIndex + width * yIndex
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        odata[index + i * width] = idata[index + i * width]


@cl.kernel
def copySharedMem(odata, idata, width: cl.Constant[int], height: cl.Constant[int]):
    tile = cl.shared_array(shape=(TILE_DIM, TILE_DIM), dtype=cl.float32)
    bx, by = cl.block_index(0), cl.block_index(1)
    tx, ty = cl.thread_index(0), cl.thread_index(1)
    xIndex = bx * TILE_DIM + tx
    yIndex = by * TILE_DIM + ty
    index = xIndex + width * yIndex
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        if xIndex < width and yIndex < height:
            tile[ty + i, tx] = idata[index + i * width]
            tile[ty + i, tx] = idata[index + i * width]
    cl.barrier_sync_block()
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        if xIndex < height and yIndex < width:
            odata[index + i * width] = tile[ty + i, tx]
            odata[index + i * width] = tile[ty + i, tx]


@cl.kernel
def transposeNaive(odata, idata, width: cl.Constant[int], height: cl.Constant[int]):
    bx, by = cl.block_index(0), cl.block_index(1)
    tx, ty = cl.thread_index(0), cl.thread_index(1)
    xIndex = bx * TILE_DIM + tx
    yIndex = by * TILE_DIM + ty
    index_in = xIndex + width * yIndex
    index_out = yIndex + height * xIndex
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        odata[index_out + i] = idata[index_in + i * width]


@cl.kernel
def transposeCoalesced(odata, idata, width: cl.Constant[int], height: cl.Constant[int]):
    tile = cl.shared_array(shape=(TILE_DIM, TILE_DIM), dtype=cl.float32)
    bx, by = cl.block_index(0), cl.block_index(1)
    tx, ty = cl.thread_index(0), cl.thread_index(1)
    xIndex = bx * TILE_DIM + tx
    yIndex = by * TILE_DIM + ty
    index_in = xIndex + yIndex * width
    xIndex = by * TILE_DIM + tx
    yIndex = bx * TILE_DIM + ty
    index_out = xIndex + yIndex * height
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        tile[ty + i, tx] = idata[index_in + i * width]
    cl.barrier_sync_block()
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        odata[index_out + i * height] = tile[tx, ty + i]


@cl.kernel
def transposeNoBankConflicts(
    odata, idata, width: cl.Constant[int], height: cl.Constant[int]
):
    tile = cl.shared_array(shape=(TILE_DIM, TILE_DIM + 1), dtype=cl.float32)
    bx, by = cl.block_index(0), cl.block_index(1)
    tx, ty = cl.thread_index(0), cl.thread_index(1)
    xIndex = bx * TILE_DIM + tx
    yIndex = by * TILE_DIM + ty
    index_in = xIndex + yIndex * width
    xIndex = by * TILE_DIM + tx
    yIndex = bx * TILE_DIM + ty
    index_out = xIndex + yIndex * height
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        tile[ty + i, tx] = idata[index_in + i * width]
    cl.barrier_sync_block()
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        odata[index_out + i * height] = tile[tx, ty + i]


@cl.kernel
def transposeDiagonal(odata, idata, width: cl.Constant[int], height: cl.Constant[int]):
    tile = cl.shared_array(shape=(TILE_DIM, TILE_DIM + 1), dtype=cl.float32)
    bx, by = cl.block_index(0), cl.block_index(1)
    tx, ty = cl.thread_index(0), cl.thread_index(1)
    gx, gy = cl.block_count(0), cl.block_count(1)
    if width == height:
        blockIdx_y = bx
        blockIdx_x = (bx + by) % gx
    else:
        bid = bx + gx * by
        blockIdx_y = bid % gy
        blockIdx_x = ((bid // gy) + blockIdx_y) % gx
    xIndex = blockIdx_x * TILE_DIM + tx
    yIndex = blockIdx_y * TILE_DIM + ty
    index_in = xIndex + yIndex * width
    xIndex = blockIdx_y * TILE_DIM + tx
    yIndex = blockIdx_x * TILE_DIM + ty
    index_out = xIndex + yIndex * height
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        tile[ty + i, tx] = idata[index_in + i * width]
    cl.barrier_sync_block()
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        odata[index_out + i * height] = tile[tx, ty + i]


@cl.kernel
def transposeFineGrained(
    odata, idata, width: cl.Constant[int], height: cl.Constant[int]
):
    block = cl.shared_array(shape=(TILE_DIM, TILE_DIM + 1), dtype=cl.float32)
    bx, by = cl.block_index(0), cl.block_index(1)
    tx, ty = cl.thread_index(0), cl.thread_index(1)
    xIndex = bx * TILE_DIM + tx
    yIndex = by * TILE_DIM + ty
    index = xIndex + yIndex * width
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        block[ty + i, tx] = idata[index + i * width]
    cl.barrier_sync_block()
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        odata[index + i * height] = block[tx, ty + i]


@cl.kernel
def transposeCoarseGrained(
    odata, idata, width: cl.Constant[int], height: cl.Constant[int]
):
    block = cl.shared_array(shape=(TILE_DIM, TILE_DIM + 1), dtype=cl.float32)
    bx, by = cl.block_index(0), cl.block_index(1)
    tx, ty = cl.thread_index(0), cl.thread_index(1)
    xIndex = bx * TILE_DIM + tx
    yIndex = by * TILE_DIM + ty
    index_in = xIndex + yIndex * width
    xIndex = by * TILE_DIM + tx
    yIndex = bx * TILE_DIM + ty
    index_out = xIndex + yIndex * height
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        block[ty + i, tx] = idata[index_in + i * width]
    cl.barrier_sync_block()
    for i in cl.static_iter(range(0, TILE_DIM, BLOCK_ROWS)):
        odata[index_out + i * height] = block[ty + i, tx]


def computeTransposeGold(idata, size_x, size_y):
    gold = torch.empty_like(idata)
    for y in range(size_y):
        for x in range(size_x):
            gold[(x * size_y) + y] = idata[(y * size_x) + x]
    return gold


@pytest.mark.parametrize(
    "kernel",
    (
        copy,
        copySharedMem,
        transposeNaive,
        transposeCoalesced,
        transposeNoBankConflicts,
        transposeCoarseGrained,
        transposeFineGrained,
        transposeDiagonal,
    ),
)
def test_transpose(kernel):
    size_x, size_y = MATRIX_SIZE_X, MATRIX_SIZE_Y
    grid = (size_x // TILE_DIM, size_y // TILE_DIM)
    threads = (TILE_DIM, BLOCK_ROWS)
    h_idata = torch.arange(size_x * size_y, dtype=torch.float32)
    h_odata = torch.zeros_like(h_idata)
    transposeGold = computeTransposeGold(h_idata, size_x, size_y)
    d_idata = h_idata.cuda()
    d_odata = h_odata.cuda()

    if kernel == copy or kernel == copySharedMem:
        gold = h_idata
    elif kernel == transposeCoarseGrained or kernel == transposeFineGrained:
        # fine- and coarse-grained kernels are not full
        # transposes, so don't verify the results.
        gold = h_odata
    else:
        gold = transposeGold

    cl.launch(
        torch.cuda.current_stream(),
        grid,
        threads,
        kernel,
        (d_odata, d_idata, size_x, size_y),
    )
    torch.cuda.synchronize()

    h_odata.copy_(d_odata.cpu())
    assert torch.allclose(gold, h_odata, atol=0.01, rtol=0.0)
