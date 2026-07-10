# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import torch


def test_matmul_sharedmem():
    '''
    https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMul/matrixMul.cu
    '''
    tile_width = 16
    m = 32
    k = 32
    n = 32

    @cl.kernel
    def matrix_mul_shared(C, A, B, wA: cl.Constant[int], wB: cl.Constant[int]):
        ds_A = cl.shared_array(shape=(tile_width, tile_width), dtype=cl.float32)
        ds_B = cl.shared_array(shape=(tile_width, tile_width), dtype=cl.float32)

        bx, by = cl.block_index(0), cl.block_index(1)
        tx, ty = cl.thread_index(0), cl.thread_index(1)

        row = by * tile_width + ty
        col = bx * tile_width + tx
        p_value = cl.float32(0.0)

        for tile_idx in cl.static_iter(range(wA // tile_width)):
            ds_A[ty, tx] = A[row * wA + (tile_idx * tile_width + tx)]
            ds_B[ty, tx] = B[(tile_idx * tile_width + ty) * wB + col]
            cl.barrier_sync_block()

            for kk in cl.static_iter(range(tile_width)):
                p_value = p_value + ds_A[ty, kk] * ds_B[kk, tx]

            cl.barrier_sync_block()

        C[row * wB + col] = p_value

    A = torch.arange(m * k, dtype=torch.float32, device="cuda").reshape(m, k) / 8.0
    B = torch.arange(k * n, dtype=torch.float32, device="cuda").reshape(k, n) / 16.0
    C = torch.zeros(m * n, dtype=torch.float32, device="cuda")

    grid = (n // tile_width, m // tile_width)
    block = (tile_width, tile_width)
    cl.launch(
        torch.cuda.current_stream(),
        grid,
        block,
        matrix_mul_shared,
        (C, A.reshape(m * k), B.reshape(k * n), k, n),
    )

    expected = torch.matmul(A, B)
    assert torch.allclose(C.reshape(m, n).cpu(), expected.cpu(), atol=1e-5, rtol=1e-5)
