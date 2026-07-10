# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

import cuda.lang as cl


__doc__ = """
Adapted from:
https://github.com/numba/numba-benchmark/blob/master/benchmarks/bench_cuda.py
"""


def run_cpu_nbody(positions, weights, eps_2):
    accelerations = np.zeros_like(positions)
    n = weights.size
    for j in range(n):
        r = positions[j] - positions
        rx = r[:, 0]
        ry = r[:, 1]
        sqr_dist = rx * rx + ry * ry + eps_2
        sixth_dist = sqr_dist * sqr_dist * sqr_dist
        inv_dist_cube = np.float32(1.0) / np.sqrt(sixth_dist)
        s = weights[j] * inv_dist_cube
        accelerations += (r.transpose() * s).transpose()
    return accelerations


@cl.kernel
def calculate_forces(
    positions,
    weights,
    accelerations,
    eps_2,
    n: cl.Constant[int],
    tile_size: cl.Constant[int],
):
    cl.static_assert(n % tile_size == 0, "n must be a multiple of tile_size")

    sh_positions = cl.shared_array(shape=(tile_size, 2), dtype=cl.float32)
    sh_weights = cl.shared_array(shape=(tile_size,), dtype=cl.float32)

    tx = cl.thread_index(0)
    bx = cl.block_index(0)
    bdx = cl.thread_count(0)
    i = bx * bdx + tx

    axi = cl.float32(0.0)
    ayi = cl.float32(0.0)
    xi = positions[i, 0]
    yi = positions[i, 1]

    for tile_start in range(0, n, tile_size):
        index = tile_start + tx
        sh_positions[tx, 0] = positions[index, 0]
        sh_positions[tx, 1] = positions[index, 1]
        sh_weights[tx] = weights[index]
        cl.barrier_sync_block()

        for j in range(tile_size):
            rx = sh_positions[j, 0] - xi
            ry = sh_positions[j, 1] - yi
            sqr_dist = rx * rx + ry * ry + eps_2
            sixth_dist = sqr_dist * sqr_dist * sqr_dist
            s = sh_weights[j] * cl._libdevice.rsqrtf(sixth_dist)
            axi = axi + rx * s
            ayi = ayi + ry * s

        cl.barrier_sync_block()

    accelerations[i, 0] = axi
    accelerations[i, 1] = ayi


@pytest.mark.parametrize("tile_size", (64, 128, 256))
def test_nbody_forces(tile_size):
    n_bodies = 512
    eps_2 = 1.0e-6

    generator = torch.Generator(device="cuda").manual_seed(0)
    positions = (
        2.0
        * torch.rand(
            (n_bodies, 2), dtype=torch.float32, device="cuda", generator=generator
        )
        - 1.0
    )
    weights = 1.0 + torch.rand(
        (n_bodies,), dtype=torch.float32, device="cuda", generator=generator
    )
    accelerations = torch.zeros_like(positions)

    cl.launch(
        torch.cuda.current_stream(),
        (n_bodies // tile_size,),
        (tile_size,),
        calculate_forces,
        (positions, weights, accelerations, eps_2, n_bodies, tile_size),
    )
    torch.cuda.synchronize()

    expected = run_cpu_nbody(positions.cpu().numpy(), weights.cpu().numpy(), eps_2)

    torch.testing.assert_close(
        accelerations.cpu(), torch.from_numpy(expected), atol=1.0e-4, rtol=1.0e-4
    )
