# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

#example-begin
import cuda.tile as ct
import cupy

TILE_SIZE = 16

# cuTile kernel for adding two dense vectors. It runs in parallel on the GPU.
@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

# Host-side function that launches the above kernel.
def vector_add(a: cupy.ndarray, b: cupy.ndarray, result: cupy.ndarray):
    assert a.shape == b.shape == result.shape
    grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
    ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))
#example-end


import numpy as np

def test_vector_add():
    rng = cupy.random.default_rng()
    a = rng.random(128)
    b = rng.random(128)
    result = cupy.zeros_like(a)

    vector_add(a, b, result)

    a_np = cupy.asnumpy(a)
    b_np = cupy.asnumpy(b)
    result_np = cupy.asnumpy(result)

    expected = a_np + b_np
    np.testing.assert_array_almost_equal(result_np, expected)
