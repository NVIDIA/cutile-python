# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating simple vector addition.
Shows how to perform elementwise operations on vectors.
"""

import cupy as cp
import numpy as np
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    # Get the 1D pid
    pid = ct.bid(0)

    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Perform elementwise addition
    result = a_tile + b_tile

    # Store result
    ct.store(c, index=(pid, ), tile=result)


def test():
    # Create input data
    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    rng = cp.random.default_rng()
    a = rng.random(vector_size)
    b = rng.random(vector_size)
    c = cp.zeros_like(a)

    # Launch kernel
    ct.launch(cp.cuda.get_current_stream(),
              grid,  # 1D grid of processors
              vector_add,
              (a, b, c, tile_size))

    # Copy to host only to compare
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    # Verify results
    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)

    print("✓ vector_add_example passed!")


if __name__ == "__main__":
    test()
