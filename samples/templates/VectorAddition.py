# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
import math

from test.kernels.vec_add import (
    vec_add_kernel_1d, vec_add_kernel_1d_gather, vec_add_kernel_2d, vec_add_kernel_2d_gather
)


# --- Wrapper Function to Dispatch to Kernels ---
def vec_add(a: torch.Tensor, b: torch.Tensor, use_gather: bool = False) -> torch.Tensor:
    """
    Performs element-wise addition of two tensors (vector or matrix) using
    different cuTile kernels based on dimensionality and gather/scatter preference.

    This function acts as a high-level interface, handling input validation,
    determining appropriate tile and grid dimensions, and dispatching to
    the correct cuTile kernel.

    Args:
        a (torch.Tensor): The first input tensor. Must be 1D or 2D and on a CUDA device.
        b (torch.Tensor): The second input tensor. Must match 'a' in shape, device, and dtype.
        use_gather (bool): If True, uses kernels with explicit gather/scatter and masking
                           for robust boundary handling (recommended for non-power-of-2
                           or non-tile-aligned dimensions).
                           If False, uses direct tiled loads/stores, which are simpler
                           but assume dimensions are perfectly divisible by tile sizes.

    Returns:
        torch.Tensor: The resulting tensor after element-wise addition.

    Raises:
        ValueError: If input tensors have mismatched shapes, incorrect dimensions,
                    are not on CUDA, or have different data types.
    """
    # --- Input Validation ---
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")
    if a.dim() > 2 or b.dim() > 2:
        raise ValueError("This function currently supports only 1D or 2D tensors.")
    if a.device != b.device:
        raise ValueError("Input tensors must be on the same device.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")
    if a.dtype != b.dtype:
        raise ValueError("Input tensors must have the same data type.")

    # Create an empty output tensor on the same device and with the same dtype as inputs.
    c = torch.empty_like(a)

    # --- Dispatch based on Dimensionality ---
    if a.dim() == 1:
        N = a.shape[0]  # Get the total size of the 1D vector

        # Heuristic for TILE size:
        # Choose a power of 2, up to 1024, that is greater than or equal to N.
        # This helps in efficient memory access patterns on the GPU.
        # Handle N=0 gracefully to avoid log2(0) errors.
        TILE = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1

        # Calculate the grid dimensions for launching the kernel.
        # `math.ceil(N / TILE)` determines the number of blocks needed to cover
        # the entire vector. Each block processes a `TILE`-sized chunk.
        grid = (math.ceil(N / TILE), 1, 1)  # (blocks_x, blocks_y, blocks_z)

        kernel = vec_add_kernel_1d_gather if use_gather else vec_add_kernel_1d
        ct.launch(torch.cuda.current_stream(), grid, kernel, (a, b, c, TILE))
    else:  # a.dim() == 2 (Matrix)
        M, N = a.shape  # Get rows (M) and columns (N) of the matrix

        # Heuristic for 2D TILE sizes:
        # TILE_Y is chosen as a power of 2, up to 1024, based on the column dimension.
        TILE_Y = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1
        # TILE_X is derived to keep the total elements processed by a block
        # (TILE_X * TILE_Y) around 1024 (a common block size limit for threads).
        TILE_X = max(1, 1024 // TILE_Y)

        # Calculate the 2D grid dimensions for launching the kernel.
        # `math.ceil(M / TILE_X)` blocks along rows, `math.ceil(N / TILE_Y)` blocks along columns.
        grid = (math.ceil(M / TILE_X), math.ceil(N / TILE_Y), 1)

        kernel = vec_add_kernel_2d_gather if use_gather else vec_add_kernel_2d
        ct.launch(torch.cuda.current_stream(), grid, kernel, (a, b, c, TILE_X, TILE_Y))

    return c  # Return the computed output tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    print("--- Running cuTile Vector/Matrix Addition Examples ---")

    # --- User Configuration ---
    VECTOR_SIZE_1D = 1_000_000
    MATRIX_SHAPE_2D = (2048, 1024)  # Rows, Columns

    # --- Test Case 1: 1D Vector Add (Direct Tiled) ---
    print("\n--- Test 1: 1D Vector Add (Direct Tiled) ---")
    # Create random input tensors on the CUDA device.
    a_1d_direct = torch.randn(VECTOR_SIZE_1D, dtype=torch.float32, device='cuda')
    b_1d_direct = torch.randn(VECTOR_SIZE_1D, dtype=torch.float32, device='cuda')
    print(f"Input 1D shape: {a_1d_direct.shape}, dtype: {a_1d_direct.dtype}")

    # Call the vec_add wrapper function, requesting the direct tiled kernel.
    c_1d_cutile_direct = vec_add(a_1d_direct, b_1d_direct, use_gather=False)
    print(
        f"""cuTile Output 1D shape: {c_1d_cutile_direct.shape},
        dtype: {c_1d_cutile_direct.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_1d_cutile_direct, a_1d_direct + b_1d_direct)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: 1D Vector Add (Gather/Scatter) ---
    print("\n--- Test 2: 1D Vector Add (Gather/Scatter) ---")
    # Use a size not perfectly divisible by typical TILE_SIZE to demonstrate
    # the gather/scatter kernel's robust boundary handling.
    VECTOR_SIZE_1D_GATHER = 1_000_001
    a_1d_gather = torch.randn(VECTOR_SIZE_1D_GATHER, dtype=torch.float32, device='cuda')
    b_1d_gather = torch.randn(VECTOR_SIZE_1D_GATHER, dtype=torch.float32, device='cuda')
    print(f"Input 1D (gather) shape: {a_1d_gather.shape}, dtype: {a_1d_gather.dtype}")

    # Call the vec_add wrapper function, requesting the gather/scatter kernel.
    c_1d_cutile_gather = vec_add(a_1d_gather, b_1d_gather, use_gather=True)
    print(
        f"""cuTile Output 1D (gather) shape: {c_1d_cutile_gather.shape},
        dtype: {c_1d_cutile_gather.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_1d_cutile_gather, a_1d_gather + b_1d_gather)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 3: 2D Matrix Add (Direct Tiled) ---
    print("\n--- Test 3: 2D Matrix Add (Direct Tiled) ---")
    a_2d_direct = torch.randn(MATRIX_SHAPE_2D, dtype=torch.float32, device='cuda')
    b_2d_direct = torch.randn(MATRIX_SHAPE_2D, dtype=torch.float32, device='cuda')
    print(f"Input 2D shape: {a_2d_direct.shape}, dtype: {a_2d_direct.dtype}")

    # Call the vec_add wrapper function for 2D, requesting the direct tiled kernel.
    c_2d_cutile_direct = vec_add(a_2d_direct, b_2d_direct, use_gather=False)
    print(
        f"""cuTile Output 2D shape: {c_2d_cutile_direct.shape},
        dtype: {c_2d_cutile_direct.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_2d_cutile_direct, a_2d_direct + b_2d_direct)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 4: 2D Matrix Add (Gather/Scatter) ---
    print("\n--- Test 4: 2D Matrix Add (Gather/Scatter) ---")
    # Use dimensions not perfectly divisible by typical tile sizes to demonstrate
    # the gather/scatter kernel's robust boundary handling in 2D.
    MATRIX_SHAPE_2D_GATHER = (2000, 1000)
    a_2d_gather = torch.randn(MATRIX_SHAPE_2D_GATHER, dtype=torch.float32, device='cuda')
    b_2d_gather = torch.randn(MATRIX_SHAPE_2D_GATHER, dtype=torch.float32, device='cuda')
    print(f"Input 2D (gather) shape: {a_2d_gather.shape}, dtype: {a_2d_gather.dtype}")

    # Call the vec_add wrapper function for 2D, requesting the gather/scatter kernel.
    c_2d_cutile_gather = vec_add(a_2d_gather, b_2d_gather, use_gather=True)
    print(
        f"""cuTile Output 2D (gather) shape: {c_2d_cutile_gather.shape},
        dtype: {c_2d_cutile_gather.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_2d_cutile_gather, a_2d_gather + b_2d_gather)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile Vector/Matrix Addition examples complete ---")
