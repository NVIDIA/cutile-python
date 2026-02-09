# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
from math import ceil  # Required for host-side grid calculation


ConstInt = ct.Constant[int]


def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    # Get the global IDs of a given block in a 1D grid.
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    # Get the global IDs of the current block in a 1D grid.
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(A, B, C,
                  tm: ConstInt,         # Tile size along M dimension (rows of C)
                  tn: ConstInt,         # Tile size along N dimension (columns of C)
                  tk: ConstInt):        # Tile size along K dimension (inner product dimension)
    """
    cuTile kernel for performing matrix multiplication C = A @ B.

    This kernel uses a tiled approach, where each block
    computes a `tm` x `tn` tile of the output matrix C. The computation
    involves iterating over the K-dimension in chunks of `tk`.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        tm (ConstInt): The height of the output tile computed by this block.
                       Corresponds to rows of A and C.
        tn (ConstInt): The width of the output tile computed by this block.
                       Corresponds to columns of B and C.
        tk (ConstInt): The depth of the inner loop (K-dimension) tile size.
                       Corresponds to columns of A and rows of B.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    # Calculate the total number of tiles along the K-dimension that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` means:
    #   "View A as an MxK tensor tiled by (tm, tk), and return the number of tiles along
    #    axis 1 (the K dimension)."
    # We pass shape=(tm, tk) to describe the 2D tiling, only `tk` matters for axis=1.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Initialize an accumulator for the current output tile (tm x tn).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
    # In each iteration, a `tm` x `tk` tile from A and a `tk` x `tn` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(tm, tk)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(tk, tn)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)

        # Perform Matrix Multiplication for the current tiles.
        # `ct.mma` computes the product of the two loaded tiles and accumulates the result.
        accumulator = ct.mma(a, b, accumulator)

    # Convert the final accumulated result to the desired output data type (C.dtype).
    # This might downcast from float32 to float16 if the output is float16.
    accumulator = ct.astype(accumulator, C.dtype)

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)


@ct.kernel
def persistent_matmul_kernel(A, B, C,
                             tm: ConstInt,   # Tile size along M dimension (rows of C)
                             tn: ConstInt,   # Tile size along N dimension (columns of C)
                             tk: ConstInt):  # Tile size along K dimension
    """
    cuTile persistent kernel for performing matrix multiplication C = A @ B.

    This kernel uses a persistent approach, where NUM_SMS tile blocks are launched
    and each tile block processes multiple output tiles.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        tm (ConstInt): The height of the output tile computed by this block.
                       Corresponds to rows of A and C.
        tn (ConstInt): The width of the output tile computed by this block.
                       Corresponds to columns of B and C.
        tk (ConstInt): The depth of the inner loop (K-dimension) tile size.
                       Corresponds to columns of A and rows of B.
    """
    GROUP_SIZE_M = 8

    bid = ct.bid(0)
    M = A.shape[0]
    N = B.shape[1]

    # Calculate the total number of K-tiles that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` extracts the K-dimension (axis 1)
    # from matrix A's shape, assuming A's shape is conceptually (M_tiles, K_tiles),
    # and then implicitly performs ceiling division by `tk` to get the number of K-tiles.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # Number of tiles along M and N
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    upper_bound = num_bid_m * num_bid_n

    # Static persistent loop: each program processes multiple tiles.
    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        # Initialize an accumulator for the current output tile (tm x tn).
        # It's common practice to use `float32` for accumulation even with `float16` inputs
        # to maintain higher precision during the sum-reduction of the matrix multiplication.
        accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
        bidx, bidy = swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, current_bid)

        # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
        # In each iteration, a `tm` x `tk` tile from A and a `tk` x `tn` tile from B
        # are loaded, multiplied, and accumulated.
        for k in range(num_tiles_k):
            # Load tile from matrix A.
            # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
            # from global memory A. `shape=(tm, tk)` defines the size of this tile.
            a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)

            # Load tile from matrix B.
            # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
            # from global memory B. `shape=(tk, tn)` defines the size of this tile.
            b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)

            # Perform Matrix Multiplication for the current tiles.
            # `ct.mma` computes the product of the two loaded tiles and accumulates the result.
            accumulator = ct.mma(a, b, accumulator)

        # Cast result back to C.dtype and store
        accumulator = ct.astype(accumulator, C.dtype)
        ct.store(C, index=(bidx, bidy), tile=accumulator)


def cutile_matmul(A: torch.Tensor, B: torch.Tensor, persistent: bool = False) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B using a cuTile kernel with a 2D grid.

    This wrapper function handles input validation, determines appropriate
    tile sizes based on data type, calculates the necessary grid dimensions,
    and launches the `matmul_kernel`.

    Args:
        A (torch.Tensor): The first input matrix (M x K). Must be on a CUDA device.
        B (torch.Tensor): The second input matrix (K x N). Must be on a CUDA device
                          and have its K dimension match A's K dimension.
        persistent (bool): Whether to use the persistent kernel.

    Returns:
        torch.Tensor: The resulting matrix C (M x N) on the CUDA device.

    Raises:
        ValueError: If matrices are incompatible (K dimensions don't match),
                    or if they are not on a CUDA device.
    """
    # --- Input Validation ---
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible matrices: K dimension of A ({A.shape[1]}) "
                         f"must match K dimension of B ({B.shape[0]})")
    if A.device != B.device:
        raise ValueError("Input tensors must be on the same device.")
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")
    # Note: cuTile handles dtype compatibility within the kernel, but inputs should generally match.

    # --- Determine Tile Shapes based on Data Type for Optimization ---
    # This logic selects optimal tile sizes (tm, tn, tk) based on whether
    # the input is half-precision (e.g., float16, bfloat16, where itemsize=2 bytes)
    # which can often leverage Tensor Cores for higher throughput,
    # or full-precision (e.g., float32, where itemsize=4 bytes).
    if A.dtype.itemsize == 2:  # Likely torch.float16 or torch.bfloat16
        tm, tn, tk = 128, 256, 64  # Larger tiles for Tensor Core friendly types
    else:  # Likely torch.float32 or other
        tm, tn, tk = 32, 32, 32   # Smaller, more general tiles

    # --- Get Matrix Dimensions ---
    m, k_a = A.shape  # M = total rows of A (and C), K_A = total columns of A
    k_b, n = B.shape  # K_B = total rows of B, N = total columns of B (and C)
    # Note: k_a and k_b must be equal due to validation. This is the 'K' dimension.

    # --- Calculate Grid Dimensions for Kernel Launch (1D Grid) ---
    # The grid defines how many CUDA blocks (CTAs) will be launched.
    # Each block computes one (tm x tn) output tile of matrix C.
    # `ceil(total_dim / tile_dim)` ensures enough blocks are launched to cover
    # the entire matrix, even if dimensions are not perfect multiples of tile sizes.
    grid_x = ceil(m / tm)  # Number of blocks needed along the M dimension (rows of C)
    grid_y = ceil(n / tn)  # Number of blocks needed along the N dimension (columns of C)
    grid_size = grid_x * grid_y
    if persistent:
        NUM_SMS = torch.cuda.get_device_properties(
            "cuda"
        ).multi_processor_count
        grid_size = min(NUM_SMS, grid_size)
    grid = (grid_size, 1, 1)

    # --- Create Output Tensor C ---
    # The output tensor `C` is initialized with the correct dimensions (M x N),
    # on the same device, and with the same data type as the input matrices.
    C = torch.empty((m, n), device=A.device, dtype=A.dtype)

    # --- Launch the cuTile Kernel ---
    # The `matmul_kernel` is launched with the calculated grid dimensions.
    # `tm`, `tn`, and `tk` are passed as Constant integers to the kernel.
    kernel = persistent_matmul_kernel if persistent else matmul_kernel
    ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, tm, tn, tk))

    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    # --- Running cuTile Matrix Multiplication Examples ---
    print("--- Running cuTile Matrix Multiplication Examples (2D Grid) ---")

    # Define common matrix dimensions for the examples
    M_dim = 512
    N_dim = 512
    K_dim = 768

    # --- Test Case 1: float16 (Half-Precision) ---
    print("\n--- Test Case 1: Matrix Multiplication with float16 (Half-Precision) ---")
    # Create random input matrices with float16 data type on the CUDA device.
    A_fp16 = torch.randn(M_dim, K_dim, dtype=torch.float16, device='cuda')
    B_fp16 = torch.randn(K_dim, N_dim, dtype=torch.float16, device='cuda')
    print(f"Input A shape: {A_fp16.shape}, dtype: {A_fp16.dtype}")
    print(f"Input B shape: {B_fp16.shape}, dtype: {B_fp16.dtype}")

    # Perform matrix multiplication using the cuTile wrapper function.
    C_fp16_cutile = cutile_matmul(A_fp16, B_fp16)
    print(f"cuTile Output C shape: {C_fp16_cutile.shape}, dtype: {C_fp16_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_fp16_cutile, A_fp16 @ B_fp16)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: float32 (Single-Precision) ---
    torch.set_float32_matmul_precision("high")
    print("\n--- Test Case 2: Matrix Multiplication with float32 (Single-Precision) ---")
    # Create random input matrices with float32 data type on the CUDA device.
    A_fp32 = torch.randn(M_dim, K_dim, dtype=torch.float32, device='cuda')
    B_fp32 = torch.randn(K_dim, N_dim, dtype=torch.float32, device='cuda')
    print(f"Input A shape: {A_fp32.shape}, dtype: {A_fp32.dtype}")
    print(f"Input B shape: {B_fp32.shape}, dtype: {B_fp32.dtype}")

    if torch.cuda.get_device_capability()[0] <= 8:
        # Ampere tfloat32 numerics is loose
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-4, 1e-3

    # Perform matrix multiplication using the cuTile wrapper function.
    C_fp32_cutile = cutile_matmul(A_fp32, B_fp32)
    print(f"cuTile Output C shape: {C_fp32_cutile.shape}, dtype: {C_fp32_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_fp32_cutile, A_fp32 @ B_fp32, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 3: Dimensions Not Multiples of Tile Sizes ---
    print("""\n--- Test Case 3: Matrix Multiplication with Dimensions
            Not Perfect Multiples of Tile Sizes ---""")
    # Define matrix dimensions that are not exact multiples of the default tile sizes (32, 32, 32).
    # This demonstrates that `ceil` in grid calculation correctly handles partial tiles.
    M_dim_non_mult = 1000
    N_dim_non_mult = 500
    K_dim_non_mult = 700
    A_non_mult = torch.randn(M_dim_non_mult, K_dim_non_mult, dtype=torch.float32, device='cuda')
    B_non_mult = torch.randn(K_dim_non_mult, N_dim_non_mult, dtype=torch.float32, device='cuda')
    print(f"Input A shape: {A_non_mult.shape}, dtype: {A_non_mult.dtype}")
    print(f"Input B shape: {B_non_mult.shape}, dtype: {B_non_mult.dtype}")

    C_non_mult_cutile = cutile_matmul(A_non_mult, B_non_mult)
    print(f"cuTile Output C shape: {C_non_mult_cutile.shape}, dtype: {C_non_mult_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_non_mult_cutile, A_non_mult @ B_non_mult, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 4: Persistent Matmul ---
    print("\n--- Test Case 4: Matrix Multiplication with Persistent Matmul ---")
    C_persistent_fp32_cutile = cutile_matmul(A_fp32, B_fp32, persistent=True)
    print(f"cuTile Output C shape: {C_persistent_fp32_cutile.shape}, "
          f"dtype: {C_persistent_fp32_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_persistent_fp32_cutile, A_fp32 @ B_fp32, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")
    torch.set_float32_matmul_precision("highest")

    print("\n--- All cuTile matrix multiplication examples completed. ---")
