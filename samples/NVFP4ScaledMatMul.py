# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
import sys
from cuda.tile._cext import get_compute_capability
from cuda.tile._bytecode.version import BytecodeVersion


from cuda.tile._cext import dev_features_enabled
from cuda.tile._compile import _get_max_supported_bytecode_version
from functools import cache
import tempfile


@cache
def get_tileiras_version():
    return _get_max_supported_bytecode_version(tempfile.gettempdir(),
                                               allow_dev=dev_features_enabled())


def swizzle_32_4_4(scale):
    '''
    Prepare the original scale tensor to align with the expected tmem layout.
    With the innermost dimensions being (m1=32, m2=4, k1=4), and the outer dimensions
    being (m0=(M // (m1 * m2)), k0=(K_s // k1)).

    Reference: PTX ISA tcgen05.mma scale factor layout.
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
    '''
    m1, m2, k1 = 32, 4, 4

    M, K_s = scale.shape
    m0 = M // (m1 * m2)
    k0 = K_s // k1
    scale = scale.reshape(m0, m2, m1, k0, k1).permute(0, 3, 2, 1, 4).contiguous()
    return scale.reshape(m0, k0, 32, 16)


def _pad_tensor(x: torch.Tensor, block_size: int):
    if not x.dim():
        raise ValueError('The input tensor must at least have 1 dimension')

    if x.shape[-1] % block_size != 0:
        print('[WARNING] the innermost dimension of the input tensor')
        print('          is not a multiple of block_size padding the input tensor...')
        pad_len = (block_size - (x.shape[-1] % block_size)) % block_size
        x = torch.nn.functional.pad(x, (0, pad_len), value=0)
    return x


def block_quantize_f4e2m1fn_f8e4m3fn(x: torch.Tensor, block_size: int):
    F4E2M1FN_MAX = 6
    F8E4M3FN_MAX = 448
    x = _pad_tensor(x, block_size)

    x_block = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    block_amax = x_block.abs().amax(dim=-1, keepdim=True)
    global_amax = x.abs().amax()
    global_scale = global_amax / (F4E2M1FN_MAX * F8E4M3FN_MAX)
    block_scale = block_amax / (F4E2M1FN_MAX * torch.where(global_scale == 0,
                                                           torch.ones_like(global_scale),
                                                           global_scale))

    block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)
    effective_block_scale = block_scale_fp8.to(x.dtype)
    effective_block_scale *= global_scale
    x_block_scaled = x_block / torch.where(effective_block_scale == 0,
                                           torch.ones_like(effective_block_scale),
                                           effective_block_scale)

    x_block_scaled_fp4_bits_packed = pack_e2m1(x_block_scaled)
    x_block_scaled = x_block_scaled_fp4_bits_packed.reshape(*x.shape[:-1], x.shape[-1] // 2)
    block_scale_fp8 = block_scale_fp8.squeeze(-1)
    return x_block_scaled, block_scale_fp8, global_scale.view((1,))


def pack_e2m1(x: torch.Tensor) -> torch.Tensor:
    """
    Input shape:  (..., K)
    Output shape: (..., K // 2)
    """
    assert x.dtype == torch.float32
    assert x.shape[-1] % 2 == 0

    mag = x.abs()

    lookup = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    midpoints = ((i + j) / 2 for i, j in zip(lookup[:-1], lookup[1:]))

    # the alternating > and >= comparisons select the even encoding
    # when a value lies exactly halfway between two values (round to nearest even encoding)
    encoding = 0
    for index, e in enumerate(midpoints):
        if index % 2:
            encoding += (mag >= e).to(torch.uint8)
        else:
            encoding += (mag > e).to(torch.uint8)

    # add the E2M1 sign bit
    encoding |= torch.signbit(x).to(torch.uint8) << 3

    # pack two E2M1 values into each byte
    packed = encoding[..., 0::2] | (encoding[..., 1::2] << 4)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def unpack_e2m1_bytes_to_float(fp4_bytes: torch.Tensor) -> torch.Tensor:
    lookup = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=fp4_bytes.device,
    )
    low = fp4_bytes & 0x0F
    high = (fp4_bytes >> 4) & 0x0F
    low_values = lookup[(low & 0x7).long()]
    high_values = lookup[(high & 0x7).long()]
    low_values = torch.where((low & 0x8) != 0, -low_values, low_values)
    high_values = torch.where((high & 0x8) != 0, -high_values, high_values)
    values = torch.empty(
        (*fp4_bytes.shape[:-1], fp4_bytes.shape[-1] * 2),
        dtype=torch.float32,
        device=fp4_bytes.device,
    )
    values[..., 0::2] = low_values
    values[..., 1::2] = high_values
    return values


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


def unswizzle_32_4_4(tile_swizzled_scale):
    '''
    Kernel-side inverse of ``swizzle_32_4_4``: take a tile loaded
    from the host swizzled scale tensor and recover the ``(M, K_s)``
    view that ``ct.mma_scaled`` expects.
    '''
    m1, m2, k1 = 32, 4, 4
    m0, k0, _, _ = tile_swizzled_scale.shape

    return (tile_swizzled_scale.reshape((m0, k0, m1, m2, k1))
                               .permute((0, 3, 2, 1, 4))
                               .reshape((m0 * m2 * m1, k0 * k1)))


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_103=2))
def nvfp4_block_scaled_matmul_kernel(
                    A, A_scale, A_global, B, B_scale, B_global, C,
                    tm: ConstInt,         # Tile size along M dimension (rows of C)
                    tn: ConstInt,         # Tile size along N dimension (columns of C)
                    tk: ConstInt,         # Tile size along K dimension (inner product dimension)
                    block_size: ConstInt):

    """
    cuTile kernel for block-scaled matrix multiplication.

    Computes C = (A * A_scale * A_global) @ (B * B_scale * B_global) if A and B contain
    packed E2M1 values, with two FP4 elements stored in each byte.

    Each TileBlock computes one tm x tn output tile. The K dimension is processed
    in chunks of tk, with tks scale values per K tile.

    If packed swizzle scales are passed, they get unswizzled into logical 2D scale tiles,
    then passed to ct.mma_scaled.

    Args:
        A:              Input matrix A with physical shape (M, K // 2) which is a
                        uint8 view of packed E2M1 data.
        A_scale:        2D scale of (M, K // block_size) or swizzle scale of
                        (M // 32 // 4, K // block_size // 4, 32, 4, 4) reshaped
                        into (M // 32 // 4, K // block_size // 4, 32, 16).
        A_global:       One element float32 tensor containing the global scale for A.
        B:              Input matrix B with physical shape (N, K // 2) which is a
                        uint8 view of packed E2M1 data.
        B_scale:        2D scale of (N, K // block_size) or swizzled scale of
                        (N // 32 // 4, K // block_size // 4, 32, 4, 4) reshaped
                        into (N // 32 // 4, K // block_size // 4, 32, 16).
        B_global:       One element float32 tensor containing the global scale for B.
        C:              Output matrix C (M x N).
        tm (ConstInt):  The height of the output tile computed by this block.
                        Corresponds to rows of A and C.
        tn (ConstInt):  The width of the output tile computed by this block.
                        Corresponds to columns of B and C.
        tk (ConstInt):  The depth of the inner loop (K-dimension) tile size.
                        Corresponds to columns of A and rows of B.

        block_size (ConstInt): The scaling block size.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[0]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)
    tks = tk // block_size

    # Calculate the total number of tiles along the K-dimension that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` means:
    #   "View A as an MxK tensor tiled by (tm, tk), and return the number of tiles along
    #    axis 1 (the K dimension)."
    # We pass shape=(tm, tk) to describe the 2D tiling, only `tk` matters for axis=1.
    # The inputs are made from two packed fp4 values
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk // 2))

    # Initialize an accumulator for the current output tile (tm x tn).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
    # In each iteration, a `tm` x `tk` tile from A and a `tn` x `tk` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(tm, tk)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(tm, tk // 2), padding_mode=zero_pad)
        a = ct.unpack_from_bytes(a.reshape((-1,)), ct.float4_e2m1fn).reshape((tm, tk))

        if len(A_scale.shape) == 2:
            # 2D scale path. A_scale is already stored in logical shape (M, K_s).
            a_scale = ct.load(A_scale, index=(bidx, k), shape=(tm, tks), padding_mode=zero_pad)
        else:
            # Load the packed scale tile, unswizzle it to the logical ct.mma_scaled shape (tm, tks).

            # unswizzle
            a_scale_swizzled = ct.load(A_scale, index=(bidx, k, 0, 0),
                                       shape=(tm // 32 // 4, tks // 4, 32, 16),
                                       padding_mode=zero_pad)
            a_scale = unswizzle_32_4_4(a_scale_swizzled)

        # Load tile from matrix B.
        # The `index=(bidy, k_tile_idx)` specifies which (N-tile, K-tile) to load
        # from global memory B. `shape=(tn, tk)` defines the size of this tile.
        b = ct.load(B, index=(bidy, k), shape=(tn, tk // 2), padding_mode=zero_pad)
        b = ct.unpack_from_bytes(b.reshape((-1,)), ct.float4_e2m1fn).reshape((tn, tk))
        b = b.permute((1, 0))

        if len(B_scale.shape) == 2:
            b_scale = ct.load(B_scale, index=(bidy, k), shape=(tn, tks), padding_mode=zero_pad)
            b_scale = b_scale.permute((1, 0))
        else:
            # B scales are stored N-major. Unswizzle it, then transpose it to the
            # logical ct.mma_scaled shape (tks, tn).

            # unswizzle
            b_scale_swizzled = ct.load(B_scale, index=(bidy, k, 0, 0),
                                       shape=(tn // 32 // 4, tks // 4, 32, 16),
                                       padding_mode=zero_pad)
            b_scale = unswizzle_32_4_4(b_scale_swizzled).permute((1, 0))

        # Perform Scaled Matrix Multiplication for the current tiles.
        # `ct.mma_scaled` computes the product of the two loaded tiles
        # and scales and accumulates the result.
        accumulator = ct.mma_scaled(a, a_scale, b, b_scale, accumulator)

    a_global = ct.load(A_global, index=(0,), shape=(1,))
    b_global = ct.load(B_global, index=(0,), shape=(1,))
    accumulator = accumulator * a_global * b_global

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)


def cutile_nvfp4_matmul(A: torch.Tensor, A_scale: torch.Tensor, A_global: torch.Tensor,
                        B: torch.Tensor, B_scale: torch.Tensor, B_global: torch.Tensor,
                        scaling_block_size: int) -> torch.Tensor:

    """
    Performs NVFP4 matrix multiplication using a cuTile kernel.

    This wrapper function handles input validation, calculates the necessary grid dimensions,
    and launches the `nvfp4_block_scaled_matmul_kernel`.

    Args:
        A (torch.Tensor):         Packed input matrix with physical shape (M, K // 2).
                                  Must be on a CUDA device.
        A_scale (torch.Tensor):   Either 2D scale with shape (M, K // NVFP4_BLOCK_SIZE) or swizzled
                                  scale of (M // 32 // 4, K // NVFP4_BLOCK_SIZE // 4, 32, 16).
        A_global (torch.Tensor):  One-element float32 tensor containing A's tensor-wide scale.
        B (torch.Tensor):         Packed input matrix with physical shape (N, K // 2).
                                  Must be on a CUDA device and have its 2nd dimension match
                                  A's 2nd dimension.
        B_scale (torch.Tensor):   Either 2D scale with shape (N, K // NVFP4_BLOCK_SIZE) or swizzled
                                  scale of  (N // 32 // 4, K // NVFP4_BLOCK_SIZE // 4, 32, 16).
        B_global (torch.Tensor):  One-element float32 tensor containing B's tensor-wide scale.
        scaling_block_size (int): The scaling block size.

    Returns:
        torch.Tensor: The resulting matrix C (M x N) on the CUDA device.

    Raises:
        ValueError: If matrices are incompatible (K dimensions don't match),
                    or if they are not on a CUDA device.
    """
    # --- Input Validation ---
    if A.shape[1] != B.shape[1]:
        raise ValueError("Incompatible matrices")
    if not (A.device == A_scale.device == A_global.device ==
            B.device == B_scale.device == B_global.device):
        raise ValueError("Input tensors must be on the same device.")
    if not (A.is_cuda and A_scale.is_cuda and A_global.is_cuda and
            B.is_cuda and B_scale.is_cuda and B_global.is_cuda):
        raise ValueError("Input tensors must be on a CUDA device.")

    A = A.view(torch.uint8)
    B = B.view(torch.uint8)

    # Note: cuTile handles dtype compatibility within the kernel,
    # but inputs should generally match.
    tm, tn, tk = 256, 256, 256

    # --- Get Matrix Dimensions ---
    m, _ = A.shape
    n, _ = B.shape

    # --- Calculate Grid Dimensions for Kernel Launch (1D Grid) ---
    # The grid defines how many CUDA blocks (CTAs) will be launched.
    # Each block computes one (tm x tn) output tile of matrix C.
    # `ct.cdiv(total_dim, tile_dim)` ensures enough blocks are launched to cover
    # the entire matrix, even if dimensions are not perfect multiples of tile sizes.
    grid_x = ct.cdiv(m, tm)  # Number of blocks needed along the M dimension (rows of C)
    grid_y = ct.cdiv(n, tn)  # Number of blocks needed along the N dimension (columns of C)
    grid_size = grid_x * grid_y

    grid = (grid_size, 1, 1)

    # --- Create Output Tensor C ---
    # The output tensor `C` is initialized with the correct dimensions (M x N),
    # on the same device, and with a datatype of float32.
    C = torch.empty((m, n), device=A.device, dtype=torch.float32)

    # --- Launch the cuTile Kernel ---
    # The `nvfp4_block_scaled_matmul_kernel` is launched with the calculated grid dimensions.
    # `tm`, `tn`, and `tk` are passed as Constant integers to the kernel.
    kernel = nvfp4_block_scaled_matmul_kernel
    ct.launch(torch.cuda.current_stream(), grid, kernel, (
        A, A_scale, A_global, B, B_scale, B_global, C, tm, tn, tk, scaling_block_size))
    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    if get_compute_capability()[0] < 10:
        print("Skipped test: NOT Running cuTile NVFP4 Matrix Multiplication Examples "
              "Blackwell or newer required.")
        sys.exit(0)

    if get_tileiras_version() < BytecodeVersion.V_13_4:
        print("Skipped test: NOT Running cuTile NVFP4 Matrix Multiplication Examples "
              "tileiras version 13.4 required.")
        sys.exit(0)

    # --- Running cuTile NVFP4 Matrix Multiplication Examples ---
    print("--- Running cuTile NVFP4 Matrix Multiplication Examples ---")

    # Define common matrix dimensions for the examples
    M_dim = 512
    N_dim = 512
    K_dim = 768

    scaling_block_size = 16

    print(f"\n--- Test Case: NVFP4 Matrix Multiplication with M = {M_dim}, N = {N_dim}, "
          f"K = {K_dim}, Scaling Block Size = {scaling_block_size} ---")

    A = torch.rand((M_dim, K_dim), device='cuda:0')
    B = torch.rand((N_dim, K_dim), device='cuda:0')

    uncompressed_ref = A @ B.T

    A, A_scale, A_global = block_quantize_f4e2m1fn_f8e4m3fn(A, scaling_block_size)
    B, B_scale, B_global = block_quantize_f4e2m1fn_f8e4m3fn(B, scaling_block_size)

    A_s_swizzled = swizzle_32_4_4(A_scale)
    B_s_swizzled = swizzle_32_4_4(B_scale)

    print(f"Input A packed shape: {A.shape}, dtype: {A.dtype}")
    print(f"Input B packed shape: {B.shape}, dtype: {B.dtype}")

    kernel_atol, kernel_rtol = 1e-4, 1e-3
    compression_atol, compression_rtol = 0.5, 0.05

    # Perform NVFP4 matrix multiplication using the cuTile wrapper function.
    C_cutile = cutile_nvfp4_matmul(A, A_scale, A_global, B, B_scale, B_global, scaling_block_size)
    torch.cuda.synchronize()
    C_cutile_swizzled = cutile_nvfp4_matmul(A, A_s_swizzled, A_global, B, B_s_swizzled, B_global,
                                            scaling_block_size)
    torch.cuda.synchronize()
    print(f"cuTile Output C shape: {C_cutile.shape}, dtype: {C_cutile.dtype}")

    if args.correctness_check:
        ref_A_scale = torch.repeat_interleave(A_scale, scaling_block_size, dim=1).to(torch.float32)
        ref_B_scale = torch.repeat_interleave(B_scale, scaling_block_size, dim=1).to(torch.float32)
        ref_A = unpack_e2m1_bytes_to_float(A.view(torch.uint8))
        ref_B = unpack_e2m1_bytes_to_float(B.view(torch.uint8))
        ref = (ref_A * ref_A_scale * A_global) @ (ref_B.T * ref_B_scale.T * B_global)

        torch.testing.assert_close(C_cutile, ref, atol=kernel_atol, rtol=kernel_rtol)
        torch.testing.assert_close(C_cutile_swizzled, ref, atol=kernel_atol, rtol=kernel_rtol)

        torch.testing.assert_close(C_cutile, uncompressed_ref,
                                   atol=compression_atol, rtol=compression_rtol)
        torch.testing.assert_close(C_cutile_swizzled, uncompressed_ref,
                                   atol=compression_atol, rtol=compression_rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile NVFP4 matrix multiplication example completed. ---")
