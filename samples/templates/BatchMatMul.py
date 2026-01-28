# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from math import ceil
import cuda.tile as ct
import torch


from test.kernels.matmul import batch_matmul_kernel


def bmm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Batch Matrix Multiplication using cuTile's standard tiled kernel.

    Args:
        a (torch.Tensor): Input tensor A with shape (Batch, M, K).
        b (torch.Tensor): Input tensor B with shape (Batch, K, N).

    Returns:
        Output tensor C with shape (Batch, M, N).
    """
    # --- Input Validation ---
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("Input tensors for BMM must be 3D (Batch, M, K) and (Batch, K, N).")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"""Batch dimensions must match:
                         A.shape[0]={a.shape[0]}, B.shape[0]={b.shape[0]}.""")
    if a.device != b.device or not a.is_cuda or not b.is_cuda or a.dtype != b.dtype:
        raise ValueError("""Input tensors must be on the same CUDA device
                         and have the same data type.""")

    # Get M, K, N dimensions
    Batch, M, K = a.shape
    _, K_b, N = b.shape
    assert K == K_b, f"Incompatible K dimensions: A's K is {K}, B's K is {K_b}"

    # Create output tensor
    output = torch.empty((Batch, M, N), device=a.device, dtype=out_dtype)

    # --- Determine Tile Shapes for Optimization (Fixed for float16 as per previous request) ---
    tm_val, tn_val, tk_val = 128, 256, 64  # Larger tiles for Tensor Core benefits

    # --- Grid calculation for standard 3D tiled kernel ---
    grid = (Batch, ceil(M / tm_val), ceil(N / tn_val))

    # --- Launch kernel ---
    ct.launch(torch.cuda.current_stream(), grid, batch_matmul_kernel,
              (a, b, output, tm_val, tn_val, tk_val))

    return output


def torch_batch_matmul_fp8(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    inv_sa = torch.tensor(1.0, device=A.device, dtype=torch.float32)
    inv_sb = torch.tensor(1.0, device=B.device, dtype=torch.float32)
    bs = A.shape[0]
    C = torch.empty((bs, A.shape[1], B.shape[2]), device=A.device, dtype=torch.float32)
    for i in range(bs):
        # Only multiplication of row-major and column-major matrices is supported by cuBLASLt
        # So we need to transpose B to column-major view
        A_row = A[i].contiguous()
        B_col = B[i].transpose(-2, -1).contiguous().transpose(-2, -1)
        C[i] = torch._scaled_mm(
            A_row, B_col, scale_a=inv_sa, scale_b=inv_sb, out_dtype=torch.float32
        )
    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()
    print("--- Running cuTile Batched Matrix Multiplication (Standard Tiled) Sample ---")

    # --- User Configuration for BMM Example ---
    BATCH_DIM = 4
    M_DIM = 512
    K_DIM = 256
    N_DIM = 1024

    # --- Test Case 1: Standard BMM (float16) ---
    print("\n--- Test 1: Standard BMM (float16) ---")
    A_fp16 = torch.randn(BATCH_DIM, M_DIM, K_DIM, dtype=torch.float16, device='cuda')
    B_fp16 = torch.randn(BATCH_DIM, K_DIM, N_DIM, dtype=torch.float16, device='cuda')
    print(f"Input A shape: {A_fp16.shape}, dtype: {A_fp16.dtype}")
    print(f"Input B shape: {B_fp16.shape}, dtype: {B_fp16.dtype}")

    C_bmm_cutile_fp16 = bmm(A_fp16, B_fp16, A_fp16.dtype)
    print(f"""cuTile Standard BMM Output C
            shape:{C_bmm_cutile_fp16.shape},
            dtype: {C_bmm_cutile_fp16.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(C_bmm_cutile_fp16, A_fp16 @ B_fp16)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: Standard BMM (float8_e4m3fn) ---
    print("\n--- Test 2: Standard BMM (float8_e4m3fn) ---")
    if torch.cuda.get_device_capability()[0] == 8:
        print("skip: Ampere does not support float8")
    else:
        A_fp8 = torch.randn(
            BATCH_DIM, M_DIM, K_DIM, dtype=torch.float32, device='cuda'
        ).to(torch.float8_e4m3fn)
        B_fp8 = torch.randn(
            BATCH_DIM, K_DIM, N_DIM, dtype=torch.float32, device='cuda'
        ).to(torch.float8_e4m3fn)
        print(f"Input A shape: {A_fp8.shape}, dtype: {A_fp8.dtype}")
        print(f"Input B shape: {B_fp8.shape}, dtype: {B_fp8.dtype}")

        C_bmm_cutile_fp32 = bmm(A_fp8, B_fp8, torch.float32)
        print(f"""cuTile Standard BMM Output C
                shape:{C_bmm_cutile_fp32.shape},
                dtype: {C_bmm_cutile_fp32.dtype}""")
        if args.correctness_check:
            torch.testing.assert_close(C_bmm_cutile_fp32, torch_batch_matmul_fp8(A_fp8, B_fp8))
            print("Correctness check passed")
        else:
            print("Correctness check disabled")

    print("\n--- cuTile Batched Matrix Multiplication (Standard Tiled) examples complete ---")
