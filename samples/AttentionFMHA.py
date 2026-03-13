# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None
import torch
import math
import sys

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from utils.benchmark import report_benchmark
from types import SimpleNamespace


import numpy as np
from cuda.tile import RoundingMode as RMd


INV_LOG_2 = 1.0 / math.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


@ct.kernel(occupancy=2)
def fmha_kernel(Q, K, V, Out,
                qk_scale: float,
                input_pos: int,
                Dqk: ConstInt,  # Head dimension of Q and K
                Dv: ConstInt,  # Head dimension of V
                H: ConstInt,
                TILE_M: ConstInt,
                TILE_N: ConstInt,
                QUERY_GROUP_SIZE: ConstInt,
                CAUSAL: ConstBool,
                EVEN_K: ConstBool):
    """
    cuTile kernel for Fused Multi-Head Attention (FMHA).
    Computes attention output for a specific batch item and head, using tiling and online softmax.
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)  # [TILE_M]
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)  # [TILE_N]
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, Dv), 0.0, dtype=np.float32)

    # Load query tile for this batch, head, and M-chunk
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, Dqk)
    ).reshape((TILE_M, Dqk))  # [TILE_M, Dqk]

    # loop over k, v and update accumulator
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        # when kv pos could exceed q pos
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        # when kv pos could exceed k_seqlen
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over K, V blocks (N-dimension chunks)
    for j in range(0, Tc):
        # --- Compute QK product ---
        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, Dqk, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((Dqk, TILE_N))  # [Dqk, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]

        # --- Apply Causal Masking ---
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=np.bool_)
            # out of bound mask
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            # causal mask
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)  # [TILE_M, TILE_N]
            mask = ct.where(mask, 0.0, -np.inf)  # [TILE_M, TILE_N]
            qk += mask

        # --- Online Softmax Update ---
        # Moving qk_scale multiplication after reduce_max is to improve performance.
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij  # [TILE_M, TILE_N]

        # attention weights
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_M, 1]
        # update m_i and l_i
        l_i = l_i * alpha + l_ij  # [TILE_M, 1]
        # scale acc
        acc = acc * alpha  # [TILE_M, TILE_N]

        # --- Compute PV product ---
        v = ct.load(
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, Dv),
            latency=4,
        ).reshape((TILE_N, Dv))  # [TILE_N, Dv]
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)  # [TILE_M, TILE_N]
        m_i = m_ij  # [TILE_M, 1]

    # --- Final Normalization and Store ---
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, Dv)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


# --- Wrapper function to launch the FMHA kernel ---
def cutile_fmha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                qk_scale: float | None = None,
                input_pos: int = 0,
                tile_m: int = 128,
                tile_n: int = 128,
                query_group_size: int = 1,
                causal: bool = False) -> torch.Tensor:
    """
    Performs Fused Multi-Head Attention (FMHA) using a cuTile kernel.

    Args:
        Q (torch.Tensor): Query tensor (Batch, Heads, SeqLen_Q, D_k).
        K (torch.Tensor): Key tensor (Batch, KV_Heads, SeqLen_KV, D_k).
        V (torch.Tensor): Value tensor (Batch, KV_Heads, SeqLen_KV, D_v).
        qk_scale (float, optional): Scaling factor for QK dot product. Defaults to 1/sqrt(D_k).
        input_pos (int, optional): Global start pos for queries (causal masking). Defaults to 0.
        tile_m (int): Tile size for Query sequence length (M dimension).
        tile_n (int): Tile size for Key/Value sequence length (N dimension).
        query_group_size (int): Number of query heads per key/value head.
        causal (bool): If True, applies causal masking.

    Returns:
        torch.Tensor: Output tensor (Batch, Heads, SeqLen_Q, D_v).
    """
    # --- Input Validation ---
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("Input tensors Q, K, V must be 4D (Batch, Heads, SeqLen, Dim).")
    if Q.shape[0] != K.shape[0] or Q.shape[0] != V.shape[0]:
        raise ValueError("Batch dimensions must match for Q, K, V.")
    if Q.shape[1] % query_group_size != 0:
        raise ValueError("Number of query heads must be divisible by query_group_size.")
    if K.shape[1] * query_group_size != Q.shape[1]:
        raise ValueError("K_Heads * query_group_size must equal Q_Heads.")
    if Q.shape[3] != K.shape[3]:
        raise ValueError("D_k (last dim of Q and K) must match.")
    if K.shape[2] != V.shape[2]:
        raise ValueError("SeqLen_KV (dim 2 of K and V) must match.")
    if Q.device != K.device or Q.device != V.device or not Q.is_cuda:
        raise ValueError("All input tensors must be on the same CUDA device.")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError("All input tensors must have the same data type.")

    Batch, Heads, SeqLen_Q, D_k = Q.shape
    _, KV_Heads, SeqLen_KV, D_v = V.shape
    even_k = (SeqLen_KV % tile_n) == 0

    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(D_k)

    # --- Create Output Tensor ---
    Out = torch.empty((Batch, Heads, SeqLen_Q, D_v), dtype=Q.dtype, device=Q.device)

    # --- Calculate Grid Dimensions ---
    grid_x = math.ceil(SeqLen_Q / tile_m)
    grid_y = Batch * Heads
    grid = (grid_x, grid_y, 1)

    # --- Launch the FMHA Kernel ---
    ct.launch(torch.cuda.current_stream(), grid, fmha_kernel, (
        Q, K, V, Out,
        qk_scale,
        input_pos,
        D_k,
        D_v,
        Heads,
        tile_m,
        tile_n,
        query_group_size,
        causal,
        even_k
    ))

    return Out


# --- Wrapper function to launch the FMHA kernel with autotuning ---
def cutile_autotune_fmha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                         qk_scale: float,
                         input_pos: int = 0,
                         query_group_size: int = 1,
                         causal: bool = False) -> tuple[torch.Tensor, dict[str, int]]:
    """
    Performs Fused Multi-Head Attention (FMHA) using a cuTile kernel with autotuning.

    Args:
        Q (torch.Tensor): Query tensor (Batch, Heads, SeqLen_Q, D_k).
        K (torch.Tensor): Key tensor (Batch, KV_Heads, SeqLen_KV, D_k).
        V (torch.Tensor): Value tensor (Batch, KV_Heads, SeqLen_KV, D_v).
        qk_scale (float, optional): Scaling factor for QK dot product. Defaults to 1/sqrt(D_k).
        input_pos (int, optional): Global start pos for queries (causal masking). Defaults to 0.
        query_group_size (int): Number of query heads per key/value head.
        causal (bool): If True, applies causal masking.
        autotuner (Autotuner | None): Autotuner object that was injected by the autotune decorator.

    Returns:
        torch.Tensor: Output tensor (Batch, Heads, SeqLen_Q, D_v).
        dict[str, int]: The best configuration found by the autotuner.
    """
    Batch, Heads, SeqLen_Q, D_k = Q.shape
    _, KV_Heads, SeqLen_KV, D_v = V.shape

    # --- Create Output Tensor ---
    Out = torch.empty((Batch, Heads, SeqLen_Q, D_v), dtype=Q.dtype, device=Q.device)

    # --- Tune/Get the best configuration for the FMHA Kernel ---
    tuned_result = ct_experimental.autotune_launch(
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (math.ceil(SeqLen_Q / cfg.TILE_M), Batch * Heads, 1),
        kernel=fmha_kernel,
        args_fn=lambda cfg: (
            Q, K, V, Out,
            qk_scale, input_pos, D_k, Heads,
            cfg.TILE_M, cfg.TILE_N, query_group_size, causal, (SeqLen_KV % cfg.TILE_N) == 0
        ),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=[
            SimpleNamespace(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=2),
            SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=2, occupancy=2),
            SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2),
            SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=1),
            SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=4),
            SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=2, occupancy=1),
            SimpleNamespace(TILE_M=64, TILE_N=32, num_ctas=1, occupancy=2),
            SimpleNamespace(TILE_M=256, TILE_N=32, num_ctas=2, occupancy=2),
            SimpleNamespace(TILE_M=32, TILE_N=32, num_ctas=1, occupancy=1),
        ],
    )

    return Out, tuned_result.tuned_config


def torch_fmha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
               is_causal: bool, enable_gqa: bool,
               use_backend_selection_rule: bool = False) -> torch.Tensor:
    if use_backend_selection_rule:
        backend = SDPBackend.CUDNN_ATTENTION \
                if (Q.shape[2] == K.shape[2]) \
                else SDPBackend.FLASH_ATTENTION
        with sdpa_kernel(backend):
            ret = scaled_dot_product_attention(Q, K, V,
                                               is_causal=is_causal,
                                               enable_gqa=enable_gqa)
    else:
        ret = scaled_dot_product_attention(Q, K, V, 
                                           is_causal=is_causal,
                                           enable_gqa=enable_gqa)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()
    print("--- Running cuTile Fused Multi-Head Attention (FMHA) Sample ---")

    # --- User Configuration ---
    BATCH_SIZE = 2
    NUM_HEADS = 32
    SEQ_LEN_Q = 128
    SEQ_LEN_KV = 256
    D_K = 128
    D_V = 64

    QUERY_GROUP_SIZE = 8
    enable_gqa = QUERY_GROUP_SIZE > 1

    DTYPE = torch.float16

    Q_input = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, D_K, dtype=DTYPE, device='cuda')
    K_input = torch.randn(BATCH_SIZE, NUM_HEADS // QUERY_GROUP_SIZE, SEQ_LEN_KV, D_K,
                          dtype=DTYPE, device='cuda')
    V_input = torch.randn(BATCH_SIZE, NUM_HEADS // QUERY_GROUP_SIZE, SEQ_LEN_KV, D_V,
                          dtype=DTYPE, device='cuda')

    print("  Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Number of Heads: {NUM_HEADS}")
    print(f"  Query Sequence Length: {SEQ_LEN_Q}")
    print(f"  KV Sequence Length: {SEQ_LEN_KV}")
    print(f"  Head Dimension (D_k): {D_K}")
    print(f"  Value Dimension (D_v): {D_V}")
    print(f"  Data Type: {DTYPE}")
    print(f"Input Q shape: {Q_input.shape}")
    print(f"Input K shape: {K_input.shape}")
    print(f"Input V shape: {V_input.shape}")

    # Test 1: Non-Causal Attention
    print("\n--- Test 1: Non-Causal Attention ---")
    output_fmha_cutile_non_causal = cutile_fmha(
        Q=Q_input, K=K_input, V=V_input,
        tile_m=128, tile_n=128,  # Increased tile sizes
        causal=False,
        query_group_size=QUERY_GROUP_SIZE
    )
    print(f"""cuTile FMHA Output shape (Non-Causal):{output_fmha_cutile_non_causal.shape},
        dtype:{output_fmha_cutile_non_causal.dtype}""")
    if args.correctness_check:
        ref_fmha = torch_fmha(Q_input, K_input, V_input,
                              is_causal=False, enable_gqa=enable_gqa)
        torch.testing.assert_close(output_fmha_cutile_non_causal, ref_fmha, atol=1e-3, rtol=1e-3)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # Test 2: Causal Attention
    print("\n--- Test 2: Causal Attention ---")
    output_fmha_cutile_causal = cutile_fmha(
        Q=Q_input, K=K_input, V=V_input,
        tile_m=128, tile_n=128,  # Increased tile sizes
        causal=True,
        query_group_size=QUERY_GROUP_SIZE
    )
    print(f"""cuTile FMHA Output shape (Causal): {output_fmha_cutile_causal.shape},
            dtype: {output_fmha_cutile_causal.dtype}""")
    if args.correctness_check:
        ref_fmha = torch_fmha(Q_input, K_input, V_input,
                              is_causal=True, enable_gqa=enable_gqa)
        torch.testing.assert_close(output_fmha_cutile_causal, ref_fmha, atol=1e-3, rtol=1e-3)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # Test 3: Causal Attention with autotuning and performance benchmarking.
    if ct_experimental is None:
        print("cuda.tile_experimental not available, skipping autotuning test")
        sys.exit(0)

    print("\n--- Test 3: Causal Attention with autotuning and performance benchmarking ---")
    # --- Increase the problem size ---
    BATCH_SIZE = 8
    NUM_HEADS = 16
    SEQ_LEN_Q = 1024
    SEQ_LEN_KV = 1024
    D_K = 64
    D_V = 64
    QUERY_GROUP_SIZE = 1

    Q_input = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, D_K, dtype=DTYPE, device='cuda')
    K_input = torch.randn(BATCH_SIZE, NUM_HEADS // QUERY_GROUP_SIZE, SEQ_LEN_KV, D_K,
                          dtype=DTYPE, device='cuda')
    V_input = torch.randn(BATCH_SIZE, NUM_HEADS // QUERY_GROUP_SIZE, SEQ_LEN_KV, D_V,
                          dtype=DTYPE, device='cuda')
    print("New Configuration:")
    print(f"Input Q shape: {Q_input.shape}")
    print(f"Input K shape: {K_input.shape}")
    print(f"Input V shape: {V_input.shape}")
    output_fmha_cutile_autotune_causal, tuned_config = cutile_autotune_fmha(
        Q=Q_input, K=K_input, V=V_input,
        qk_scale=1.0 / math.sqrt(D_K),
        causal=True,
        query_group_size=QUERY_GROUP_SIZE
    )
    print(f"""cuTile FMHA Output shape (Causal): {output_fmha_cutile_autotune_causal.shape},
            dtype: {output_fmha_cutile_autotune_causal.dtype}""")
    print(f"Tuned config: {tuned_config}")
    if args.correctness_check:
        ref_fmha = torch_fmha(Q_input, K_input, V_input, is_causal=True, enable_gqa=enable_gqa)
        torch.testing.assert_close(
            output_fmha_cutile_autotune_causal, ref_fmha, atol=1e-2, rtol=5e-2
        )
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    stats_cutile_autotuned = report_benchmark(
        cutile_autotune_fmha,
        (Q_input, K_input, V_input, 1.0 / math.sqrt(D_K), 0, QUERY_GROUP_SIZE, True)
    )
    stats_torch = report_benchmark(
        torch_fmha,
        (Q_input, K_input, V_input, True, False)
    )
    print("Benchmark results:")
    print(f"  cuTile FMHA with tuned parameters: {stats_cutile_autotuned['mean_time_ms']:.5f} ms")
    print(f"  torch FMHA: {stats_torch['mean_time_ms']:.5f} ms")
    speedup_autotuned = stats_torch["mean_time_ms"] / stats_cutile_autotuned["mean_time_ms"]
    print(f"Speedup with autotuned parameters: {speedup_autotuned:.3f}x")

    print("\n--- cuTile Fused Multi-Head Attention (FMHA) Sample execution complete ---")
