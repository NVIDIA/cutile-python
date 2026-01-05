# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import numpy as np
import math

from cuda.tile import RoundingMode as RMd
from cuda.tile._numeric_semantics import PaddingMode

INV_LOG_2 = 1.0 / math.log(2)


# Define type aliases for Constant integers and booleans
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
allow_tma = True

# --- FMHA Kernel Implementation ---
@ct.kernel(occupancy=2)
def fmha_kernel(Q, K, V, Out, Lse, qk_scale: float, input_pos: int, TILE_D: ConstInt, 
                H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, QUERY_GROUP_SIZE: ConstInt,
                CAUSAL: ConstBool, EVEN_K: ConstBool):
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
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)
    lse = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    # Load query tile for this batch, head, and M-chunk
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D), allow_tma=allow_tma
    ).reshape((TILE_M, TILE_D))  # [TILE_M, TILE_D]

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
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
            allow_tma=allow_tma
        )
        k = k.reshape((TILE_D, TILE_N))  # [TILE_D, TILE_N]
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
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
            latency=4,
            allow_tma=allow_tma
        ).reshape((TILE_N, TILE_D))  # [TILE_N, TILE_D]
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)  # [TILE_M, TILE_N]
        m_i = m_ij  # [TILE_M, 1]

    # --- Final Normalization and Store ---
    lse = m_i + ct.log2(l_i)
    lse = lse.reshape((1, 1, TILE_M))
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Lse, index=(batch_idx, head_idx, bid_x), tile=lse)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


@ct.kernel(occupancy=2)
def fmha_bwd_dq_kernel(Q, K, V, Grad, Delta, Lse, DQ, qk_scale: float, input_pos: int, TILE_D: ConstInt, 
                       H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, QUERY_GROUP_SIZE: ConstInt,
                       CAUSAL: ConstBool, EVEN_K: ConstBool):
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)  # [TILE_M]
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)  # [TILE_N]
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Load query tile for this batch, head, and M-chunk
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D), allow_tma=allow_tma, padding_mode=PaddingMode.ZERO
    ).reshape((TILE_M, TILE_D))  # [TILE_M, TILE_D
    lse_i = ct.load(Lse, index=(batch_idx, head_idx, bid_x), shape=(1, 1, TILE_M),
                    allow_tma=allow_tma).reshape((TILE_M, 1))
    delta_i = ct.load(Delta, index=(batch_idx, head_idx, bid_x), shape=(1, 1, TILE_M),
                      allow_tma=allow_tma).reshape((TILE_M, 1))
    do = ct.load(
        Grad, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D), allow_tma=allow_tma, padding_mode=PaddingMode.ZERO
    ).reshape((TILE_M, TILE_D))  # [TILE_M, TILE_D]
    dq = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32) # [TILE_M, TILE_D]
    # loop over k, v and update accumulator
    k_seqlen = K.shape[2]
    if CAUSAL:
        m_end = input_pos + (bid_x + 1) * TILE_M
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = Tc
        if not EVEN_K:
            mask_start = Tc - 1
    for j in range(0, mask_start):
        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
            allow_tma=allow_tma, padding_mode=PaddingMode.ZERO
        ).reshape((TILE_D, TILE_N))  # [TILE_D, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]
        qk = qk * qk_scale * INV_LOG_2
        p = ct.exp2(qk - lse_i, flush_to_zero=True) # [TILE_M, TILE_N]

        v = ct.load(
            V, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=4,
            allow_tma=allow_tma, padding_mode=PaddingMode.ZERO
        ).reshape((TILE_D, TILE_N)) # [TILE_D, TILE_N]

        dp = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32) # [TILE_M, TILE_N]
        dp = ct.mma(do, v, dp)  # [TILE_M, TILE_N]
        dp = dp - delta_i
        ds = p * dp # [TILE_M, TILE_N]
        ds = ds.astype(k.dtype)

        dskt = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32) # [TILE_M, TILE_D]
        kt = ct.permute(k, (1, 0)) # [TILE_N, TILE_D]
        dskt = ct.mma(ds, kt, dskt) # [TILE_M, TILE_D]
        dq = dq + dskt # [TILE_M, TILE_D]

    for j in range(mask_start, Tc):
        # --- Compute QK product ---
        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
            allow_tma=allow_tma, padding_mode=PaddingMode.ZERO
        )
        k = k.reshape((TILE_D, TILE_N))  # [TILE_D, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]

        # --- Apply Causal Masking ---
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

        qk = qk * qk_scale * INV_LOG_2
        p = ct.exp2(qk - lse_i, flush_to_zero=True) # [TILE_M, TILE_N]

        v = ct.load(
            V, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=4,
            allow_tma=allow_tma, padding_mode=PaddingMode.ZERO
        ).reshape((TILE_D, TILE_N)) # [TILE_D, TILE_N]

        dp = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32) # [TILE_M, TILE_N]
        dp = ct.mma(do, v, dp)  # [TILE_M, TILE_N]
        dp = dp - delta_i
        ds = p * dp # [TILE_M, TILE_N]
        ds = ds.astype(k.dtype)

        dskt = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32) # [TILE_M, TILE_D]
        kt = ct.permute(k, (1, 0)) # [TILE_N, TILE_D]
        dskt = ct.mma(ds, kt, dskt) # [TILE_M, TILE_D]
        dq = dq + dskt # [TILE_M, TILE_D]

    dq = dq * qk_scale
    dq = dq.astype(q.dtype).reshape((1, 1, TILE_M, TILE_D))
    ct.store(DQ, index=(batch_idx, head_idx, bid_x, 0), tile=dq)


@ct.kernel(occupancy=2)
def fmha_bwd_dk_dv_kernel(Q, K, V, Grad, Delta, Lse, DK, DV, qk_scale: float, input_pos: int, TILE_D: ConstInt, 
                          H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, QUERY_GROUP_SIZE: ConstInt,
                          CAUSAL: ConstBool, EVEN_K: ConstBool):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    # Adjust qk_scale for exp2
    k = ct.load(
            K, index=(batch_idx, head_idx, 0, bid_x), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            allow_tma=allow_tma, padding_mode=PaddingMode.ZERO
        ).reshape((TILE_D, TILE_N))
    v = ct.load(
            V, index=(batch_idx, head_idx, 0, bid_x), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            allow_tma=allow_tma, padding_mode=PaddingMode.ZERO
        ).reshape((TILE_D, TILE_N))

    dk = ct.full((TILE_N, TILE_D), 0.0, dtype=np.float32) # [TILE_N, TILE_D] 
    dv = ct.full((TILE_N, TILE_D), 0.0, dtype=np.float32) # [TILE_N, TILE_D]
    
    # Initialize local offsets for query/key/value tile (M or N-dimension)
    offs_m_tile = ct.arange(TILE_M, dtype=np.int32) + input_pos # [TILE_M]
    offs_m_tile = offs_m_tile[:, None]  # [TILE_M, 1]
    offs_n = bid_x * TILE_N + ct.arange(TILE_N, dtype=np.int32)
    offs_n = offs_n[None, :] 
    Tr = ct.cdiv(Q.shape[2], TILE_M)
    
    if CAUSAL:
        m_start = bid_x * TILE_N // TILE_M
        mask_end = ct.cdiv((bid_x + 1) * TILE_N, TILE_M)
    else:
        m_start = 0
    for j in range(QUERY_GROUP_SIZE):
        for i in range(m_start, Tr):
            q = ct.load(Q, index=(batch_idx, head_idx * QUERY_GROUP_SIZE + j, i, 0), shape=(1, 1, TILE_M, TILE_D), 
                        latency=2, allow_tma=allow_tma, padding_mode=PaddingMode.ZERO).reshape((TILE_M, TILE_D))   # [TILE_M, TILE_D]
            qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
            qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]
            if (CAUSAL or not EVEN_K) and i <= mask_end:
                mask = ct.full((TILE_M, TILE_N), True, dtype=np.bool_)
                offs_m = i * TILE_M + offs_m_tile
                mask = mask & (offs_m >= offs_n)
                mask = ct.where(mask, 0.0, -np.inf)
                qk += mask

            lse_i =  ct.load(Lse, index=(batch_idx, head_idx * QUERY_GROUP_SIZE + j, i), shape=(1, 1, TILE_M),
                        allow_tma=allow_tma, padding_mode=PaddingMode.ZERO).reshape((TILE_M, 1))
            qk = qk * qk_scale * INV_LOG_2
            p = ct.exp2(qk - lse_i, flush_to_zero=True) # [TILE_M, TILE_N]
            pt = ct.permute(p, (1, 0)) # [TILE_N, TILE_M]

            do = ct.load(Grad, index=(batch_idx, head_idx * QUERY_GROUP_SIZE + j, i, 0), shape=(1, 1, TILE_M, TILE_D), 
                latency=4, allow_tma=allow_tma, padding_mode=PaddingMode.ZERO).reshape((TILE_M, TILE_D))  # [TILE_M, TILE_D]
            pt = pt.astype(do.dtype)
            dv = ct.mma(pt, do, dv) # [TILE_N, TILE_D]

            dp = ct.full((TILE_M, TILE_N), 0., dtype=np.float32) # [TILE_M, TILE_N]
            dp = ct.mma(do, v, dp)  # [TILE_M, TILE_N]
            delta_i = ct.load(Delta, index=(batch_idx, head_idx * QUERY_GROUP_SIZE + j, i), shape=(1, 1, TILE_M),
                        allow_tma=allow_tma, padding_mode=PaddingMode.ZERO).reshape((TILE_M, 1))
            dp = dp - delta_i
            ds = p * dp # [TILE_M, TILE_N]
            dst = ct.permute(ds, (1, 0)) # [TILE_N, TILE_M]
            dst = dst.astype(q.dtype)
            dk = ct.mma(dst, q, dk) # [TILE_N, TILE_D]

    dk = dk * qk_scale
    dk = dk.astype(k.dtype).reshape((1, 1, TILE_N, TILE_D))
    dv = dv.astype(v.dtype).reshape((1, 1, TILE_N, TILE_D))
    ct.store(DK, index=(batch_idx, head_idx, bid_x, 0), tile=dk)
    ct.store(DV, index=(batch_idx, head_idx, bid_x, 0), tile=dv)


@ct.kernel(occupancy=1)
def fmha_bwd_preprocess_kernel(O, Grad, Delta, 
                               H: ConstInt,
                               TILE_M: ConstInt, 
                               TILE_D: ConstInt):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    o = ct.load(O, index=(batch_idx, head_idx, bid_x, 0), 
                shape=(1, 1, TILE_M, TILE_D), 
                latency=2, allow_tma=allow_tma
        ).reshape((TILE_M, TILE_D))
    do = ct.load(Grad, index=(batch_idx, head_idx, bid_x, 0), 
                 shape=(1, 1, TILE_M, TILE_D), 
                 latency=2, allow_tma=allow_tma
         ).reshape((TILE_M, TILE_D))
    delta = ct.mul(o.astype(ct.float32), do.astype(ct.float32), flush_to_zero=True)
    delta = ct.sum(delta, axis=1).reshape((1, 1, TILE_M))
    ct.store(Delta, index=(batch_idx, head_idx, bid_x), tile=delta)