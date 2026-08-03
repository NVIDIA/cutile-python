# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Blackwell FMHA-prefill port from CUTLASS Python DSL to CUDA Lang.

This is a native fused implementation, with no SIMT, PyTorch-SDPA, library,
source-kernel, or precompiled-PTX fallback.  It supports the source tutorial's
fixed and variable sequence storage, D=32/64/128, FP16/BF16/E4M3 inputs and
outputs, MHA/GQA, mask variants, LSE, sink logits, scaling controls, and
correction/epilogue options.

The device kernel retains the source's structural contract:

* one 512-thread CTA and one CTA per ``(256 Q rows, Q head, batch)`` tile;
* softmax warps 0-7, correction warps 8-11, MMA warp 12, TMA warp 13,
  epilogue warp 14, and CLC scheduler warp 15;
* CTA-group-1 tcgen05 QK (128x128x16) and PV (128x64x16);
* two Q stages, three unified K/V stages, two output stages, and all 512
  TMEM columns with the source S/P/O aliases;
* online softmax, packed-FP16 P in TMEM, FP32 O correction in TMEM, TMA
  global/shared transfers, and a one-slot 480-consumer CLC mailbox.

The source-compatible host configuration, chunked reference, correctness
matrix, cold-L2/CUDA-graph benchmark, and CLI share this public entry point.
Invalid combinations raise clear diagnostics and are never silently redirected
to another attention implementation.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, TypeAlias

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability
import torch

import fmha_prefill_helpers as fmha_utils


# ---------------------------------------------------------------------------
# Authoritative specialization and role/resource map.
# ---------------------------------------------------------------------------

WARP_SIZE = 32
THREADS_PER_CTA = 16 * WARP_SIZE
SOFTMAX0_WARPS = (0, 1, 2, 3)
SOFTMAX1_WARPS = (4, 5, 6, 7)
CORRECTION_WARPS = (8, 9, 10, 11)
MMA_WARP = 12
LOAD_WARP = 13
EPILOGUE_WARP = 14
SCHEDULER_WARP = 15

MMA_M = 128
MMA_N = 128
HEAD_DIM = 64
CTA_M = 2 * MMA_M
MMA_K = 16
Q_STAGES = 2
KV_STAGES = 3
P_READY_STAGES = 2
O_STAGES = 2
TMEM_COLUMNS = 512

TMEM_S0 = 0
TMEM_S1 = 128
TMEM_P0 = 32
TMEM_P1 = 160
TMEM_O0 = 256
TMEM_O1 = 384

Q_STAGE_ELEMENTS = MMA_M * HEAD_DIM
KV_STAGE_ELEMENTS = MMA_N * HEAD_DIM
O_STAGE_ELEMENTS = MMA_M * HEAD_DIM
TMA_TILE_BYTES = MMA_M * HEAD_DIM * 2
TMA_KV_L2_CACHE_HINT = 0x1000000000000000
CLC_BYTES = cl.clusterlaunchcontrol_token.bitwidth // 8

TMEM_ALLOC_BARRIER = 2
TMEM_ALLOC_THREADS = (len(CORRECTION_WARPS) + 1) * WARP_SIZE
SOFTMAX_SINK_BARRIER = 3
SOFTMAX_SEQUENCE_0 = 4
SOFTMAX_SEQUENCE_1 = 5
SOFTMAX_WG_0 = 6
SOFTMAX_WG_1 = 7
CORRECTION_STORE_BARRIER = 8

DEFAULT_Q_SHAPE = (4, 1024, 8, 64)
DEFAULT_K_SHAPE = (4, 1024, 8, 64)
DEFAULT_MMA_TILER = (128, 128)
DEFAULT_TOLERANCE = 1.0e-1

ELEMENT_F16 = 0
ELEMENT_BF16 = 1
ELEMENT_E4M3 = 2

ShapeSpec: TypeAlias = tuple[int, int | tuple[int, ...], int, int]


class UnsupportedFmhaConfiguration(RuntimeError):
    """Raised instead of using a structurally non-equivalent fallback."""


cc = get_compute_capability()
if cc.major != 10:
    raise RuntimeError(
        "CUDA Lang FMHA requires Blackwell compute capability 10.x; "
        f"detected compute capability {cc.major}.{cc.minor}"
    )


def _wait_mbarrier(mbar, phase):
    cl.mbarrier_wait_parity(mbar, phase, time_hint=10_000_000)


def _fence_acq_rel_cta():
    """Order generic shared-memory mailbox accesses at CTA scope."""
    cl.fence(cl.MemoryOrder.ACQ_REL, cl.MemoryScope.BLOCK)


def _tmem_pointer(base, lane_offset, column_offset):
    return cl.tcgen05_tmem_offset(
        base,
        lane_offset=lane_offset,
        column_offset=column_offset,
    )


def _fma_packed_f32x2(a0, a1, b0, b1, c0, c1):
    """Issue the source kernel's packed two-lane FP32 FMA."""
    result = cl._inline_ptx(
        """
        {
            .reg .b64 a, b, c, d;
            mov.b64 a, {%2, %3};
            mov.b64 b, {%4, %5};
            mov.b64 c, {%6, %7};
            fma.rn.f32x2 d, a, b, c;
            mov.b64 {%0, %1}, d;
        }
        """,
        ("=f", cl.float32),
        ("=f", cl.float32),
        ("f", a0),
        ("f", a1),
        ("f", b0),
        ("f", b1),
        ("f", c0),
        ("f", c1),
    )
    return cl.Vector(result[0], result[1], dtype=cl.float32)


def _add_packed_f32x2(a0, a1, b0, b1):
    """Issue the source kernel's packed two-lane FP32 add."""
    return cl._inline_ptx(
        """
        {
            .reg .b64 a, b, c;
            mov.b64 a, {%2, %3};
            mov.b64 b, {%4, %5};
            add.rn.f32x2 c, a, b;
            mov.b64 {%0, %1}, c;
        }
        """,
        ("=f", cl.float32),
        ("=f", cl.float32),
        ("f", a0),
        ("f", a1),
        ("f", b0),
        ("f", b1),
    )


def _pack_e4m3_quad(values):
    lo = cl.uint32(cl.uint16(cl._nvvm.ff_to_e4m3x2_rn(values[1], values[0])))
    hi = cl.uint32(cl.uint16(cl._nvvm.ff_to_e4m3x2_rn(values[3], values[2])))
    return cl.int32(lo | (hi << 16))


def _compute_p_fma_window(
    scores,
    window_start,
    p_window_elems,
    scale,
    minus_row_max_scale,
):
    affine01 = _fma_packed_f32x2(
        scores[window_start],
        scores[window_start + 1],
        scale,
        scale,
        minus_row_max_scale,
        minus_row_max_scale,
    )
    if p_window_elems == 2:
        return affine01
    affine23 = _fma_packed_f32x2(
        scores[window_start + 2],
        scores[window_start + 3],
        scale,
        scale,
        minus_row_max_scale,
        minus_row_max_scale,
    )
    return cl.Vector(
        affine01[0],
        affine01[1],
        affine23[0],
        affine23[1],
        dtype=cl.float32,
    )


def _materialize_p_fma_window(p_fma, p_window_elems):
    """Force a P-FMA window into PTX at the current program point."""
    materialized = tuple(
        cl._inline_ptx(
            "mov.b32 %0, %1;",
            ("=f", cl.float32),
            ("f", p_fma[item]),
        )[0]
        for item in cl.static_iter(range(p_window_elems))
    )
    return cl.Vector(*materialized, dtype=cl.float32)


def _exp_p_window(p_fma, p_window_elems):
    cl.static_assert(p_fma.element_count == p_window_elems)
    probabilities = tuple(
        cl.exp2(p_fma[item], flush_to_zero=True)
        for item in cl.static_iter(range(p_window_elems))
    )
    return cl.Vector(*probabilities, dtype=cl.float32)


def _pack_p_window(probabilities, input_kind):
    if input_kind == ELEMENT_E4M3:
        return _pack_e4m3_quad(probabilities)
    if input_kind == ELEMENT_BF16:
        return probabilities.astype(cl.bfloat16).reinterpret_as_scalar(cl.int32)
    return probabilities.astype(cl.float16).reinterpret_as_scalar(cl.int32)


def _accumulate_p_window_sum(
    block_sum0,
    block_sum1,
    probabilities,
    p_window_elems,
):
    for pair in cl.static_iter(range(p_window_elems // 2)):
        block_sum0, block_sum1 = _add_packed_f32x2(
            probabilities[2 * pair],
            probabilities[2 * pair + 1],
            block_sum0,
            block_sum1,
        )
    return block_sum0, block_sum1


def _qk_descriptor(pointer, head_dim, element_bits, tma_slices, swizzle):
    """SM100 descriptor for one 128-row Q/K tile."""
    return cl.Tcgen05SharedMemoryDescriptor(
        matrix_start_address=pointer,
        leading_dimension_byte_offset=0,
        stride_dimension_byte_offset=head_dim * element_bits // tma_slices,
        swizzle_mode=swizzle,
    ).encode()


def _pv_descriptor(
    pointer, head_dim, element_bits, tma_slices, swizzle, tile_bytes
):
    """SM100 descriptor for V as the B-major operand of P*V."""
    leading_offset = 0
    if tma_slices > 1:
        leading_offset = tile_bytes // tma_slices
    return cl.Tcgen05SharedMemoryDescriptor(
        matrix_start_address=pointer,
        leading_dimension_byte_offset=leading_offset,
        stride_dimension_byte_offset=head_dim * element_bits // tma_slices,
        swizzle_mode=swizzle,
    ).encode()


def _scheduler_divrem(value, divisor):
    """Return quotient and remainder for nonnegative scheduler values."""
    quotient = value // divisor
    remainder = value - quotient * divisor
    return quotient, remainder


def _decode_work_tile(tile_id, seq_tiles, batch_count, head_count, is_causal):
    """Decode the source dense or cuDNN-style causal CLC order."""
    total_tiles = seq_tiles * batch_count * head_count
    _, safe_tile_id = _scheduler_divrem(tile_id, total_tiles)
    if is_causal:
        hb_count = batch_count * head_count
        group_size = hb_count
        if group_size > 8:
            group_size = 8
        group_tile_count = seq_tiles * group_size
        hb_group, group_offset = _scheduler_divrem(
            safe_tile_id, group_tile_count
        )
        hb_group_start = hb_group * group_size
        group_hb_count = hb_count - hb_group_start
        if group_hb_count > group_size:
            group_hb_count = group_size
        raw_seq, hb_residual = _scheduler_divrem(
            group_offset, group_hb_count
        )
        hb = hb_group_start + hb_residual
        batch, head = _scheduler_divrem(hb, head_count)
        seq_tile = seq_tiles - 1 - raw_seq
    else:
        hb, seq_tile = _scheduler_divrem(safe_tile_id, seq_tiles)
        batch, head = _scheduler_divrem(hb, head_count)
    return seq_tile, head, batch


def _query_next_work_tile(
    clc_token,
    clc_full,
    clc_empty,
    clc_full_phase,
    clc_empty_phase,
    elect_one,
    seq_tiles,
    batch_count,
    head_count,
    is_causal,
):
    """Run one CLC full/empty transaction and decode its response."""
    _wait_mbarrier(clc_empty, clc_empty_phase)
    if elect_one:
        cl.mbarrier_arrive_expect_transaction(
            clc_full,
            CLC_BYTES,
            scope=cl.MbarrierScope.BLOCK,
        )
        cl.clusterlaunchcontrol_try_cancel(clc_token.get_base_pointer(), clc_full)
    _wait_mbarrier(clc_full, clc_full_phase)

    token = clc_token[0]
    has_more = cl.clusterlaunchcontrol_is_canceled(token)
    next_linear = cl.int32(0)
    if has_more:
        next_linear = cl.clusterlaunchcontrol_get_first_block_index(token, axis=0)
    cl.fence_proxy(
        cl.FenceProxyKind.ASYNC_SHARED,
        space=cl.MemorySpace.SHARED,
    )
    next_seq, next_head, next_batch = _decode_work_tile(
        next_linear,
        seq_tiles,
        batch_count,
        head_count,
        is_causal,
    )
    cl.mbarrier_arrive(clc_empty, scope=cl.MbarrierScope.BLOCK)
    return (
        next_seq,
        next_head,
        next_batch,
        has_more,
        clc_full_phase ^ 1,
        clc_empty_phase ^ 1,
    )


def _consume_scheduled_tile(
    sched_tile,
    sched_valid,
    sched_full,
    sched_empty,
    full_phase,
):
    """Consume the scheduler's one-slot mailbox as one of 480 threads."""
    _wait_mbarrier(sched_full, full_phase)
    seq_tile = sched_tile[0]
    head = sched_tile[1]
    batch = sched_tile[2]
    valid = sched_valid[0] != cl.int32(0)
    _fence_acq_rel_cta()
    cl.mbarrier_arrive(sched_empty, scope=cl.MbarrierScope.BLOCK)
    return seq_tile, head, batch, valid, full_phase ^ 1


def _pack_output_values(values, scale, output_kind):
    """Scale and convert 32 FP32 TMEM values into packed output words."""
    values_f32 = values[:32]
    scaled_f32 = values_f32 * scale
    if output_kind == ELEMENT_E4M3:
        return cl.Vector(
            *tuple(
                cl.bitcast(
                    _pack_e4m3_quad(scaled_f32[4 * word:4 * word + 4]),
                    cl.uint32,
                )
                for word in cl.static_iter(range(8))
            ),
            dtype=cl.uint32,
        )
    if output_kind == ELEMENT_BF16:
        converted = scaled_f32.astype(cl.bfloat16)
        return cl.Vector(
            *tuple(
                converted[2 * word:2 * word + 2].reinterpret_as_scalar(cl.uint32)
                for word in cl.static_iter(range(16))
            ),
            dtype=cl.uint32,
        )
    converted = scaled_f32.astype(cl.float16)
    return cl.Vector(
        *tuple(
            converted[2 * word:2 * word + 2].reinterpret_as_scalar(cl.uint32)
            for word in cl.static_iter(range(16))
        ),
        dtype=cl.uint32,
    )


def _store_output_values(
    o_stage,
    packed_words,
    row,
    column,
    head_dim,
    output_kind,
    output_bits,
    output_tma_slices,
):
    """Store packed output words into the output's swizzled SMEM tile."""
    out_ptr = cl.bitcast(
        o_stage,
        cl.pointer_dtype(cl.uint32, cl.MemorySpace.SHARED),
    )
    slice_elements = head_dim // output_tma_slices
    slice_bytes = slice_elements * output_bits // 8
    swizzle_mask = (slice_bytes // 16 - 1) << 7
    slice_index = column // slice_elements
    column_in_slice = column % slice_elements
    word_count = 16
    if output_kind == ELEMENT_E4M3:
        word_count = 8
    # One PTX shared-memory vector store carries four 32-bit registers.  The
    # swizzle is constant within each 16-byte group, so form the same v4
    # transfer units as CUTLASS's store_swizzled lowering.
    for group in cl.static_iter(range(word_count // 4)):
        byte_offset = (
            slice_index * MMA_M * slice_bytes
            + row * slice_bytes
            + column_in_slice * output_bits // 8
            + group * 16
        )
        swizzled = byte_offset ^ (((byte_offset & swizzle_mask) >> 7) << 4)
        words = packed_words[group * 4:group * 4 + 4]
        (out_ptr + swizzled // 4).store(words, alignment=16)


def _store_output_global(
    output,
    packed_words,
    query_row,
    query_base,
    head,
    batch,
    column,
    output_kind,
    variable_length,
):
    """Guarded tail-row store to a fixed or flattened output view."""
    if variable_length:
        dst = output.get_element_pointer((column, head, query_base + query_row))
    else:
        dst = output.get_element_pointer((column, head, query_row, batch))

    if output_kind == ELEMENT_E4M3:
        dst_u8 = cl.bitcast(
            dst,
            cl.pointer_dtype(cl.uint8, cl.MemorySpace.GLOBAL),
        )
        for word in cl.static_iter(range(8)):
            packed = packed_words[word]
            for byte in cl.static_iter(range(4)):
                (dst_u8 + word * 4 + byte).store(
                    cl.uint8(packed >> (byte * 8)),
                    alignment=1,
                )
    else:
        # CUTLASS keeps the converted FP16/BF16 lanes scalar on the guarded
        # tail path, which lowers to one b16 store per valid element.
        dst_u16 = cl.bitcast(
            dst,
            cl.pointer_dtype(cl.uint16, cl.MemorySpace.GLOBAL),
        )
        for word in cl.static_iter(range(16)):
            packed = packed_words[word]
            (dst_u16 + word * 2).store(cl.uint16(packed), alignment=2)
            (dst_u16 + word * 2 + 1).store(
                cl.uint16(packed >> 16),
                alignment=2,
            )


def _mask_score(
    score,
    query_index,
    key_index,
    seq_q,
    seq_k,
    bottom_right_align,
    window_left,
    window_right,
):
    valid = query_index < seq_q and key_index < seq_k
    offset = 0
    if bottom_right_align:
        offset = seq_k - seq_q
    if window_left >= 0:
        valid = valid and key_index >= query_index + offset - window_left
    if window_right >= 0:
        valid = valid and key_index <= query_index + offset + window_right
    result = score
    if not valid:
        result = cl.float32(-float("inf"))
    return result


def _sequence_context(
    cumulative_q,
    cumulative_k,
    batch,
    seq_q,
    seq_k,
    variable_length,
):
    """Resolve one scheduled batch to local lengths and flattened offsets."""
    query_base = cl.int32(0)
    key_base = cl.int32(0)
    task_seq_q = cl.int32(seq_q)
    task_seq_k = cl.int32(seq_k)
    if variable_length:
        query_base = cumulative_q[batch]
        key_base = cumulative_k[batch]
        task_seq_q = cumulative_q[batch + 1] - query_base
        task_seq_k = cumulative_k[batch + 1] - key_base
    return query_base, key_base, task_seq_q, task_seq_k


def _key_trip_context(
    seq_tile,
    seq_q,
    seq_k,
    bottom_right_align,
    window_left,
    window_right,
):
    """Return the source super-tile K interval and its Q0 invalid tail.

    CUTLASS prunes whole 128-column K/V tiles using the union of the two
    128-row Q halves.  Q0 can therefore have one fully-invalid trailing tile
    that Q1 still needs.  That tile remains in the barrier protocol but does
    not issue QK0/PV0 math.
    """
    query_begin = seq_tile * CTA_M
    offset = cl.int32(0)
    if bottom_right_align:
        offset = seq_k - seq_q

    trip_start = cl.int32(0)
    if window_left >= 0:
        trip_start = (query_begin + offset - window_left) // MMA_N
        if trip_start < 0:
            trip_start = cl.int32(0)

    key_tile_limit = cl.cdiv(seq_k, MMA_N)
    trip_end = key_tile_limit
    q0_trip_end = key_tile_limit
    if window_right >= 0:
        trip_end = cl.cdiv(
            query_begin + CTA_M + offset + window_right,
            MMA_N,
        )
        if trip_end > key_tile_limit:
            trip_end = key_tile_limit
        q0_trip_end = cl.cdiv(
            query_begin + MMA_M + offset + window_right,
            MMA_N,
        )
        if q0_trip_end > key_tile_limit:
            q0_trip_end = key_tile_limit

    trip_count = trip_end - trip_start
    q0_trip_count = q0_trip_end - trip_start
    if q0_trip_count < 0:
        q0_trip_count = cl.int32(0)
    q0_trailing_invalid = trip_count - q0_trip_count > 0
    return trip_start, trip_count, q0_trailing_invalid


def _issue_qk(
    mma_kind,
    score_tmem,
    q_desc,
    k_desc,
    qk_instruction,
    score_ready,
    qk_phases,
    k_step,
    input_bits,
    input_tma_slices,
    input_tile_bytes,
    issue_math,
    elect_one,
):
    """Issue one QK tile, or only its source-compatible ready token."""
    if issue_math:
        qk_half = qk_phases // 2
        qk_step_bytes = k_step * input_bits // 8
        qk_extra_bytes = qk_step_bytes * qk_half
        if input_tma_slices > 1:
            qk_extra_bytes = input_tile_bytes // input_tma_slices
        for kk in cl.static_iter(range(qk_phases)):
            qk_byte_offset = qk_step_bytes * kk
            if kk >= qk_half:
                qk_byte_offset = qk_extra_bytes + qk_step_bytes * (kk - qk_half)
            descriptor_increment = qk_byte_offset >> 4
            if elect_one:
                cl.tcgen05_mma(
                    mma_kind,
                    score_tmem,
                    q_desc + descriptor_increment,
                    k_desc + descriptor_increment,
                    qk_instruction,
                    accumulate=kk != 0,
                    cta_group=cl.CTAGroup.CTA_1,
                )
    if cl.elect_sync():
        cl.tcgen05_commit(score_ready, cta_group=cl.CTAGroup.CTA_1)


def _issue_pv(
    mma_kind,
    output_tmem,
    p_tmem_base,
    v_desc,
    pv_instruction,
    p_full0,
    p_full1,
    p_empty0,
    p_empty1,
    o_full_mbar,
    o_empty_mbar,
    pv_event,
    key_ordinal,
    pv_phases_per_ready,
    k_step,
    head_dim,
    input_bits,
    input_tma_slices,
    issue_math,
    elect_one,
):
    """Consume one P tile and publish one O partial, optionally token-only."""
    _wait_mbarrier(o_empty_mbar, 1 ^ (pv_event & 1))
    for half in cl.static_iter(range(P_READY_STAGES)):
        p_full_mbar = p_full0
        p_empty_mbar = p_empty0
        if half == 1:
            p_full_mbar = p_full1
            p_empty_mbar = p_empty1
        _wait_mbarrier(p_full_mbar, pv_event & 1)
        if issue_math:
            for local_kk in cl.static_iter(range(pv_phases_per_ready)):
                kk = half * pv_phases_per_ready + local_kk
                p_tmem = _tmem_pointer(
                    p_tmem_base,
                    0,
                    kk * 8,
                )
                if elect_one:
                    cl.tcgen05_mma(
                        mma_kind,
                        output_tmem,
                        p_tmem,
                        v_desc
                        + (
                            k_step
                            * (head_dim // input_tma_slices)
                            * input_bits
                            // 8
                            >> 4
                        )
                        * kk,
                        pv_instruction,
                        accumulate=key_ordinal != 0 or kk != 0,
                        cta_group=cl.CTAGroup.CTA_1,
                    )
        # A tcgen commit is required for real consumers and is conservatively
        # retained for the token-only Q0 tail to preserve the proven-safe ring
        # reuse ordering.
        if cl.elect_sync():
            cl.tcgen05_commit(p_empty_mbar, cta_group=cl.CTAGroup.CTA_1)
    if cl.elect_sync():
        cl.tcgen05_commit(o_full_mbar, cta_group=cl.CTAGroup.CTA_1)


@cl.kernel(
    max_threads_per_block=(THREADS_PER_CTA,),
    min_blocks_per_sm=1,
)
def _fmha_prefill_kernel(
    q,
    k,
    v,
    o,
    lse,
    sinks,
    cumulative_q,
    cumulative_k,
    batch_count: cl.Constant[int],
    seq_q: cl.Constant[int],
    seq_k: cl.Constant[int],
    total_q: cl.Constant[int],
    heads_q: cl.Constant[int],
    heads_k: cl.Constant[int],
    head_dim: cl.Constant[int],
    scheduler_seq_tiles: cl.int32,
    scheduler_batch_count: cl.int32,
    scheduler_head_count: cl.int32,
    input_kind: cl.Constant[int],
    output_kind: cl.Constant[int],
    scale_softmax_log2: cl.float32,
    scale_softmax: cl.float32,
    scale_output: cl.float32,
    calculate_lse: cl.Constant[bool],
    use_sinks: cl.Constant[bool],
    enable_skip_correction: cl.Constant[bool],
    enable_approx_epilogue_rcp: cl.Constant[bool],
    causal_scheduler: cl.Constant[bool],
    bottom_right_align: cl.Constant[bool],
    window_left: cl.Constant[int],
    window_right: cl.Constant[int],
    variable_length: cl.Constant[bool],
):
    """Fused CTA-1 FMHA for fixed or flattened variable-length inputs."""

    input_dtype = cl.float16
    input_bits = 16
    k_step = 16
    input_tma_slices = 1
    input_tma_granularity = head_dim
    input_swizzle = cl.SwizzleMode.SWIZZLE_128B
    input_l2 = cl.TensorMapL2Promotion.L2_128B
    mma_kind = cl.Tcgen05MMAKind.F16
    if cl.ensure_constant(input_kind == ELEMENT_BF16):
        input_dtype = cl.bfloat16
    if cl.ensure_constant(input_kind == ELEMENT_E4M3):
        input_dtype = cl.int8
        input_bits = 8
        k_step = 32
        mma_kind = cl.Tcgen05MMAKind.F8F6F4
    if cl.ensure_constant(input_bits == 16 and head_dim == 128):
        input_tma_slices = 2
        input_tma_granularity = 64
    if cl.ensure_constant(input_bits == 16 and head_dim == 32):
        input_swizzle = cl.SwizzleMode.SWIZZLE_64B
        input_l2 = cl.TensorMapL2Promotion.L2_64B
    if cl.ensure_constant(input_bits == 8 and head_dim == 32):
        input_swizzle = cl.SwizzleMode.SWIZZLE_32B
        input_l2 = cl.TensorMapL2Promotion.NONE
    if cl.ensure_constant(input_bits == 8 and head_dim == 64):
        input_swizzle = cl.SwizzleMode.SWIZZLE_64B
        input_l2 = cl.TensorMapL2Promotion.L2_64B

    output_dtype = cl.float16
    output_bits = 16
    output_tma_slices = 1
    output_tma_granularity = head_dim
    output_swizzle = cl.SwizzleMode.SWIZZLE_128B
    if cl.ensure_constant(output_kind == ELEMENT_BF16):
        output_dtype = cl.bfloat16
    if cl.ensure_constant(output_kind == ELEMENT_E4M3):
        output_dtype = cl.int8
        output_bits = 8
    if cl.ensure_constant(output_bits == 16 and head_dim == 128):
        output_tma_slices = 2
        output_tma_granularity = 64
    if cl.ensure_constant(output_bits == 16 and head_dim == 32):
        output_swizzle = cl.SwizzleMode.SWIZZLE_64B
    if cl.ensure_constant(output_bits == 8 and head_dim == 32):
        output_swizzle = cl.SwizzleMode.SWIZZLE_32B
    if cl.ensure_constant(output_bits == 8 and head_dim == 64):
        output_swizzle = cl.SwizzleMode.SWIZZLE_64B

    q_stage_elements = MMA_M * head_dim
    kv_stage_elements = MMA_N * head_dim
    o_stage_elements = MMA_M * head_dim
    input_tile_bytes = MMA_M * head_dim * input_bits // 8
    qk_phases = head_dim // k_step
    pv_phases = MMA_N // k_step
    pv_phases_per_ready = pv_phases // P_READY_STAGES

    # Source-sized shared-memory payloads. K and V intentionally share the
    # three-stage buffer; S/P/stats intentionally alias in TMEM.
    q_smem = cl.shared_array(
        (Q_STAGES, q_stage_elements),
        input_dtype,
        dynamic=True,
        alignment=1024,
    )
    kv_smem = cl.shared_array(
        (KV_STAGES, kv_stage_elements),
        input_dtype,
        dynamic=True,
        alignment=1024,
    )
    o_smem = cl.shared_array(
        (O_STAGES, o_stage_elements),
        output_dtype,
        dynamic=True,
        alignment=1024,
    )
    # The source stages per-head sink logits through softmax0-owned SMEM once
    # before persistent work.  Keep it absent from non-sink specializations so
    # canonical shared-memory sizing remains identical to sinks=None.
    sinks_smem = None
    if cl.ensure_constant(use_sinks):
        sinks_smem = cl.shared_array(heads_q, cl.float16, alignment=16)

    q_full = cl.shared_array(Q_STAGES, cl.mbarrier, alignment=8)
    q_empty = cl.shared_array(Q_STAGES, cl.mbarrier, alignment=8)
    kv_full = cl.shared_array(KV_STAGES, cl.mbarrier, alignment=8)
    kv_empty = cl.shared_array(KV_STAGES, cl.mbarrier, alignment=8)
    score_full = cl.shared_array(2, cl.mbarrier, alignment=8)
    score_empty = cl.shared_array(2, cl.mbarrier, alignment=8)
    p_full = cl.shared_array((2, P_READY_STAGES), cl.mbarrier, alignment=8)
    p_empty = cl.shared_array((2, P_READY_STAGES), cl.mbarrier, alignment=8)
    stats_full = cl.shared_array(2, cl.mbarrier, alignment=8)
    stats_empty = cl.shared_array(2, cl.mbarrier, alignment=8)
    o_full = cl.shared_array(2, cl.mbarrier, alignment=8)
    o_empty = cl.shared_array(2, cl.mbarrier, alignment=8)
    epi_full = cl.shared_array(2, cl.mbarrier, alignment=8)
    epi_empty = cl.shared_array(2, cl.mbarrier, alignment=8)
    tmem_dealloc = cl.shared_array(1, cl.mbarrier, alignment=8)
    tmem_storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR),
        alignment=4,
    )

    clc_bar = cl.shared_array(1, cl.mbarrier, alignment=8)
    clc_empty = cl.shared_array(1, cl.mbarrier, alignment=8)
    clc_token = cl.shared_array(1, cl.clusterlaunchcontrol_token, alignment=16)
    sched_full = cl.shared_array(1, cl.mbarrier, alignment=8)
    sched_empty = cl.shared_array(1, cl.mbarrier, alignment=8)
    sched_tile = cl.shared_array(3, cl.int32, alignment=16)
    sched_valid = cl.shared_array(1, cl.int32, alignment=4)

    tid = cl.thread_index(0)
    # Match CUTLASS's make_warp_uniform(warp_idx): broadcasting lane 0's
    # value lets the backend treat every role predicate as warp-uniform.
    warp = cl.shfl_sync(tid // WARP_SIZE, 0)
    lane = tid % WARP_SIZE
    if variable_length:
        q_tmap = cl.tensor_map_tiled(
            q,
            (input_tma_granularity, 1, MMA_M),
            order=(0, 1, 2),
            swizzle=input_swizzle,
            l2_promotion=input_l2,
        )
        k_tmap = cl.tensor_map_tiled(
            k,
            (input_tma_granularity, 1, MMA_N),
            order=(0, 1, 2),
            swizzle=input_swizzle,
            l2_promotion=input_l2,
        )
        v_tmap = cl.tensor_map_tiled(
            v,
            (input_tma_granularity, 1, MMA_N),
            order=(0, 1, 2),
            swizzle=input_swizzle,
            l2_promotion=input_l2,
        )
        o_tmap = cl.tensor_map_tiled(
            o,
            (output_tma_granularity, 1, MMA_M),
            order=(0, 1, 2),
            swizzle=output_swizzle,
            l2_promotion=cl.TensorMapL2Promotion.NONE,
        )
    else:
        q_tmap = cl.tensor_map_tiled(
            q,
            (input_tma_granularity, 1, MMA_M, 1),
            order=(0, 1, 2, 3),
            swizzle=input_swizzle,
            l2_promotion=input_l2,
        )
        k_tmap = cl.tensor_map_tiled(
            k,
            (input_tma_granularity, 1, MMA_N, 1),
            order=(0, 1, 2, 3),
            swizzle=input_swizzle,
            l2_promotion=input_l2,
        )
        v_tmap = cl.tensor_map_tiled(
            v,
            (input_tma_granularity, 1, MMA_N, 1),
            order=(0, 1, 2, 3),
            swizzle=input_swizzle,
            l2_promotion=input_l2,
        )
        o_tmap = cl.tensor_map_tiled(
            o,
            (output_tma_granularity, 1, MMA_M, 1),
            order=(0, 1, 2, 3),
            swizzle=output_swizzle,
            l2_promotion=cl.TensorMapL2Promotion.NONE,
        )

    # CUTLASS initializes Blackwell pipeline stages cooperatively with warp 0.
    # Dynamic lane indexing keeps one static PTX site per pipeline half while
    # still initializing every physical barrier exactly once.
    if warp == SOFTMAX0_WARPS[0]:
        if lane < Q_STAGES:
            cl.mbarrier_initialize(q_full.get_element_pointer(lane), 1)
            cl.mbarrier_initialize(q_empty.get_element_pointer(lane), 1)
        if lane < KV_STAGES:
            cl.mbarrier_initialize(kv_full.get_element_pointer(lane), 1)
            cl.mbarrier_initialize(kv_empty.get_element_pointer(lane), 1)
        for qid in cl.static_iter(range(2)):
            if lane < 1:
                cl.mbarrier_initialize(score_full.get_element_pointer(qid), 1)
                cl.mbarrier_initialize(score_empty.get_element_pointer(qid), 128)
                cl.mbarrier_initialize(stats_full.get_element_pointer(qid), 128)
                cl.mbarrier_initialize(stats_empty.get_element_pointer(qid), 128)
            if lane < P_READY_STAGES:
                cl.mbarrier_initialize(
                    p_full.get_element_pointer((qid, lane)), 128
                )
                cl.mbarrier_initialize(
                    p_empty.get_element_pointer((qid, lane)), 1
                )
        if lane < O_STAGES:
            cl.mbarrier_initialize(o_full.get_element_pointer(lane), 1)
            cl.mbarrier_initialize(o_empty.get_element_pointer(lane), 128)
            cl.mbarrier_initialize(epi_full.get_element_pointer(lane), 128)
            cl.mbarrier_initialize(epi_empty.get_element_pointer(lane), WARP_SIZE)
        if lane < 1:
            cl.mbarrier_initialize(clc_bar.get_element_pointer(lane), 1)
            cl.mbarrier_initialize(
                clc_empty.get_element_pointer(lane), WARP_SIZE
            )

    # Non-pipeline barriers remain single-thread initialized.
    if warp == SCHEDULER_WARP and cl.elect_sync():
        cl.mbarrier_initialize(tmem_dealloc.get_base_pointer(), 384)
        cl.mbarrier_initialize(sched_full.get_base_pointer(), 1)
        cl.mbarrier_initialize(sched_empty.get_base_pointer(), 480)
    cl.fence_mbarrier_initialize()
    cl.barrier_sync_block()

    # ------------------------------------------------------------------ CLC
    if warp == SCHEDULER_WARP:
        cl.setmaxregister_decrease(32)
        scheduler_elect_one = cl.elect_sync()
        clc_full_phase = 0
        clc_empty_phase = 1
        sched_empty_phase = 0
        (
            next_seq,
            next_head,
            next_batch,
            has_more,
            clc_full_phase,
            clc_empty_phase,
        ) = _query_next_work_tile(
            clc_token,
            clc_bar.get_base_pointer(),
            clc_empty.get_base_pointer(),
            clc_full_phase,
            clc_empty_phase,
            scheduler_elect_one,
            scheduler_seq_tiles,
            scheduler_batch_count,
            scheduler_head_count,
            causal_scheduler,
        )
        if scheduler_elect_one:
            sched_tile[0] = next_seq
            sched_tile[1] = next_head
            sched_tile[2] = next_batch
            sched_valid[0] = cl.int32(has_more)
            _fence_acq_rel_cta()
            cl.mbarrier_arrive(
                sched_full.get_base_pointer(), scope=cl.MbarrierScope.BLOCK
            )

        while has_more:
            (
                next_seq,
                next_head,
                next_batch,
                has_more,
                clc_full_phase,
                clc_empty_phase,
            ) = _query_next_work_tile(
                clc_token,
                clc_bar.get_base_pointer(),
                clc_empty.get_base_pointer(),
                clc_full_phase,
                clc_empty_phase,
                scheduler_elect_one,
                scheduler_seq_tiles,
                scheduler_batch_count,
                scheduler_head_count,
                causal_scheduler,
            )
            _wait_mbarrier(
                sched_empty.get_base_pointer(),
                sched_empty_phase,
            )
            if scheduler_elect_one:
                sched_tile[0] = next_seq
                sched_tile[1] = next_head
                sched_tile[2] = next_batch
                sched_valid[0] = cl.int32(has_more)
                _fence_acq_rel_cta()
                cl.mbarrier_arrive(
                    sched_full.get_base_pointer(), scope=cl.MbarrierScope.BLOCK
                )
            sched_empty_phase ^= 1

    # -------------------------------------------------------------- TMA load
    elif warp == LOAD_WARP:
        cl.setmaxregister_decrease(32)
        if cl.elect_sync():
            cl.prefetch_tensor_map(q_tmap)
            cl.prefetch_tensor_map(k_tmap)
            cl.prefetch_tensor_map(v_tmap)
            cl.prefetch_tensor_map(o_tmap)

        seq_tile, head, batch = _decode_work_tile(
            cl.block_index(0),
            scheduler_seq_tiles,
            scheduler_batch_count,
            scheduler_head_count,
            causal_scheduler,
        )
        valid = True
        sched_phase = 0
        data_task = 0
        kv_index = 0
        kv_empty_phase = 1
        while valid:
            query_base, key_base, task_seq_q, task_seq_k = _sequence_context(
                cumulative_q,
                cumulative_k,
                batch,
                seq_q,
                seq_k,
                variable_length,
            )
            work_valid = True
            if variable_length:
                work_valid = seq_tile * CTA_M < task_seq_q
            if not work_valid:
                seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                    sched_tile,
                    sched_valid,
                    sched_full.get_base_pointer(),
                    sched_empty.get_base_pointer(),
                    sched_phase,
                )
                continue
            query_offset = seq_tile * CTA_M
            key_head = head // (heads_q // heads_k)
            q_phase = 1 ^ (data_task & 1)
            task_trip_start, task_trip_count, _ = _key_trip_context(
                seq_tile,
                task_seq_q,
                task_seq_k,
                bottom_right_align,
                window_left,
                window_right,
            )
            first_key_offset = task_trip_start * MMA_N

            # Q0
            _wait_mbarrier(q_empty.get_element_pointer(0), q_phase)
            if cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    q_full.get_element_pointer(0),
                    input_tile_bytes,
                    scope=cl.MbarrierScope.BLOCK,
                )
                for part in cl.static_iter(range(input_tma_slices)):
                    q_coordinate = (
                        part * input_tma_granularity,
                        head,
                        query_offset,
                        batch,
                    )
                    if variable_length:
                        q_coordinate = (
                            part * input_tma_granularity,
                            head,
                            query_base + query_offset,
                        )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        q_tmap,
                        q_coordinate,
                        q_smem.get_element_pointer(
                            (0, part * MMA_M * input_tma_granularity)
                        ),
                        q_full.get_element_pointer(0),
                    )

            # K0
            _wait_mbarrier(kv_empty.get_element_pointer(kv_index), kv_empty_phase)
            if cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    kv_full.get_element_pointer(kv_index),
                    input_tile_bytes,
                    scope=cl.MbarrierScope.BLOCK,
                )
                for part in cl.static_iter(range(input_tma_slices)):
                    k_coordinate = (
                        part * input_tma_granularity,
                        key_head,
                        first_key_offset,
                        batch,
                    )
                    if variable_length:
                        k_coordinate = (
                            part * input_tma_granularity,
                            key_head,
                            key_base + first_key_offset,
                        )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        k_tmap,
                        k_coordinate,
                        cl.map_shared_to_cluster(
                            kv_smem.get_element_pointer(
                                (
                                    kv_index,
                                    part * MMA_N * input_tma_granularity,
                                )
                            ),
                            0,
                        ),
                        kv_full.get_element_pointer(kv_index),
                        l2_cache_hint=TMA_KV_L2_CACHE_HINT,
                        cta_group=cl.CTAGroup.CTA_1,
                    )
            kv_index += 1
            if kv_index == KV_STAGES:
                kv_index = 0
                kv_empty_phase ^= 1

            # Q1
            _wait_mbarrier(q_empty.get_element_pointer(1), q_phase)
            if cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    q_full.get_element_pointer(1),
                    input_tile_bytes,
                    scope=cl.MbarrierScope.BLOCK,
                )
                for part in cl.static_iter(range(input_tma_slices)):
                    q_coordinate = (
                        part * input_tma_granularity,
                        head,
                        query_offset + MMA_M,
                        batch,
                    )
                    if variable_length:
                        q_coordinate = (
                            part * input_tma_granularity,
                            head,
                            query_base + query_offset + MMA_M,
                        )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        q_tmap,
                        q_coordinate,
                        q_smem.get_element_pointer(
                            (1, part * MMA_M * input_tma_granularity)
                        ),
                        q_full.get_element_pointer(1),
                    )

            # V0
            _wait_mbarrier(kv_empty.get_element_pointer(kv_index), kv_empty_phase)
            if cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    kv_full.get_element_pointer(kv_index),
                    input_tile_bytes,
                    scope=cl.MbarrierScope.BLOCK,
                )
                for part in cl.static_iter(range(input_tma_slices)):
                    v_coordinate = (
                        part * input_tma_granularity,
                        key_head,
                        first_key_offset,
                        batch,
                    )
                    if variable_length:
                        v_coordinate = (
                            part * input_tma_granularity,
                            key_head,
                            key_base + first_key_offset,
                        )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        v_tmap,
                        v_coordinate,
                        cl.map_shared_to_cluster(
                            kv_smem.get_element_pointer(
                                (
                                    kv_index,
                                    part * MMA_N * input_tma_granularity,
                                )
                            ),
                            0,
                        ),
                        kv_full.get_element_pointer(kv_index),
                        l2_cache_hint=TMA_KV_L2_CACHE_HINT,
                        cta_group=cl.CTAGroup.CTA_1,
                    )
            kv_index += 1
            if kv_index == KV_STAGES:
                kv_index = 0
                kv_empty_phase ^= 1

            load_trip_count = cl.int32(task_trip_count)
            for key_ordinal in range(1, load_trip_count):
                key_offset = (task_trip_start + key_ordinal) * MMA_N
                _wait_mbarrier(kv_empty.get_element_pointer(kv_index), kv_empty_phase)
                if cl.elect_sync():
                    cl.mbarrier_arrive_expect_transaction(
                        kv_full.get_element_pointer(kv_index),
                        input_tile_bytes,
                        scope=cl.MbarrierScope.BLOCK,
                    )
                    for part in cl.static_iter(range(input_tma_slices)):
                        k_coordinate = (
                            part * input_tma_granularity,
                            key_head,
                            key_offset,
                            batch,
                        )
                        if variable_length:
                            k_coordinate = (
                                part * input_tma_granularity,
                                key_head,
                                key_base + key_offset,
                            )
                        cl.copy_async_bulk_tensor_global_to_shared(
                            k_tmap,
                            k_coordinate,
                            cl.map_shared_to_cluster(
                                kv_smem.get_element_pointer(
                                    (
                                        kv_index,
                                        part * MMA_N * input_tma_granularity,
                                    )
                                ),
                                0,
                            ),
                            kv_full.get_element_pointer(kv_index),
                            l2_cache_hint=TMA_KV_L2_CACHE_HINT,
                            cta_group=cl.CTAGroup.CTA_1,
                        )
                kv_index += 1
                if kv_index == KV_STAGES:
                    kv_index = 0
                    kv_empty_phase ^= 1

                _wait_mbarrier(kv_empty.get_element_pointer(kv_index), kv_empty_phase)
                if cl.elect_sync():
                    cl.mbarrier_arrive_expect_transaction(
                        kv_full.get_element_pointer(kv_index),
                        input_tile_bytes,
                        scope=cl.MbarrierScope.BLOCK,
                    )
                    for part in cl.static_iter(range(input_tma_slices)):
                        v_coordinate = (
                            part * input_tma_granularity,
                            key_head,
                            key_offset,
                            batch,
                        )
                        if variable_length:
                            v_coordinate = (
                                part * input_tma_granularity,
                                key_head,
                                key_base + key_offset,
                            )
                        cl.copy_async_bulk_tensor_global_to_shared(
                            v_tmap,
                            v_coordinate,
                            cl.map_shared_to_cluster(
                                kv_smem.get_element_pointer(
                                    (
                                        kv_index,
                                        part * MMA_N * input_tma_granularity,
                                    )
                                ),
                                0,
                            ),
                            kv_full.get_element_pointer(kv_index),
                            l2_cache_hint=TMA_KV_L2_CACHE_HINT,
                            cta_group=cl.CTAGroup.CTA_1,
                        )
                kv_index += 1
                if kv_index == KV_STAGES:
                    kv_index = 0
                    kv_empty_phase ^= 1

            seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                sched_tile,
                sched_valid,
                sched_full.get_base_pointer(),
                sched_empty.get_base_pointer(),
                sched_phase,
            )
            data_task += 1

    # --------------------------------------------------------------- tcgen MMA
    elif warp == MMA_WARP:
        cl.setmaxregister_decrease(32)
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(),
            TMEM_COLUMNS,
            cta_group=cl.CTAGroup.CTA_1,
        )
        cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_1)
        cl.barrier_sync_block(
            number_of_threads=TMEM_ALLOC_THREADS,
            barrier_id=TMEM_ALLOC_BARRIER,
        )
        tmem_base = tmem_storage[0]
        instruction_ab_type = cl.Tcgen05InstructionDescriptor.F16Type.F16
        if input_kind == ELEMENT_BF16:
            instruction_ab_type = cl.Tcgen05InstructionDescriptor.F16Type.BF16
        qk_instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=instruction_ab_type,
            b_type=instruction_ab_type,
            n=MMA_N,
            m=MMA_M,
        ).encode()
        pv_instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=instruction_ab_type,
            b_type=instruction_ab_type,
            transpose_b=True,
            n=head_dim,
            m=MMA_M,
        ).encode()

        seq_tile, head, batch = _decode_work_tile(
            cl.block_index(0),
            scheduler_seq_tiles,
            scheduler_batch_count,
            scheduler_head_count,
            causal_scheduler,
        )
        valid = True
        sched_phase = 0
        data_task = 0
        score_cursor = 0
        pv_cursor = 0
        kv_index = 0
        kv_full_phase = 0
        while valid:
            # CUTLASS reuses one elected MMA-warp lane for all UMMA math in a
            # work tile. Pipeline commits elect independently in their helper.
            mma_elect_one = cl.elect_sync()
            query_base, key_base, task_seq_q, task_seq_k = _sequence_context(
                cumulative_q,
                cumulative_k,
                batch,
                seq_q,
                seq_k,
                variable_length,
            )
            work_valid = True
            if variable_length:
                work_valid = seq_tile * CTA_M < task_seq_q
            if not work_valid:
                seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                    sched_tile,
                    sched_valid,
                    sched_full.get_base_pointer(),
                    sched_empty.get_base_pointer(),
                    sched_phase,
                )
                continue
            _, task_trip_count, q0_trailing_invalid = _key_trip_context(
                seq_tile,
                task_seq_q,
                task_seq_k,
                bottom_right_align,
                window_left,
                window_right,
            )
            q_phase = data_task & 1
            for qid in cl.static_iter(range(2)):
                _wait_mbarrier(q_full.get_element_pointer(qid), q_phase)
            q_desc0 = _qk_descriptor(
                q_smem.get_element_pointer((0, 0)),
                head_dim,
                input_bits,
                input_tma_slices,
                input_swizzle,
            )
            q_desc1 = _qk_descriptor(
                q_smem.get_element_pointer((1, 0)),
                head_dim,
                input_bits,
                input_tma_slices,
                input_swizzle,
            )

            # QK0(0), QK1(0): both halves share K0 before its ring slot is
            # released. The acquire mbarrier wait orders the TMA async writes
            # before the tcgen MMA, matching CUTLASS's TMA-to-UMMA pipeline.
            k_stage = kv_index
            _wait_mbarrier(kv_full.get_element_pointer(k_stage), kv_full_phase)
            k_desc = _qk_descriptor(
                kv_smem.get_element_pointer((k_stage, 0)),
                head_dim,
                input_bits,
                input_tma_slices,
                input_swizzle,
            )
            initial_score_event = score_cursor
            _wait_mbarrier(
                score_empty.get_element_pointer(0),
                1 ^ (initial_score_event & 1),
            )
            _issue_qk(
                mma_kind,
                _tmem_pointer(tmem_base, 0, TMEM_S0),
                q_desc0,
                k_desc,
                qk_instruction,
                score_full.get_element_pointer(0),
                qk_phases,
                k_step,
                input_bits,
                input_tma_slices,
                input_tile_bytes,
                True,
                mma_elect_one,
            )
            _wait_mbarrier(
                score_empty.get_element_pointer(1),
                1 ^ (initial_score_event & 1),
            )
            _issue_qk(
                mma_kind,
                _tmem_pointer(tmem_base, 0, TMEM_S1),
                q_desc1,
                k_desc,
                qk_instruction,
                score_full.get_element_pointer(1),
                qk_phases,
                k_step,
                input_bits,
                input_tma_slices,
                input_tile_bytes,
                True,
                mma_elect_one,
            )
            if cl.elect_sync():
                cl.tcgen05_commit(
                    kv_empty.get_element_pointer(k_stage),
                    cta_group=cl.CTAGroup.CTA_1,
                )
            kv_index += 1
            if kv_index == KV_STAGES:
                kv_index = 0
                kv_full_phase ^= 1

            # PV0(0) consumes P0 but retains V0 for the delayed PV1(0).
            previous_v_stage = kv_index
            _wait_mbarrier(
                kv_full.get_element_pointer(previous_v_stage), kv_full_phase
            )
            previous_v_desc = _pv_descriptor(
                kv_smem.get_element_pointer((previous_v_stage, 0)),
                head_dim,
                input_bits,
                input_tma_slices,
                input_swizzle,
                input_tile_bytes,
            )
            _issue_pv(
                mma_kind,
                _tmem_pointer(tmem_base, 0, TMEM_O0),
                _tmem_pointer(tmem_base, 0, TMEM_P0),
                previous_v_desc,
                pv_instruction,
                p_full.get_element_pointer((0, 0)),
                p_full.get_element_pointer((0, 1)),
                p_empty.get_element_pointer((0, 0)),
                p_empty.get_element_pointer((0, 1)),
                o_full.get_element_pointer(0),
                o_empty.get_element_pointer(0),
                pv_cursor,
                cl.int32(0),
                pv_phases_per_ready,
                k_step,
                head_dim,
                input_bits,
                input_tma_slices,
                True,
                mma_elect_one,
            )
            kv_index += 1
            if kv_index == KV_STAGES:
                kv_index = 0
                kv_full_phase ^= 1

            for key_ordinal in range(1, task_trip_count):
                score_event = score_cursor + key_ordinal
                # QK0(i) starts the next score tile before PV1(i-1).
                k_stage = kv_index
                _wait_mbarrier(
                    kv_full.get_element_pointer(k_stage), kv_full_phase
                )
                k_desc = _qk_descriptor(
                    kv_smem.get_element_pointer((k_stage, 0)),
                    head_dim,
                    input_bits,
                    input_tma_slices,
                    input_swizzle,
                )
                skip_q0_tail = (
                    q0_trailing_invalid
                    and key_ordinal == task_trip_count - 1
                )
                _wait_mbarrier(
                    score_empty.get_element_pointer(0),
                    1 ^ (score_event & 1),
                )
                _issue_qk(
                    mma_kind,
                    _tmem_pointer(tmem_base, 0, TMEM_S0),
                    q_desc0,
                    k_desc,
                    qk_instruction,
                    score_full.get_element_pointer(0),
                    qk_phases,
                    k_step,
                    input_bits,
                    input_tma_slices,
                    input_tile_bytes,
                    not skip_q0_tail,
                    mma_elect_one,
                )

                # PV1(i-1), then release the retained V(i-1) ring slot.
                previous_pv_event = pv_cursor + key_ordinal - 1
                _issue_pv(
                    mma_kind,
                    _tmem_pointer(tmem_base, 0, TMEM_O1),
                    _tmem_pointer(tmem_base, 0, TMEM_P1),
                    previous_v_desc,
                    pv_instruction,
                    p_full.get_element_pointer((1, 0)),
                    p_full.get_element_pointer((1, 1)),
                    p_empty.get_element_pointer((1, 0)),
                    p_empty.get_element_pointer((1, 1)),
                    o_full.get_element_pointer(1),
                    o_empty.get_element_pointer(1),
                    previous_pv_event,
                    key_ordinal - 1,
                    pv_phases_per_ready,
                    k_step,
                    head_dim,
                    input_bits,
                    input_tma_slices,
                    True,
                    mma_elect_one,
                )
                if cl.elect_sync():
                    cl.tcgen05_commit(
                        kv_empty.get_element_pointer(previous_v_stage),
                        cta_group=cl.CTAGroup.CTA_1,
                    )

                # QK1(i) reuses Ki, after P1 has vacated the S1/P1 alias.
                _wait_mbarrier(
                    score_empty.get_element_pointer(1),
                    1 ^ (score_event & 1),
                )
                _issue_qk(
                    mma_kind,
                    _tmem_pointer(tmem_base, 0, TMEM_S1),
                    q_desc1,
                    k_desc,
                    qk_instruction,
                    score_full.get_element_pointer(1),
                    qk_phases,
                    k_step,
                    input_bits,
                    input_tma_slices,
                    input_tile_bytes,
                    True,
                    mma_elect_one,
                )
                if cl.elect_sync():
                    cl.tcgen05_commit(
                        kv_empty.get_element_pointer(k_stage),
                        cta_group=cl.CTAGroup.CTA_1,
                    )
                kv_index += 1
                if kv_index == KV_STAGES:
                    kv_index = 0
                    kv_full_phase ^= 1

                # PV0(i) retains Vi for PV1(i) in the next iteration/final.
                previous_v_stage = kv_index
                _wait_mbarrier(
                    kv_full.get_element_pointer(previous_v_stage), kv_full_phase
                )
                previous_v_desc = _pv_descriptor(
                    kv_smem.get_element_pointer((previous_v_stage, 0)),
                    head_dim,
                    input_bits,
                    input_tma_slices,
                    input_swizzle,
                    input_tile_bytes,
                )
                _issue_pv(
                    mma_kind,
                    _tmem_pointer(tmem_base, 0, TMEM_O0),
                    _tmem_pointer(tmem_base, 0, TMEM_P0),
                    previous_v_desc,
                    pv_instruction,
                    p_full.get_element_pointer((0, 0)),
                    p_full.get_element_pointer((0, 1)),
                    p_empty.get_element_pointer((0, 0)),
                    p_empty.get_element_pointer((0, 1)),
                    o_full.get_element_pointer(0),
                    o_empty.get_element_pointer(0),
                    pv_cursor + key_ordinal,
                    key_ordinal,
                    pv_phases_per_ready,
                    k_step,
                    head_dim,
                    input_bits,
                    input_tma_slices,
                    not skip_q0_tail,
                    mma_elect_one,
                )
                kv_index += 1
                if kv_index == KV_STAGES:
                    kv_index = 0
                    kv_full_phase ^= 1

            # Q0's sentinel is published before final PV1, mirroring the
            # source overlap.  Q buffers are safe to recycle after all QKs.
            final_score_event = score_cursor + task_trip_count
            _wait_mbarrier(
                score_empty.get_element_pointer(0),
                1 ^ (final_score_event & 1),
            )
            if cl.elect_sync():
                cl.tcgen05_commit(
                    q_empty.get_element_pointer(0),
                    cta_group=cl.CTAGroup.CTA_1,
                )
            if cl.elect_sync():
                cl.tcgen05_commit(
                    q_empty.get_element_pointer(1),
                    cta_group=cl.CTAGroup.CTA_1,
                )
            if cl.elect_sync():
                cl.tcgen05_commit(
                    score_full.get_element_pointer(0),
                    cta_group=cl.CTAGroup.CTA_1,
                )

            # PV1(last), release V(last), then publish Q1's final sentinel.
            final_key_ordinal = task_trip_count - 1
            _issue_pv(
                mma_kind,
                _tmem_pointer(tmem_base, 0, TMEM_O1),
                _tmem_pointer(tmem_base, 0, TMEM_P1),
                previous_v_desc,
                pv_instruction,
                p_full.get_element_pointer((1, 0)),
                p_full.get_element_pointer((1, 1)),
                p_empty.get_element_pointer((1, 0)),
                p_empty.get_element_pointer((1, 1)),
                o_full.get_element_pointer(1),
                o_empty.get_element_pointer(1),
                pv_cursor + final_key_ordinal,
                final_key_ordinal,
                pv_phases_per_ready,
                k_step,
                head_dim,
                input_bits,
                input_tma_slices,
                True,
                mma_elect_one,
            )
            if cl.elect_sync():
                cl.tcgen05_commit(
                    kv_empty.get_element_pointer(previous_v_stage),
                    cta_group=cl.CTAGroup.CTA_1,
                )
            _wait_mbarrier(
                score_empty.get_element_pointer(1),
                1 ^ (final_score_event & 1),
            )
            if cl.elect_sync():
                cl.tcgen05_commit(
                    score_full.get_element_pointer(1),
                    cta_group=cl.CTAGroup.CTA_1,
                )
            score_cursor += task_trip_count + 1
            pv_cursor += task_trip_count

            seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                sched_tile,
                sched_valid,
                sched_full.get_base_pointer(),
                sched_empty.get_base_pointer(),
                sched_phase,
            )
            data_task += 1

        _wait_mbarrier(tmem_dealloc.get_base_pointer(), 0)
        cl.tcgen05_deallocate(tmem_base, TMEM_COLUMNS, cta_group=cl.CTAGroup.CTA_1)

    # ------------------------------------------------------------ softmax 0/1
    elif warp < CORRECTION_WARPS[0]:
        if use_sinks:
            # Source ownership: softmax0 stages the sink vector once, and both
            # softmax warp groups rendezvous before persistent work begins.
            if warp < SOFTMAX1_WARPS[0]:
                sink_index = tid
                while sink_index < heads_q:
                    sinks_smem[sink_index] = cl.float16(sinks[sink_index])
                    sink_index += len(SOFTMAX0_WARPS) * WARP_SIZE
            cl.barrier_sync_block(256, SOFTMAX_SINK_BARRIER)
        cl.setmaxregister_increase(192)
        # Allocating all 512 TMEM columns fixes the allocation base at zero.
        # Softmax deliberately constructs that raw base instead of racing the
        # MMA warp's shared allocation-pointer publication.
        tmem_base = cl.bitcast(
            cl.uint32(0),
            cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR),
        )
        qid = warp // 4
        warp_in_group = warp % 4
        row_in_tile = warp_in_group * WARP_SIZE + lane
        score_column = qid * 128
        p_column = TMEM_P0 + qid * 128

        seq_tile, head, batch = _decode_work_tile(
            cl.block_index(0),
            scheduler_seq_tiles,
            scheduler_batch_count,
            scheduler_head_count,
            causal_scheduler,
        )
        valid = True
        sched_phase = 0
        score_cursor = 0
        pv_cursor = 0
        while valid:
            query_base, key_base, task_seq_q, task_seq_k = _sequence_context(
                cumulative_q,
                cumulative_k,
                batch,
                seq_q,
                seq_k,
                variable_length,
            )
            work_valid = True
            if variable_length:
                work_valid = seq_tile * CTA_M < task_seq_q
            if not work_valid:
                seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                    sched_tile,
                    sched_valid,
                    sched_full.get_base_pointer(),
                    sched_empty.get_base_pointer(),
                    sched_phase,
                )
                continue
            task_trip_start, task_trip_count, q0_trailing_invalid = (
                _key_trip_context(
                    seq_tile,
                    task_seq_q,
                    task_seq_k,
                    bottom_right_align,
                    window_left,
                    window_right,
                )
            )
            row_max = cl.float32(-float("inf"))
            row_sum = cl.float32(0.0)
            if use_sinks:
                row_max = cl.float32(sinks_smem[head]) / scale_softmax
                row_sum = cl.float32(1.0)

            for key_ordinal in range(task_trip_count):
                score_event = score_cursor
                pv_event = pv_cursor
                _wait_mbarrier(
                    score_full.get_element_pointer(qid), score_event & 1
                )
                row_tmem = _tmem_pointer(
                    tmem_base,
                    warp_in_group * WARP_SIZE,
                    score_column,
                )
                token_only_q0 = (
                    qid == 0
                    and q0_trailing_invalid
                    and key_ordinal > 0
                    and key_ordinal == task_trip_count - 1
                )
                if token_only_q0:
                    # Q0's fully-invalid super-tile tail exchanges all pipeline
                    # tokens but leaves row statistics, P, and O unchanged.
                    _wait_mbarrier(
                        stats_empty.get_element_pointer(qid),
                        1 ^ (score_event & 1),
                    )
                    cl.mbarrier_arrive(
                        stats_full.get_element_pointer(qid),
                        scope=cl.MbarrierScope.BLOCK,
                    )
                    for half in cl.static_iter(range(P_READY_STAGES)):
                        _wait_mbarrier(
                            p_empty.get_element_pointer((qid, half)),
                            1 ^ (pv_event & 1),
                        )
                        cl.mbarrier_arrive(
                            p_full.get_element_pointer((qid, half)),
                            scope=cl.MbarrierScope.BLOCK,
                        )
                        if half == 0:
                            if qid == 1:
                                cl.barrier_arrive_block(256, SOFTMAX_SEQUENCE_0)
                                cl.barrier_sync_block(256, SOFTMAX_SEQUENCE_1)
                    cl.mbarrier_arrive(
                        score_empty.get_element_pointer(qid),
                        scope=cl.MbarrierScope.BLOCK,
                    )
                else:
                    raw_score_chunks = tuple(
                        cl.tcgen05_load(
                            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                            _tmem_pointer(row_tmem, 0, chunk * 32),
                            element_count=32,
                            dtype=cl.float32,
                        )
                        for chunk in cl.static_iter(range(4))
                    )
                    score_chunks = raw_score_chunks
                    # Exact source fast path: aligned fixed dense tiles have
                    # no element mask at all.  General tails and varlen keep
                    # the guarded path.
                    if not (
                        not variable_length
                        and window_left < 0
                        and window_right < 0
                        and seq_q % CTA_M == 0
                        and seq_k % MMA_N == 0
                    ):
                        query_index = (
                            seq_tile * CTA_M + qid * MMA_M + row_in_tile
                        )
                        global_key_tile = task_trip_start + key_ordinal
                        score_chunks = tuple(
                            cl.Vector(
                                *tuple(
                                    _mask_score(
                                        raw_score_chunks[chunk][item],
                                        query_index,
                                        global_key_tile * MMA_N
                                        + chunk * 32
                                        + item,
                                        task_seq_q,
                                        task_seq_k,
                                        bottom_right_align,
                                        window_left,
                                        window_right,
                                    )
                                    for item in cl.static_iter(range(32))
                                ),
                                dtype=cl.float32,
                            )
                            for chunk in cl.static_iter(range(4))
                        )

                    # Match CUTLASS's four 32-value vector reductions:
                    # MAXIMUMF propagates NaNs within each chunk, while the
                    # outer maxnum combines retain the source's semantics.
                    new_row_max = cl.float32(-float("inf"))
                    for chunk in cl.static_iter(score_chunks):
                        chunk_max = chunk[:32].reduce(
                            cl.VectorReduction.max,
                            propagate_nan=True,
                        )
                        new_row_max = cl.maximum(new_row_max, chunk_max)
                    old_max = row_max
                    row_max = cl.maximum(new_row_max, row_max)
                    safe_max = row_max
                    if safe_max == cl.float32(-float("inf")):
                        safe_max = cl.float32(0.0)
                    alpha = cl.exp2(
                        (old_max - safe_max) * scale_softmax_log2, flush_to_zero=True
                    )

                    # Publish correction alpha before P, exactly as the source
                    # stats->correction pipeline does.
                    _wait_mbarrier(
                        stats_empty.get_element_pointer(qid),
                        1 ^ (score_event & 1),
                    )
                    stats_vec = cl.Vector(
                        alpha,
                        safe_max,
                        dtype=cl.float32,
                    )
                    cl.tcgen05_store(
                        cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                        row_tmem,
                        stats_vec,
                    )
                    # Acquire the first P-ready stage while the statistics
                    # store is in flight.  The second acquire and STTM wait
                    # are split into the initial FMA lookahead below, matching
                    # CUTLASS's machine schedule.
                    _wait_mbarrier(
                        p_empty.get_element_pointer((qid, 0)),
                        1 ^ (pv_event & 1),
                    )

                    block_sum0 = cl.float32(0.0)
                    block_sum1 = cl.float32(0.0)
                    minus_safe_max_scale = -safe_max * scale_softmax_log2
                    # Keep the DKG constexpr-loop topology while using flattened
                    # local arrays in place of its mutable Python lists.
                    if input_kind == ELEMENT_E4M3:
                        p_window_elems = 4
                        fma_pipe_windows = 3
                        convert_pipe_windows = 2
                    else:
                        p_window_elems = 2
                        fma_pipe_windows = 6
                        convert_pipe_windows = 4
                    p_windows_per_store = 64 // p_window_elems
                    p_windows_per_s_chunk = 32 // p_window_elems
                    total_p_windows = 128 // p_window_elems
                    p_store_words = p_windows_per_store
                    p_window_alignment = p_window_elems * 4
                    row_sum_reduce_lag_windows = 8

                    with (
                        cl.local_array(
                            total_p_windows * p_window_elems,
                            cl.float32,
                            alignment=16,
                        ) as p_windows,
                        cl.local_array(
                            total_p_windows * p_window_elems,
                            cl.float32,
                            alignment=16,
                        ) as p_fma_windows,
                        cl.local_array(
                            p_store_words,
                            cl.int32,
                            alignment=16,
                        ) as p_data_packed,
                    ):
                        p_windows_ptr = p_windows.get_base_pointer()
                        p_fma_windows_ptr = (
                            p_fma_windows.get_base_pointer()
                        )
                        p_data_packed_ptr = (
                            p_data_packed.get_base_pointer()
                        )

                        # Preserve the initial CUDA Lang lookahead:
                        # three FMA windows, the second P acquire and
                        # STTM wait, then the remaining FP16 windows.
                        for window_idx in cl.static_iter(range(3)):
                            chunk_idx = (
                                window_idx
                                // p_windows_per_s_chunk
                            )
                            window_start = (
                                window_idx
                                % p_windows_per_s_chunk
                            ) * p_window_elems
                            fma_window = (
                                _materialize_p_fma_window(
                                    _compute_p_fma_window(
                                        score_chunks[chunk_idx],
                                        window_start,
                                        p_window_elems,
                                        scale_softmax_log2,
                                        minus_safe_max_scale,
                                    ),
                                    p_window_elems,
                                )
                            )
                            (
                                p_fma_windows_ptr
                                + window_idx * p_window_elems
                            ).store(
                                fma_window,
                                alignment=p_window_alignment,
                            )

                        _wait_mbarrier(
                            p_empty.get_element_pointer((qid, 1)),
                            1 ^ (pv_event & 1),
                        )
                        cl.tcgen05_wait_store()

                        for window_idx in cl.static_iter(
                            range(3, fma_pipe_windows)
                        ):
                            chunk_idx = (
                                window_idx
                                // p_windows_per_s_chunk
                            )
                            window_start = (
                                window_idx
                                % p_windows_per_s_chunk
                            ) * p_window_elems
                            fma_window = (
                                _materialize_p_fma_window(
                                    _compute_p_fma_window(
                                        score_chunks[chunk_idx],
                                        window_start,
                                        p_window_elems,
                                        scale_softmax_log2,
                                        minus_safe_max_scale,
                                    ),
                                    p_window_elems,
                                )
                            )
                            (
                                p_fma_windows_ptr
                                + window_idx * p_window_elems
                            ).store(
                                fma_window,
                                alignment=p_window_alignment,
                            )

                        cl.mbarrier_arrive(
                            stats_full.get_element_pointer(qid),
                            scope=cl.MbarrierScope.BLOCK,
                        )

                        for window_idx in cl.static_iter(
                            range(total_p_windows)
                        ):
                            store_global_window_idx = (
                                window_idx
                                - convert_pipe_windows
                            )
                            if store_global_window_idx >= 0:
                                store_window_idx = (
                                    store_global_window_idx
                                    % p_windows_per_store
                                )
                                if (
                                    store_window_idx
                                    < p_windows_per_store
                                    - convert_pipe_windows
                                ):
                                    probability = (
                                        p_windows_ptr
                                        + store_global_window_idx
                                        * p_window_elems
                                    ).load(
                                        count=p_window_elems,
                                        alignment=p_window_alignment,
                                    )
                                    p_data_packed[
                                        store_window_idx
                                    ] = _pack_p_window(
                                        probability,
                                        input_kind,
                                    )

                                    if (
                                        store_global_window_idx
                                        == p_windows_per_store
                                        // 2
                                        - 1
                                    ):
                                        packed_values = (
                                            p_data_packed_ptr.load(
                                                count=(
                                                    p_store_words
                                                    // 2
                                                ),
                                                alignment=16,
                                            )
                                        )
                                        p_tmem = _tmem_pointer(
                                            tmem_base,
                                            warp_in_group
                                            * WARP_SIZE,
                                            p_column,
                                        )
                                        cl.tcgen05_store(
                                            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                                            p_tmem,
                                            packed_values,
                                        )

                            fma_window = (
                                p_fma_windows_ptr
                                + window_idx * p_window_elems
                            ).load(
                                count=p_window_elems,
                                alignment=p_window_alignment,
                            )
                            probability = _exp_p_window(
                                fma_window,
                                p_window_elems,
                            )
                            (
                                p_windows_ptr
                                + window_idx * p_window_elems
                            ).store(
                                probability,
                                alignment=p_window_alignment,
                            )

                            future_window_idx = (
                                window_idx + fma_pipe_windows
                            )
                            if future_window_idx < total_p_windows:
                                chunk_idx = (
                                    future_window_idx
                                    // p_windows_per_s_chunk
                                )
                                window_start = (
                                    future_window_idx
                                    % p_windows_per_s_chunk
                                ) * p_window_elems
                                fma_window = (
                                    _compute_p_fma_window(
                                        score_chunks[chunk_idx],
                                        window_start,
                                        p_window_elems,
                                        scale_softmax_log2,
                                        minus_safe_max_scale,
                                    )
                                )
                                if (
                                    future_window_idx
                                    // p_windows_per_store
                                    != window_idx
                                    // p_windows_per_store
                                ):
                                    fma_window = (
                                        _materialize_p_fma_window(
                                            fma_window,
                                            p_window_elems,
                                        )
                                    )
                                (
                                    p_fma_windows_ptr
                                    + future_window_idx
                                    * p_window_elems
                                ).store(
                                    fma_window,
                                    alignment=p_window_alignment,
                                )

                            reduce_window_idx = (
                                window_idx
                                - row_sum_reduce_lag_windows
                            )
                            if reduce_window_idx >= 0:
                                reduce_window = (
                                    p_windows_ptr
                                    + reduce_window_idx
                                    * p_window_elems
                                ).load(
                                    count=p_window_elems,
                                    alignment=p_window_alignment,
                                )
                                (
                                    block_sum0,
                                    block_sum1,
                                ) = _accumulate_p_window_sum(
                                    block_sum0,
                                    block_sum1,
                                    reduce_window,
                                    p_window_elems,
                                )

                            if (
                                (window_idx + 1)
                                % p_windows_per_store
                                == 0
                            ):
                                pair_idx = (
                                    window_idx
                                    // p_windows_per_store
                                )
                                for tail_idx in cl.static_iter(
                                    range(
                                        convert_pipe_windows
                                    )
                                ):
                                    store_global_window_idx = (
                                        window_idx
                                        + 1
                                        - convert_pipe_windows
                                        + tail_idx
                                    )
                                    store_window_idx = (
                                        store_global_window_idx
                                        % p_windows_per_store
                                    )
                                    probability = (
                                        p_windows_ptr
                                        + store_global_window_idx
                                        * p_window_elems
                                    ).load(
                                        count=p_window_elems,
                                        alignment=p_window_alignment,
                                    )
                                    p_data_packed[
                                        store_window_idx
                                    ] = _pack_p_window(
                                        probability,
                                        input_kind,
                                    )

                                if pair_idx == 0:
                                    packed_values = (
                                        p_data_packed_ptr
                                        + p_store_words // 2
                                    ).load(
                                        count=p_store_words
                                        // 2,
                                        alignment=16,
                                    )
                                    p_tmem_offset = (
                                        p_store_words // 2
                                    )
                                else:
                                    packed_values = (
                                        p_data_packed_ptr.load(
                                            count=p_store_words,
                                            alignment=16,
                                        )
                                    )
                                    p_tmem_offset = (
                                        pair_idx
                                        * p_store_words
                                    )
                                p_tmem = _tmem_pointer(
                                    tmem_base,
                                    warp_in_group * WARP_SIZE,
                                    p_column
                                    + p_tmem_offset,
                                )
                                cl.tcgen05_store(
                                    cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                                    p_tmem,
                                    packed_values,
                                )
                                cl.tcgen05_wait_store()
                                cl.mbarrier_arrive(
                                    p_full.get_element_pointer(
                                        (qid, pair_idx)
                                    ),
                                    scope=cl.MbarrierScope.BLOCK,
                                )
                                if pair_idx == 0:
                                    if qid == 1:
                                        cl.barrier_arrive_block(
                                            256,
                                            SOFTMAX_SEQUENCE_0,
                                        )
                                        cl.barrier_sync_block(
                                            256,
                                            SOFTMAX_SEQUENCE_1,
                                        )

                        cl.mbarrier_arrive(
                            score_empty.get_element_pointer(qid),
                            scope=cl.MbarrierScope.BLOCK,
                        )

                        for tail_idx in cl.static_iter(
                            range(
                                row_sum_reduce_lag_windows
                            )
                        ):
                            reduce_window_idx = (
                                total_p_windows
                                - row_sum_reduce_lag_windows
                                + tail_idx
                            )
                            reduce_window = (
                                p_windows_ptr
                                + reduce_window_idx
                                * p_window_elems
                            ).load(
                                count=p_window_elems,
                                alignment=p_window_alignment,
                            )
                            (
                                block_sum0,
                                block_sum1,
                            ) = _accumulate_p_window_sum(
                                block_sum0,
                                block_sum1,
                                reduce_window,
                                p_window_elems,
                            )
                    half_scaled_old_sum = row_sum * alpha * cl.float32(0.5)
                    block_sum0, block_sum1 = _add_packed_f32x2(
                        half_scaled_old_sum,
                        half_scaled_old_sum,
                        block_sum0,
                        block_sum1,
                    )
                    row_sum = block_sum0 + block_sum1

                if qid == 0:
                    cl.barrier_arrive_block(256, SOFTMAX_SEQUENCE_1)
                    cl.barrier_sync_block(256, SOFTMAX_SEQUENCE_0)
                    cl.barrier_sync_block(128, SOFTMAX_WG_0)
                else:
                    cl.barrier_sync_block(128, SOFTMAX_WG_1)
                score_cursor += 1
                pv_cursor += 1

            final_score_event = score_cursor
            _wait_mbarrier(
                score_full.get_element_pointer(qid), final_score_event & 1
            )
            _wait_mbarrier(
                stats_empty.get_element_pointer(qid),
                1 ^ (final_score_event & 1),
            )
            final_vec = cl.Vector(
                row_sum,
                row_max,
                dtype=cl.float32,
            )
            final_tmem = _tmem_pointer(
                tmem_base,
                warp_in_group * WARP_SIZE,
                score_column,
            )
            cl.tcgen05_store(
                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                final_tmem,
                final_vec,
            )
            cl.tcgen05_wait_store()
            cl.mbarrier_arrive(
                stats_full.get_element_pointer(qid),
                scope=cl.MbarrierScope.BLOCK,
            )
            # S and the final statistics alias in TMEM. Do not return the
            # score slot to MMA until correction has consumed (sum,max).
            _wait_mbarrier(
                stats_empty.get_element_pointer(qid), final_score_event & 1
            )
            cl.mbarrier_arrive(
                score_empty.get_element_pointer(qid),
                scope=cl.MbarrierScope.BLOCK,
            )
            score_cursor += 1

            seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                sched_tile,
                sched_valid,
                sched_full.get_base_pointer(),
                sched_empty.get_base_pointer(),
                sched_phase,
            )
        cl.mbarrier_arrive(
            tmem_dealloc.get_base_pointer(), scope=cl.MbarrierScope.BLOCK
        )

    # ------------------------------------------------------------- correction
    elif warp >= CORRECTION_WARPS[0] and warp <= CORRECTION_WARPS[-1]:
        cl.setmaxregister_decrease(96)
        cl.barrier_sync_block(
            number_of_threads=TMEM_ALLOC_THREADS,
            barrier_id=TMEM_ALLOC_BARRIER,
        )
        tmem_base = tmem_storage[0]
        warp_in_group = warp - CORRECTION_WARPS[0]
        row_in_tile = warp_in_group * WARP_SIZE + lane

        seq_tile, head, batch = _decode_work_tile(
            cl.block_index(0),
            scheduler_seq_tiles,
            scheduler_batch_count,
            scheduler_head_count,
            causal_scheduler,
        )
        valid = True
        sched_phase = 0
        data_task = 0
        score_cursor = 0
        pv_cursor = 0
        while valid:
            query_base, key_base, task_seq_q, task_seq_k = _sequence_context(
                cumulative_q,
                cumulative_k,
                batch,
                seq_q,
                seq_k,
                variable_length,
            )
            work_valid = True
            if variable_length:
                work_valid = seq_tile * CTA_M < task_seq_q
            if not work_valid:
                seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                    sched_tile,
                    sched_valid,
                    sched_full.get_base_pointer(),
                    sched_empty.get_base_pointer(),
                    sched_phase,
                )
                continue
            _, task_trip_count, q0_trailing_invalid = _key_trip_context(
                seq_tile,
                task_seq_q,
                task_seq_k,
                bottom_right_align,
                window_left,
                window_right,
            )
            # First alpha has no preceding O accumulator to correct.
            first_stats_event = score_cursor
            for qid in cl.static_iter(range(2)):
                _wait_mbarrier(
                    stats_full.get_element_pointer(qid), first_stats_event & 1
                )
                cl.mbarrier_arrive(
                    stats_empty.get_element_pointer(qid),
                    scope=cl.MbarrierScope.BLOCK,
                )

            for key_ordinal in range(1, task_trip_count):
                stats_event = score_cursor + key_ordinal
                previous_o_event = pv_cursor + key_ordinal - 1
                for qid in cl.static_iter(range(2)):
                    skip_q0_rescale = False
                    if qid == 0:
                        skip_q0_rescale = (
                            q0_trailing_invalid
                            and key_ordinal == task_trip_count - 1
                        )
                    _wait_mbarrier(
                        stats_full.get_element_pointer(qid), stats_event & 1
                    )
                    _wait_mbarrier(
                        o_full.get_element_pointer(qid),
                        previous_o_event & 1,
                    )
                    if not skip_q0_rescale:
                        stats_tmem = _tmem_pointer(
                            tmem_base,
                            warp_in_group * WARP_SIZE,
                            qid * 128,
                        )
                        stats = cl.tcgen05_load(
                            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                            stats_tmem,
                            element_count=2,
                            dtype=cl.float32,
                        )
                        alpha = stats[0]
                        should_rescale = True
                        if enable_skip_correction:
                            should_rescale = cl.vote_any_sync(
                                alpha != cl.float32(1.0), mask=cl.int32(-1)
                            )
                        if should_rescale:
                            for column in cl.static_iter(range(0, head_dim, 8)):
                                output_tmem = _tmem_pointer(
                                    tmem_base,
                                    warp_in_group * WARP_SIZE,
                                    TMEM_O0 + qid * 128 + column,
                                )
                                values = cl.tcgen05_load(
                                    cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                                    output_tmem,
                                    element_count=8,
                                    dtype=cl.float32,
                                )
                                scaled = values * alpha
                                cl.tcgen05_store(
                                    cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                                    output_tmem,
                                    scaled,
                                )
                            cl.tcgen05_wait_store()
                    cl.mbarrier_arrive(
                        o_empty.get_element_pointer(qid),
                        scope=cl.MbarrierScope.BLOCK,
                    )
                    cl.mbarrier_arrive(
                        stats_empty.get_element_pointer(qid),
                        scope=cl.MbarrierScope.BLOCK,
                    )

            # Final (sum,max) and final O partial.
            final_stats_event = score_cursor + task_trip_count
            final_o_event = pv_cursor + task_trip_count - 1
            for qid in cl.static_iter(range(2)):
                _wait_mbarrier(
                    stats_full.get_element_pointer(qid), final_stats_event & 1
                )
                _wait_mbarrier(
                    o_full.get_element_pointer(qid),
                    final_o_event & 1,
                )
                _wait_mbarrier(
                    epi_empty.get_element_pointer(qid),
                    1 ^ (data_task & 1),
                )
                stats_tmem = _tmem_pointer(
                    tmem_base,
                    warp_in_group * WARP_SIZE,
                    qid * 128,
                )
                stats = cl.tcgen05_load(
                    cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                    stats_tmem,
                    element_count=2,
                    dtype=cl.float32,
                )
                cl.tcgen05_wait_store()
                row_sum = stats[0]
                row_max = stats[1]
                # The final statistics are now resident in registers; release
                # their TMEM alias before doing the comparatively long output
                # conversion, matching the source pipeline cadence.
                cl.mbarrier_arrive(
                    stats_empty.get_element_pointer(qid),
                    scope=cl.MbarrierScope.BLOCK,
                )
                inv_sum = cl.float32(1.0) / row_sum
                if enable_approx_epilogue_rcp:
                    inv_sum = cl._nvvm.rcp_approx_ftz_f(row_sum)
                final_scale = scale_output * inv_sum
                o_stage = o_smem.get_element_pointer((qid, 0))
                half_query_offset = seq_tile * CTA_M + qid * MMA_M
                query_row = half_query_offset + row_in_tile
                full_output_half = half_query_offset + MMA_M <= task_seq_q
                for column in cl.static_iter(range(0, head_dim, 32)):
                    output_tmem = _tmem_pointer(
                        tmem_base,
                        warp_in_group * WARP_SIZE,
                        TMEM_O0 + qid * 128 + column,
                    )
                    values = cl.tcgen05_load(
                        cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                        output_tmem,
                        element_count=32,
                        dtype=cl.float32,
                    )
                    packed_output = _pack_output_values(
                        values, final_scale, output_kind
                    )
                    if full_output_half:
                        _store_output_values(
                            o_stage,
                            packed_output,
                            row_in_tile,
                            column,
                            head_dim,
                            output_kind,
                            output_bits,
                            output_tma_slices,
                        )
                    elif query_row < task_seq_q:
                        _store_output_global(
                            o,
                            packed_output,
                            query_row,
                            query_base,
                            head,
                            batch,
                            column,
                            output_kind,
                            variable_length,
                        )

                if calculate_lse and query_row < task_seq_q:
                    lse_index = (batch * heads_q + head) * seq_q + query_row
                    if variable_length:
                        lse_index = head * total_q + query_base + query_row
                    lse[lse_index] = (
                        cl.log(row_sum) + scale_softmax * row_max
                    )
                cl.fence_proxy(
                    cl.FenceProxyKind.ASYNC_SHARED,
                    space=cl.MemorySpace.SHARED,
                )
                cl.mbarrier_arrive(
                    epi_full.get_element_pointer(qid),
                    scope=cl.MbarrierScope.BLOCK,
                )
                cl.mbarrier_arrive(
                    o_empty.get_element_pointer(qid),
                    scope=cl.MbarrierScope.BLOCK,
                )
            score_cursor += task_trip_count + 1
            pv_cursor += task_trip_count
            seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                sched_tile,
                sched_valid,
                sched_full.get_base_pointer(),
                sched_empty.get_base_pointer(),
                sched_phase,
            )
            data_task += 1

        cl.mbarrier_arrive(
            tmem_dealloc.get_base_pointer(), scope=cl.MbarrierScope.BLOCK
        )

    # --------------------------------------------------------------- epilogue
    elif warp == EPILOGUE_WARP:
        cl.setmaxregister_decrease(32)
        seq_tile, head, batch = _decode_work_tile(
            cl.block_index(0),
            scheduler_seq_tiles,
            scheduler_batch_count,
            scheduler_head_count,
            causal_scheduler,
        )
        valid = True
        sched_phase = 0
        data_task = 0
        while valid:
            query_base, key_base, task_seq_q, task_seq_k = _sequence_context(
                cumulative_q,
                cumulative_k,
                batch,
                seq_q,
                seq_k,
                variable_length,
            )
            work_valid = True
            if variable_length:
                work_valid = seq_tile * CTA_M < task_seq_q
            if not work_valid:
                seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                    sched_tile,
                    sched_valid,
                    sched_full.get_base_pointer(),
                    sched_empty.get_base_pointer(),
                    sched_phase,
                )
                continue
            query_offset = seq_tile * CTA_M
            store_elect_one = cl.elect_sync()
            for qid in cl.static_iter(range(2)):
                _wait_mbarrier(epi_full.get_element_pointer(qid), data_task & 1)
                if store_elect_one:
                    full_output_half = (
                        query_offset + (qid + 1) * MMA_M <= task_seq_q
                    )
                    if full_output_half:
                        for part in cl.static_iter(range(output_tma_slices)):
                            output_coordinate = (
                                part * output_tma_granularity,
                                head,
                                query_offset + qid * MMA_M,
                                batch,
                            )
                            if variable_length:
                                output_coordinate = (
                                    part * output_tma_granularity,
                                    head,
                                    query_base + query_offset + qid * MMA_M,
                                )
                            cl.copy_async_bulk_tensor_shared_to_global(
                                o_smem.get_element_pointer(
                                    (
                                        qid,
                                        part * MMA_M * output_tma_granularity,
                                    )
                                ),
                                o_tmap,
                                output_coordinate,
                            )
                        cl.copy_async_bulk_commit_group()
                        cl.copy_async_bulk_wait_group(0, read=True)
                cl.mbarrier_arrive(
                    epi_empty.get_element_pointer(qid),
                    scope=cl.MbarrierScope.BLOCK,
                )
            seq_tile, head, batch, valid, sched_phase = _consume_scheduled_tile(
                sched_tile,
                sched_valid,
                sched_full.get_base_pointer(),
                sched_empty.get_base_pointer(),
                sched_phase,
            )
            data_task += 1


# ---------------------------------------------------------------------------
# Source-compatible host configuration, reference, tests, and CLI.
# ---------------------------------------------------------------------------

_DTYPE_NAMES = {
    "Float16": torch.float16,
    "BFloat16": torch.bfloat16,
    "Float8E4M3FN": torch.float8_e4m3fn,
}


@dataclass(frozen=True)
class FmhaConfig:
    q_shape: ShapeSpec = DEFAULT_Q_SHAPE
    k_shape: ShapeSpec = DEFAULT_K_SHAPE
    in_dtype: str = "Float16"
    out_dtype: str = "Float16"
    qk_acc_dtype: str = "Float32"
    pv_acc_dtype: str = "Float32"
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER
    is_causal: bool = False
    bottom_right_align: bool = False
    lse_calculation: bool = False
    window_size: tuple[int, int] = (-1, -1)
    scale_q: float = 1.0
    scale_k: float = 1.0
    scale_v: float = 1.0
    inv_scale_o: float = 1.0
    scale_softmax: float = 0.0
    use_sinks: bool = False
    enable_skip_correction: bool = True
    enable_approx_epilogue_rcp: bool = True

    @property
    def variable_length(self) -> bool:
        return isinstance(self.q_shape[1], tuple) or isinstance(self.k_shape[1], tuple)

    @property
    def attention_scale(self) -> float:
        base = self.scale_softmax
        if base == 0.0:
            base = 1.0 / math.sqrt(self.head_dim)
        return self.scale_q * self.scale_k * base

    @property
    def attention_scale_log2(self) -> float:
        return self.attention_scale * math.log2(math.e)

    @property
    def output_scale(self) -> float:
        return self.scale_v * self.inv_scale_o

    @property
    def head_dim(self) -> int:
        return int(self.q_shape[3])

    @property
    def batch(self) -> int:
        return int(self.q_shape[0])

    @property
    def heads_q(self) -> int:
        return int(self.q_shape[2])

    @property
    def heads_k(self) -> int:
        return int(self.k_shape[2])

    @property
    def effective_window(self) -> tuple[Optional[int], Optional[int]]:
        left, right = self.window_size
        left_value = None if left == -1 else left
        right_value = None if right == -1 else right
        if self.is_causal:
            right_value = 0
        return left_value, right_value

    def fixed_lengths(self) -> tuple[int, int]:
        if self.variable_length:
            raise UnsupportedFmhaConfiguration(
                "fixed_lengths() is only valid for fixed-length configurations"
            )
        return int(self.q_shape[1]), int(self.k_shape[1])

    def validate(self) -> None:
        if len(self.q_shape) != 4 or len(self.k_shape) != 4:
            raise ValueError("Q and K shapes must have four fields: B,S,H,D")
        bq, sq, hq, dq = self.q_shape
        bk, sk, hk, dk = self.k_shape
        if min(int(bq), int(hq), int(dq), int(bk), int(hk), int(dk)) <= 0:
            raise ValueError("all non-sequence shape dimensions must be positive")
        if bq != bk:
            raise ValueError("Q and K batch dimensions must match")
        if dq != dk:
            raise ValueError("Q and K head dimensions must match")
        if dq not in (32, 64, 128):
            raise ValueError("head dimension must be 32, 64, or 128")
        if hq % hk != 0:
            raise ValueError("Hq must be divisible by Hk")
        if self.in_dtype not in _DTYPE_NAMES:
            raise ValueError("in_dtype must be Float16, BFloat16, or Float8E4M3FN")
        if self.out_dtype not in _DTYPE_NAMES:
            raise ValueError("out_dtype must be Float16, BFloat16, or Float8E4M3FN")
        if self.qk_acc_dtype != "Float32" or self.pv_acc_dtype != "Float32":
            raise ValueError("QK and PV accumulator dtypes must both be Float32")
        if self.mma_tiler_mn != DEFAULT_MMA_TILER:
            raise ValueError(
                "mma_tiler_mn must be 128,128; the source TMEM map is fixed to it"
            )
        if len(self.window_size) != 2 or min(self.window_size) < -1:
            raise ValueError("window_size must contain two integers >= -1")
        if isinstance(sq, tuple) != isinstance(sk, tuple):
            raise ValueError("Q and K must both be fixed length or both varlen")
        if isinstance(sq, tuple):
            if len(sq) != bq or len(sk) != bk:
                raise ValueError("varlen tuples must contain one length per batch")
            if min(*sq, *sk) <= 0:
                raise ValueError("all variable sequence lengths must be positive")
        else:
            if int(sq) <= 0 or int(sk) <= 0:
                raise ValueError("sequence lengths must be positive")
            self._validate_nonempty_windows(int(sq), int(sk))
        if self.use_sinks and self.attention_scale == 0.0:
            raise ValueError("sink logits require a nonzero attention scale")

    def _validate_nonempty_windows(self, seq_q: int, seq_k: int) -> None:
        left, right = self.effective_window
        if left is None and right is None:
            return
        offset = seq_k - seq_q if self.bottom_right_align else 0
        for query in range(seq_q):
            begin = 0 if left is None else max(0, query + offset - left)
            end = seq_k if right is None else min(seq_k, query + offset + right + 1)
            # Match the source's diagnostic: interior empty rows are rejected.
            if begin >= end and query not in (0, seq_q - 1):
                raise ValueError(
                    f"sliding window removes every key for query row {query}"
                )

    def phase1_unsupported_reasons(self) -> tuple[str, ...]:
        # The source's two floor-divided QK loops are both empty for E4M3/D32.
        # CUDA Lang expresses the intended operation directly as its one valid
        # K=32 tcgen05 phase, so this combination is numerically implemented
        # instead of reproducing the source driver's zero-MMA defect.
        return ()

    def require_phase1_kernel(self) -> None:
        # The central host contract validates both fixed and jagged lengths,
        # including each batch's bottom-right/window mask.
        fmha_utils.make_tensor_plan(self)
        reasons = self.phase1_unsupported_reasons()
        if reasons:
            raise UnsupportedFmhaConfiguration(
                "native phase-1 kernel does not yet implement: "
                + "; ".join(reasons)
                + ". No fallback was used."
            )


@dataclass(frozen=True)
class PrefillCorrectnessTestCase:
    name: str
    q_shape: ShapeSpec
    k_shape: ShapeSpec
    in_dtype: str = "Float16"
    out_dtype: str = "Float16"
    is_causal: bool = False
    bottom_right_align: bool = False
    window_size: tuple[int, int] = (-1, -1)

    def config(self) -> FmhaConfig:
        return FmhaConfig(
            q_shape=self.q_shape,
            k_shape=self.k_shape,
            in_dtype=self.in_dtype,
            out_dtype=self.out_dtype,
            is_causal=self.is_causal,
            bottom_right_align=self.bottom_right_align,
            window_size=self.window_size,
        )


_PREFILL_CORRECTNESS_TESTS = (
    PrefillCorrectnessTestCase("shape=2,512,8,128", (2, 512, 8, 128), (2, 512, 8, 128)),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64", (2, 512, 64, 64), (2, 512, 8, 64)
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,32", (2, 512, 64, 32), (2, 512, 8, 32)
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,128 causal",
        (2, 512, 64, 128),
        (2, 512, 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,259,64/8,128 non-power-of-2 seqlen causal",
        (2, 259, 64, 128),
        (2, 259, 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,5000,64/8,128 non-power-of-2 long seqlen causal",
        (2, 5000, 64, 128),
        (2, 5000, 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,(512,256),64/8,128 var seqlen",
        (2, (512, 256), 64, 128),
        (2, (512, 256), 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,(511,768),64/8,128 non-power-of-two var seqlen",
        (2, (511, 768), 64, 128),
        (2, (511, 768), 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,128 fp8 input fp16 output",
        (2, 512, 64, 128),
        (2, 512, 8, 128),
        in_dtype="Float8E4M3FN",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,128 fp8 input fp8 output",
        (2, 512, 64, 128),
        (2, 512, 8, 128),
        in_dtype="Float8E4M3FN",
        out_dtype="Float8E4M3FN",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64 fp8 input fp16 output",
        (2, 512, 64, 64),
        (2, 512, 8, 64),
        in_dtype="Float8E4M3FN",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64 fp8 input fp8 output",
        (2, 512, 64, 64),
        (2, 512, 8, 64),
        in_dtype="Float8E4M3FN",
        out_dtype="Float8E4M3FN",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64 bf16 input bf16 output",
        (2, 512, 64, 64),
        (2, 512, 8, 64),
        in_dtype="BFloat16",
        out_dtype="BFloat16",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,1/97,64/8,64 decode phase",
        (2, 1, 64, 64),
        (2, 97, 8, 64),
        is_causal=True,
        bottom_right_align=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64 left-window causal",
        (2, 512, 64, 64),
        (2, 512, 8, 64),
        is_causal=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,511,64/8,64 seqlen=511 left-window causal",
        (2, 511, 64, 64),
        (2, 511, 8, 64),
        is_causal=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,513,64/8,64 seqlen=513 left-window causal",
        (2, 513, 64, 64),
        (2, 513, 8, 64),
        is_causal=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,1/97,64/8,64 left-window decode phase",
        (2, 1, 64, 64),
        (2, 97, 8, 64),
        is_causal=True,
        bottom_right_align=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,1/251,64/8,64 seqlen=511 left-window decode phase",
        (2, 1, 64, 64),
        (2, 251, 8, 64),
        is_causal=True,
        bottom_right_align=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,1/253,64/8,64 seqlen=513 left-window decode phase",
        (2, 1, 64, 64),
        (2, 253, 8, 64),
        is_causal=True,
        bottom_right_align=True,
        window_size=(128, -1),
    ),
)


PORT_STATUS = (
    "Native fused D32/D64/D128 FP16/BF16/E4M3 fixed and varlen kernel.",
    "Dense, causal, bottom-right, sliding-window, and residual-tail scheduling.",
    "MHA/GQA, LSE, sink logits, scales, FP8 output, and correction controls.",
    "Source-like 512-thread role split, Q2/KV3/O2 stages, TMA, TMEM, tcgen05, and CLC.",
    "Timed launches support CUDA-graph replay and rotating cold-L2 workspaces.",
    "FP8/D32 issues the intended single QK phase; the source's two floor-divided loops issue none.",
)

# Retain the original public diagnostic name for scripts written during the
# migration while reporting the final implementation state accurately.
PHASE1_LIMITATIONS = PORT_STATUS


def phase1_report() -> str:
    lines = ["CUDA Lang FMHA port status:"]
    lines.extend(f"  - {item}" for item in PHASE1_LIMITATIONS)
    return "\n".join(lines)


def _make_tma_view(tensor: torch.Tensor) -> torch.Tensor:
    """Present contiguous BSHD/THD as fastest-first TMA coordinates."""
    if tensor.ndim not in (3, 4) or tensor.shape[-1] not in (32, 64, 128):
        raise ValueError("TMA view requires a contiguous B,S,H,D or T,H,D tensor")
    if not tensor.is_contiguous():
        raise ValueError("FMHA tensors must be contiguous")
    if tensor.ndim == 3:
        total_sequence, heads, depth = tensor.shape
        return torch.as_strided(
            tensor,
            size=(depth, heads, total_sequence),
            stride=(1, depth, heads * depth),
        )
    batch, sequence, heads, depth = tensor.shape
    return torch.as_strided(
        tensor,
        size=(depth, heads, sequence, batch),
        stride=(1, depth, heads * depth, sequence * heads * depth),
    )


def prepare_tensors(
    config: FmhaConfig,
    *,
    seed: int = 1111,
    q_input=None,
    k_input=None,
    v_input=None,
) -> dict[str, torch.Tensor]:
    prepared = fmha_utils.prepare_tensors(
        config,
        device="cuda",
        seed=seed,
        q_input=q_input,
        k_input=k_input,
        v_input=v_input,
    )
    tensors = prepared.as_mapping()
    # CUDA Lang launch arguments cannot be ``None``. Keep the public mapping's
    # optional values intact while retaining tiny launch-only placeholders.
    tensors["_lse_arg"] = (
        prepared.lse
        if prepared.lse is not None
        else torch.empty((1,), device="cuda", dtype=torch.float32)
    )
    tensors["_sinks_arg"] = (
        prepared.sinks
        if prepared.sinks is not None
        else torch.empty((1,), device="cuda", dtype=torch.float16)
    )
    tensors["_cum_seqlen_q_arg"] = (
        prepared.cumulative_q
        if prepared.cumulative_q is not None
        else torch.zeros((1,), device="cuda", dtype=torch.int32)
    )
    tensors["_cum_seqlen_k_arg"] = (
        prepared.cumulative_k
        if prepared.cumulative_k is not None
        else torch.zeros((1,), device="cuda", dtype=torch.int32)
    )
    return tensors


def launch(
    tensors: dict[str, torch.Tensor],
    config: FmhaConfig,
    *,
    stream=None,
) -> None:
    config.require_phase1_kernel()
    if stream is None:
        stream = torch.cuda.current_stream()
    plan = fmha_utils.make_tensor_plan(config)
    seq_q = max(plan.q_lengths)
    seq_k = max(plan.k_lengths)
    grid = math.ceil(seq_q / CTA_M) * config.batch * config.heads_q
    window_left, window_right = config.effective_window
    output_launch = tensors["out"]
    if config.out_dtype == "Float8E4M3FN":
        # CUDA Lang cannot lower pointer arithmetic on a torch FP8 tensor.
        # The byte view is zero-copy and matches the kernel's explicit E4M3
        # packing and UINT8 tensor-map encoding.
        output_launch = output_launch.view(torch.int8)
    cl.launch(
        stream,
        (grid,),
        (THREADS_PER_CTA,),
        _fmha_prefill_kernel,
        (
            _make_tma_view(tensors["q"]),
            _make_tma_view(tensors["k"]),
            _make_tma_view(tensors["v"]),
            _make_tma_view(output_launch),
            tensors["_lse_arg"].reshape(-1),
            tensors["_sinks_arg"],
            tensors["_cum_seqlen_q_arg"],
            tensors["_cum_seqlen_k_arg"],
            config.batch,
            seq_q,
            seq_k,
            plan.total_q,
            config.heads_q,
            config.heads_k,
            config.head_dim,
            math.ceil(seq_q / CTA_M),
            config.batch,
            config.heads_q,
            {
                "Float16": ELEMENT_F16,
                "BFloat16": ELEMENT_BF16,
                "Float8E4M3FN": ELEMENT_E4M3,
            }[config.in_dtype],
            {
                "Float16": ELEMENT_F16,
                "BFloat16": ELEMENT_BF16,
                "Float8E4M3FN": ELEMENT_E4M3,
            }[config.out_dtype],
            config.attention_scale_log2,
            config.attention_scale,
            config.output_scale,
            config.lse_calculation,
            config.use_sinks,
            config.enable_skip_correction,
            config.enable_approx_epilogue_rcp,
            config.is_causal,
            config.bottom_right_align,
            -1 if window_left is None else window_left,
            -1 if window_right is None else window_right,
            config.variable_length,
        ),
        block_in_cluster_count=(1, 1, 1),
    )


def torch_reference(
    tensors: dict[str, torch.Tensor], config: FmhaConfig
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked source-style FP32 reference for fixed and jagged storage."""
    prepared = fmha_utils.prepared_from_mapping(config, tensors)
    result = fmha_utils.torch_reference(prepared)
    return result.output, result.lse


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual - expected).abs()
    max_abs = difference.max().item()
    denominator = expected.abs().clamp_min(torch.finfo(torch.float32).tiny)
    max_rel = (difference / denominator).max().item()
    return max_abs, max_rel


def verify(
    tensors: dict[str, torch.Tensor],
    config: FmhaConfig,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, float]:
    prepared = fmha_utils.prepared_from_mapping(config, tensors)
    result = fmha_utils.verify(prepared, tolerance=tolerance)
    output = result.output
    metrics = {
        "output_max_abs": output.max_abs,
        "output_max_rel": output.max_rel,
        "atol": output.atol,
        "rtol": output.rtol,
    }
    print(
        f"output: max_abs={output.max_abs:.6g}, max_rel={output.max_rel:.6g}, "
        f"atol={output.atol:.6g}, rtol={output.rtol:.6g}"
    )
    if config.lse_calculation:
        assert result.lse is not None
        metrics["lse_max_abs"] = result.lse.max_abs
        metrics["lse_max_rel"] = result.lse.max_rel
        print(
            f"LSE: max_abs={result.lse.max_abs:.6g}, "
            f"max_rel={result.lse.max_rel:.6g}, "
            f"atol={result.lse.atol:.6g}, rtol={result.lse.rtol:.6g}"
        )
    return metrics


def _workspace_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in tensors.items()
        if name
        in (
            "q",
            "k",
            "v",
            "out",
            "lse",
            "sinks",
            "cum_seqlen_q",
            "cum_seqlen_k",
        )
        and tensor is not None
    )


def benchmark(
    config: FmhaConfig,
    tensors: dict[str, torch.Tensor],
    warmup_iterations: int,
    iterations: int,
    *,
    use_cold_l2: bool = False,
    use_cuda_graphs: bool = True,
) -> float:
    if warmup_iterations < 0 or iterations < 0:
        raise ValueError("warmup_iterations and iterations must be nonnegative")
    workspaces = [tensors]
    if use_cold_l2:
        target_bytes = 256 * 1024 * 1024
        workspace_bytes = _workspace_bytes(tensors)
        count = max(2, math.ceil(target_bytes / workspace_bytes))
        count = min(count, 16)
        workspaces.extend(
            prepare_tensors(config, seed=1111 + i) for i in range(1, count)
        )
        print(
            "Cold-L2 workspaces: "
            f"count={len(workspaces)}, bytes_each={workspace_bytes}, "
            f"bytes_total={len(workspaces) * workspace_bytes}"
        )
    if iterations == 0:
        for index in range(warmup_iterations):
            launch(workspaces[index % len(workspaces)], config)
        torch.cuda.synchronize()
        return 0.0

    if use_cuda_graphs:
        # Match the CUTLASS testing helper's semantics: capture the complete
        # warmup and profiling loops on a non-default stream, replay each graph
        # once, and divide the profiled graph time by its launch count.  Each
        # workspace has static graph arguments, so cold-L2 rotation is retained.
        workspace_index = 0
        if warmup_iterations:
            warmup_graph = torch.cuda.CUDAGraph()
            warmup_capture = torch.cuda.graph(
                warmup_graph, capture_error_mode="thread_local"
            )
            with warmup_capture:
                for _ in range(warmup_iterations):
                    launch(workspaces[workspace_index], config)
                    workspace_index = (workspace_index + 1) % len(workspaces)
            warmup_graph.replay()

        profiling_graph = torch.cuda.CUDAGraph()
        profiling_capture = torch.cuda.graph(
            profiling_graph, capture_error_mode="thread_local"
        )
        with profiling_capture:
            for _ in range(iterations):
                launch(workspaces[workspace_index], config)
                workspace_index = (workspace_index + 1) % len(workspaces)

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        # CUDAGraph.replay() enqueues the graph on the current stream, not the
        # stream that was used while capturing it.  Keep both timing events on
        # that replay stream so they actually enclose the graph execution.
        replay_stream = torch.cuda.current_stream()
        start.record(replay_stream)
        profiling_graph.replay()
        stop.record(replay_stream)
        stop.synchronize()
        elapsed_us = start.elapsed_time(stop) * 1000.0 / iterations
        print(f"Benchmark mode: CUDA graph ({iterations} captured launches)")
        print(f"Kernel time: {elapsed_us:.3f} us")
        return elapsed_us

    for index in range(warmup_iterations):
        launch(workspaces[index % len(workspaces)], config)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in range(iterations):
        launch(workspaces[index % len(workspaces)], config)
    stop.record()
    stop.synchronize()
    elapsed_us = start.elapsed_time(stop) * 1000.0 / iterations
    print(f"Kernel time: {elapsed_us:.3f} us")
    return elapsed_us


def run(
    q_shape: ShapeSpec,
    k_shape: ShapeSpec,
    in_dtype: str = "Float16",
    out_dtype: str = "Float16",
    qk_acc_dtype: str = "Float32",
    pv_acc_dtype: str = "Float32",
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER,
    is_causal: bool = False,
    bottom_right_align: bool = False,
    lse_calculation: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    inv_scale_o: float = 1.0,
    scale_softmax: float = 0.0,
    tolerance: float = DEFAULT_TOLERANCE,
    warmup_iterations: int = 1,
    iterations: int = 0,
    skip_ref_check: bool = False,
    skip_dump_config: bool = False,
    use_cold_l2: bool = False,
    use_cuda_graphs: bool = True,
    keep_ptx: bool = False,
    q_input_np=None,
    k_input_np=None,
    v_input_np=None,
    use_sinks: bool = False,
    enable_skip_correction: bool = True,
    enable_approx_epilogue_rcp: bool = True,
) -> tuple:
    config = FmhaConfig(
        q_shape=q_shape,
        k_shape=k_shape,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        qk_acc_dtype=qk_acc_dtype,
        pv_acc_dtype=pv_acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        is_causal=is_causal,
        bottom_right_align=bottom_right_align,
        lse_calculation=lse_calculation,
        window_size=window_size,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
        inv_scale_o=inv_scale_o,
        scale_softmax=scale_softmax,
        use_sinks=use_sinks,
        enable_skip_correction=enable_skip_correction,
        enable_approx_epilogue_rcp=enable_approx_epilogue_rcp,
    )
    config.require_phase1_kernel()
    if keep_ptx:
        logs = os.environ.get("CUDA_LANG_LOGS", "")
        os.environ["CUDA_LANG_LOGS"] = ",".join(filter(None, (logs, "PTX")))
    if not skip_dump_config:
        print(f"config: {config}")
    tensors = prepare_tensors(
        config,
        q_input=q_input_np,
        k_input=k_input_np,
        v_input=v_input_np,
    )
    launch(tensors, config)
    torch.cuda.synchronize()
    metrics = None
    if not skip_ref_check:
        metrics = verify(tensors, config, tolerance)
    elapsed_us = benchmark(
        config,
        tensors,
        warmup_iterations,
        iterations,
        use_cold_l2=use_cold_l2,
        use_cuda_graphs=use_cuda_graphs,
    )
    plan = fmha_utils.make_tensor_plan(config)
    causal_factor = 0.5 if is_causal and window_size[0] == -1 else 1.0
    query_key_pairs = sum(
        seq_q * seq_k for seq_q, seq_k in zip(plan.q_lengths, plan.k_lengths)
    )
    flops = 4.0 * config.heads_q * config.head_dim * query_key_pairs
    flops *= causal_factor
    tflops = 0.0 if elapsed_us == 0.0 else flops / (elapsed_us * 1.0e-6) / 1.0e12
    if not skip_dump_config and elapsed_us:
        print(f"Throughput: {tflops:.3f} TFLOP/s")
    if skip_ref_check:
        return elapsed_us, tflops
    return elapsed_us, tflops, tensors["out"].float().cpu(), metrics


def run_prefill_correctness_tests() -> None:
    unsupported: list[str] = []
    failures: list[str] = []
    for index, test in enumerate(_PREFILL_CORRECTNESS_TESTS, start=1):
        print(f"[{index}/{len(_PREFILL_CORRECTNESS_TESTS)}] {test.name}")
        config = test.config()
        try:
            run(
                config.q_shape,
                config.k_shape,
                config.in_dtype,
                config.out_dtype,
                config.qk_acc_dtype,
                config.pv_acc_dtype,
                config.mma_tiler_mn,
                config.is_causal,
                config.bottom_right_align,
                config.lse_calculation,
                config.window_size,
                config.scale_q,
                config.scale_k,
                config.scale_v,
                config.inv_scale_o,
                config.scale_softmax,
                DEFAULT_TOLERANCE,
                0,
                0,
                False,
                True,
            )
            print("PASS")
        except UnsupportedFmhaConfiguration as error:
            unsupported.append(f"{test.name}: {error}")
            print(f"UNSUPPORTED: {error}")
        except Exception as error:  # keep the complete matrix report
            failures.append(f"{test.name}: {error}")
            print(f"FAIL: {error}")
    if failures or unsupported:
        details = "\n".join(
            [
                *(f"FAIL {item}" for item in failures),
                *(f"TODO {item}" for item in unsupported),
            ]
        )
        raise RuntimeError(
            "FMHA correctness matrix is incomplete; no case was reclassified "
            f"as passed:\n{details}"
        )


def run_focused_feature_tests() -> None:
    cases = (
        ("canonical LSE", FmhaConfig(lse_calculation=True)),
        ("canonical sinks", FmhaConfig(use_sinks=True)),
        (
            "FP8 finite non-default scales",
            FmhaConfig(
                in_dtype="Float8E4M3FN",
                out_dtype="Float8E4M3FN",
                scale_q=0.75,
                scale_k=1.25,
                scale_v=0.5,
                inv_scale_o=1.5,
                scale_softmax=0.2,
            ),
        ),
        (
            "skip correction disabled",
            FmhaConfig(enable_skip_correction=False),
        ),
    )
    incomplete: list[str] = []
    for name, config in cases:
        print(f"focused: {name}")
        try:
            run(
                config.q_shape,
                config.k_shape,
                config.in_dtype,
                config.out_dtype,
                config.qk_acc_dtype,
                config.pv_acc_dtype,
                config.mma_tiler_mn,
                config.is_causal,
                config.bottom_right_align,
                config.lse_calculation,
                config.window_size,
                config.scale_q,
                config.scale_k,
                config.scale_v,
                config.inv_scale_o,
                config.scale_softmax,
                DEFAULT_TOLERANCE,
                0,
                0,
                False,
                True,
                use_sinks=config.use_sinks,
                enable_skip_correction=config.enable_skip_correction,
            )
            print("PASS")
        except UnsupportedFmhaConfiguration as error:
            incomplete.append(f"{name}: {error}")
            print(f"UNSUPPORTED: {error}")
    if incomplete:
        raise RuntimeError(
            "focused feature tests incomplete:\n" + "\n".join(incomplete)
        )


def _parse_shape(value: str) -> ShapeSpec:
    """Parse `B,S,H,D` or source-style `B,(S0,S1),H,D`."""
    text = value.strip()
    try:
        if "(" not in text:
            fields = tuple(int(item.strip()) for item in text.split(","))
            if len(fields) != 4:
                raise ValueError
            return fields  # type: ignore[return-value]
        start = text.index("(")
        end = text.index(")", start)
        before = [item for item in text[:start].rstrip(",").split(",") if item]
        nested = tuple(int(item.strip()) for item in text[start + 1:end].split(","))
        after = [item for item in text[end + 1:].lstrip(",").split(",") if item]
        fields: list[int | tuple[int, ...]] = [
            *(int(item.strip()) for item in before),
            nested,
            *(int(item.strip()) for item in after),
        ]
        if len(fields) != 4:
            raise ValueError
        return tuple(fields)  # type: ignore[return-value]
    except (ValueError, IndexError) as error:
        raise argparse.ArgumentTypeError(
            "shape must be B,S,H,D or B,(S0,S1,...),H,D"
        ) from error


def _parse_pair(value: str) -> tuple[int, int]:
    try:
        fields = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected two comma-separated integers"
        ) from error
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers")
    return fields


def _parse_dtype(value: str) -> str:
    aliases = {
        "float16": "Float16",
        "f16": "Float16",
        "half": "Float16",
        "bfloat16": "BFloat16",
        "bf16": "BFloat16",
        "float8e4m3fn": "Float8E4M3FN",
        "fp8": "Float8E4M3FN",
        "float32": "Float32",
        "f32": "Float32",
    }
    key = value.replace("_", "").lower()
    result = aliases.get(key)
    if result is None:
        raise argparse.ArgumentTypeError(f"unsupported dtype {value!r}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Native CUDA Lang Blackwell FMHA prefill tutorial",
    )
    parser.add_argument(
        "--in_dtype", "--in-dtype", type=_parse_dtype, default="Float16"
    )
    parser.add_argument(
        "--out_dtype", "--out-dtype", type=_parse_dtype, default="Float16"
    )
    parser.add_argument(
        "--qk_acc_dtype", "--qk-acc-dtype", type=_parse_dtype, default="Float32"
    )
    parser.add_argument(
        "--pv_acc_dtype", "--pv-acc-dtype", type=_parse_dtype, default="Float32"
    )
    parser.add_argument(
        "--mma_tiler_mn", "--mma-tiler-mn", type=_parse_pair, default=DEFAULT_MMA_TILER
    )
    parser.add_argument("--is_causal", "--is-causal", action="store_true")
    parser.add_argument(
        "--bottom_right_align", "--bottom-right-align", action="store_true"
    )
    parser.add_argument("--lse_calculation", "--lse-calculation", action="store_true")
    parser.add_argument(
        "--disable_skip_correction",
        "--disable-skip-correction",
        action="store_false",
        dest="enable_skip_correction",
        default=True,
    )
    parser.add_argument(
        "--disable_approx_epilogue_rcp",
        "--disable-approx-epilogue-rcp",
        action="store_false",
        dest="enable_approx_epilogue_rcp",
        default=True,
    )
    parser.add_argument(
        "--window_size", "--window-size", type=_parse_pair, default=(-1, -1)
    )
    parser.add_argument(
        "--q_shape", "--q-shape", type=_parse_shape, default=DEFAULT_Q_SHAPE
    )
    parser.add_argument(
        "--k_shape", "--k-shape", type=_parse_shape, default=DEFAULT_K_SHAPE
    )
    parser.add_argument("--scale_q", "--scale-q", type=float, default=1.0)
    parser.add_argument("--scale_k", "--scale-k", type=float, default=1.0)
    parser.add_argument("--scale_v", "--scale-v", type=float, default=1.0)
    parser.add_argument("--inv_scale_o", "--inv-scale-o", type=float, default=1.0)
    parser.add_argument("--scale_softmax", "--scale-softmax", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--warmup_iterations", "--warmup-iterations", type=int, default=10
    )
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--skip_ref_check", "--skip-ref-check", action="store_true")
    parser.add_argument("--skip_dump_config", "--skip-dump-config", action="store_true")
    parser.add_argument("--use_cold_l2", "--use-cold-l2", action="store_true")
    parser.add_argument(
        "--no_cuda_graphs",
        "--no-cuda-graphs",
        action="store_false",
        dest="use_cuda_graphs",
        default=True,
        help="Use direct timed launches instead of CUDA-graph replay",
    )
    parser.add_argument("--use_sinks", "--use-sinks", action="store_true")
    parser.add_argument("--keep_ptx", "--keep-ptx", action="store_true")
    parser.add_argument(
        "--run-correctness-tests", "--run_correctness_tests", action="store_true"
    )
    parser.add_argument(
        "--run-focused-feature-tests",
        "--run_focused_feature_tests",
        action="store_true",
    )
    parser.add_argument("--phase1-report", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.phase1_report:
        print(phase1_report())
        return
    if args.run_correctness_tests:
        run_prefill_correctness_tests()
        print("ALL CORRECTNESS TESTS PASSED")
        return
    if args.run_focused_feature_tests:
        run_focused_feature_tests()
        print("ALL FOCUSED FEATURE TESTS PASSED")
        return
    run(
        args.q_shape,
        args.k_shape,
        args.in_dtype,
        args.out_dtype,
        args.qk_acc_dtype,
        args.pv_acc_dtype,
        args.mma_tiler_mn,
        args.is_causal,
        args.bottom_right_align,
        args.lse_calculation,
        args.window_size,
        args.scale_q,
        args.scale_k,
        args.scale_v,
        args.inv_scale_o,
        args.scale_softmax,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.skip_dump_config,
        args.use_cold_l2,
        args.use_cuda_graphs,
        keep_ptx=args.keep_ptx,
        use_sinks=args.use_sinks,
        enable_skip_correction=args.enable_skip_correction,
        enable_approx_epilogue_rcp=args.enable_approx_epilogue_rcp,
    )
    print("PASS")


if __name__ == "__main__":
    main()
