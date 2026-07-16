# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Two-CTA, four-warpgroup pipeline and fixed 192/128 head dimensions.
See thunderkittens bf16_b300_mha_noncausal.cu for reference.
"""

import math

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability
import pytest
import torch

cc = get_compute_capability()
if tuple(cc) != (10, 0):
    pytest.skip("requires Blackwell", allow_module_level=True)


WARP_SIZE = 32
WARPGROUP_SIZE = 128
THREADS = 4 * WARPGROUP_SIZE
CLUSTER_SIZE = 2
NUM_SMS = 148

BLOCK_M = 128
BLOCK_N = 128
HEAD_DIM_QK = 192
HEAD_DIM_V = 128
LOAD_STAGES = 3
MMA_K = 16
SCALE_LOG2 = 1.44269504089 / math.sqrt(HEAD_DIM_QK)
LN2 = math.log(2.0)
MAX_SHARED_MEMORY = 227 * 1024
DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024

Q_TILE_ELEMENTS = BLOCK_M * HEAD_DIM_QK
KV_TILE_ELEMENTS = (BLOCK_N // 2) * HEAD_DIM_QK
O_TILE_ELEMENTS = BLOCK_M * HEAD_DIM_V
DYNAMIC_STORAGE_BYTES = (
    2 * Q_TILE_ELEMENTS * 2
    + LOAD_STAGES * KV_TILE_ELEMENTS * 2
    + O_TILE_ELEMENTS * 2
    + 2 * 2 * BLOCK_M * 4
)


def wait_mbarrier(mbar, phase):
    while not cl.mbarrier_try_wait_parity(mbar, phase, time_hint=10_000):
        pass


def fast_exp2(value):
    (result,) = cl._inline_ptx(
        "ex2.approx.ftz.f32 %0, %1;",
        ("=f", cl.float32),
        ("f", value),
    )
    return result


def fast_log2(value):
    return cl._nvvm.lg2_approx_ftz_f(value)


def max_vector32(values, current):
    for i in cl.static_iter(range(32)):
        current = cl.maximum(current, cl.bitcast(values[i], cl.float32))
    return current


def probability_vector(values, scale, offset):
    exps = tuple(
        fast_exp2(cl.bitcast(values[i], cl.float32) * scale + offset)
        for i in cl.static_iter(range(32))
    )
    total = exps[0]
    for exp in cl.static_iter(exps[1:]):
        total += exp
    as_bf16 = tuple(
        cl._nvvm.ff2bf16x2_rn(exps[2 * i + 1], exps[2 * i])
        for i in cl.static_iter(range(16))
    )
    as_i32 = tuple(cl.bitcast(value, cl.int32) for value in cl.static_iter(as_bf16))
    return cl.Vector(*as_i32), total


def scale_vector16(values, scale):
    floats = tuple(cl.bitcast(values[i], cl.float32) for i in cl.static_iter(range(16)))
    scaled = tuple(value * scale for value in cl.static_iter(floats))
    ints = tuple(cl.bitcast(value, cl.int32) for value in cl.static_iter(scaled))
    return cl.Vector(*ints)


def store_output_pairs(o_smem, values, inv_norm, row, column):
    out_ptr = cl.bitcast(
        o_smem.get_base_pointer(),
        cl.pointer_dtype(cl.uint32, cl.MemorySpace.SHARED),
    )
    for i in cl.static_iter(range(8)):
        lo = cl.bitcast(values[2 * i], cl.float32) * inv_norm
        hi = cl.bitcast(values[2 * i + 1], cl.float32) * inv_norm
        packed = cl.bitcast(cl._nvvm.ff2bf16x2_rn(hi, lo), cl.uint32)
        logical_element = (column // 64) * BLOCK_M * 64 + row * 64 + column % 64 + 2 * i
        byte_offset = logical_element * 2
        swizzled = byte_offset ^ (((byte_offset & 0x380) >> 7) << 4)
        (out_ptr + swizzled // 4).store(packed, alignment=4)


def qk_descriptor(pointer, row_count, chunk):
    base = cl.Tcgen05SharedMemoryDescriptor(
        matrix_start_address=pointer,
        leading_dimension_byte_offset=16,
        stride_dimension_byte_offset=8 * 128,
        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
    ).encode()
    return base + 2 * (chunk % 4) + (row_count // 16) * 128 * (chunk // 4)


def v_descriptor(pointer, chunk):
    base = cl.Tcgen05SharedMemoryDescriptor(
        matrix_start_address=pointer,
        leading_dimension_byte_offset=4096,
        stride_dimension_byte_offset=8 * 128,
        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
    ).encode()
    return base + 128 * chunk


@cl.kernel(
    max_threads_per_block=(THREADS,),
    max_blocks_per_cluster=CLUSTER_SIZE,
    min_blocks_per_sm=1,
)
def mha_kernel(
    q,
    k,
    v,
    o,
    lse,
    batch: cl.Constant[int],
    q_sequence: cl.Constant[int],
    kv_sequence: cl.Constant[int],
    heads: cl.Constant[int],
):
    q_smem = cl.shared_array(
        (2, Q_TILE_ELEMENTS), cl.bfloat16, dynamic=True, alignment=1024
    )
    kv_smem = cl.shared_array(
        (LOAD_STAGES, KV_TILE_ELEMENTS),
        cl.bfloat16,
        dynamic=True,
        alignment=1024,
    )
    o_smem = cl.shared_array(O_TILE_ELEMENTS, cl.bfloat16, dynamic=True, alignment=1024)
    max_vec = cl.shared_array((2, BLOCK_M), cl.float32, dynamic=True, alignment=1024)
    lse_vec = cl.shared_array((2, BLOCK_M), cl.float32, dynamic=True, alignment=1024)
    cl.shared_array(
        DYNAMIC_SHARED_MEMORY - DYNAMIC_STORAGE_BYTES, cl.int8, dynamic=True
    )

    q_arrived = cl.shared_array(2, cl.mbarrier, alignment=8)
    q_finished = cl.shared_array(2, cl.mbarrier, alignment=8)
    kv_arrived = cl.shared_array(LOAD_STAGES, cl.mbarrier, alignment=8)
    kv_finished = cl.shared_array(LOAD_STAGES, cl.mbarrier, alignment=8)
    scores_arrived = cl.shared_array(2, cl.mbarrier, alignment=8)
    norm_scores_arrived = cl.shared_array(2, cl.mbarrier, alignment=8)
    norm_quarter_arrived = cl.shared_array((3, 2), cl.mbarrier, alignment=8)
    corr_arrived = cl.shared_array(2, cl.mbarrier, alignment=8)
    tile_arrived = cl.shared_array(2, cl.mbarrier, alignment=8)
    rescale_finished = cl.shared_array(2, cl.mbarrier, alignment=8)
    tmem_storage = cl.shared_array(
        1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
    )

    tid = cl.thread_index(0)
    warp = tid // WARP_SIZE
    warpgroup = tid // WARPGROUP_SIZE
    warp_in_group = (tid % WARPGROUP_SIZE) // WARP_SIZE
    lane_in_group = tid % WARPGROUP_SIZE
    rank = cl.block_in_cluster_index(0)

    q_tmap = cl.tensor_map_tiled(
        q, (64, BLOCK_M, HEAD_DIM_QK // 64, 1, 1), swizzle=cl.SwizzleMode.SWIZZLE_128B
    )
    k_tmap = cl.tensor_map_tiled(
        k,
        (64, BLOCK_N // 2, HEAD_DIM_QK // 64, 1, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    v_tmap = cl.tensor_map_tiled(
        v, (64, BLOCK_N, 1, 1, 1), swizzle=cl.SwizzleMode.SWIZZLE_128B
    )
    o_tmap = cl.tensor_map_tiled(
        o, (64, BLOCK_M, HEAD_DIM_V // 64, 1, 1), swizzle=cl.SwizzleMode.SWIZZLE_128B
    )

    if tid == 0:
        cl.prefetch_tensor_map(q_tmap)
        cl.prefetch_tensor_map(k_tmap)
        cl.prefetch_tensor_map(v_tmap)
        cl.prefetch_tensor_map(o_tmap)
        for qid in cl.static_iter(range(2)):
            cl.mbarrier_initialize(q_arrived.get_element_pointer(qid), 1)
            cl.mbarrier_initialize(q_finished.get_element_pointer(qid), 1)
            cl.mbarrier_initialize(scores_arrived.get_element_pointer(qid), 1)
            cl.mbarrier_initialize(norm_scores_arrived.get_element_pointer(qid), 10)
            for quarter in cl.static_iter(range(3)):
                cl.mbarrier_initialize(
                    norm_quarter_arrived.get_element_pointer((quarter, qid)), 8
                )
            cl.mbarrier_initialize(corr_arrived.get_element_pointer(qid), 4)
            cl.mbarrier_initialize(tile_arrived.get_element_pointer(qid), 1)
            cl.mbarrier_initialize(rescale_finished.get_element_pointer(qid), 1)
        for stage in cl.static_iter(range(LOAD_STAGES)):
            cl.mbarrier_initialize(kv_arrived.get_element_pointer(stage), 1)
            cl.mbarrier_initialize(kv_finished.get_element_pointer(stage), 2)
        cl.fence_mbarrier_initialize()

    if warp == 0:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(), 512, cta_group=cl.CTAGroup.CTA_2
        )
    cl.tcgen05_fence_before_thread_sync()
    cl.barrier_sync_block()
    cl.tcgen05_fence_after_thread_sync()
    cl.barrier_sync_cluster(aligned=True)

    if warpgroup == 3:
        cl.setmaxregister_decrease(128)
    elif warpgroup == 2:
        cl.setmaxregister_decrease(48)
    else:
        cl.setmaxregister_increase(168)

    total_bids = batch * heads * (q_sequence // (BLOCK_M * 2))
    iterations = kv_sequence // BLOCK_N

    if warp == 15 and cl.elect_sync():
        kv_index = 0
        kv_phase = 1
        q_phase = 1
        current_bid = cl.block_index(0)
        while current_bid < total_bids:
            cluster_linear = current_bid // CLUSTER_SIZE
            clusters_m = (q_sequence // BLOCK_M) // 4
            clusters_per_batch = heads * clusters_m
            batch_idx = cluster_linear // clusters_per_batch
            rem = cluster_linear - batch_idx * clusters_per_batch
            head_idx = rem // clusters_m
            m_base = (rem - head_idx * clusters_m) * 4

            for qid in cl.static_iter(range(2)):
                wait_mbarrier(q_finished.get_element_pointer(qid), q_phase)
                q_dst = cl.map_shared_to_cluster(
                    q_smem.get_element_pointer((qid, 0)), rank
                )
                q_bar = q_arrived.get_element_pointer(qid)
                cl.copy_async_bulk_tensor_global_to_shared(
                    q_tmap,
                    (0, (m_base + rank * 2 + qid) * BLOCK_M, 0, head_idx, batch_idx),
                    q_dst,
                    cl.map_shared_to_leader_block(q_bar),
                    multicast_mask=cl.int16(1 << rank),
                    cta_group=cl.CTAGroup.CTA_2,
                )

            for key_block in range(iterations):
                wait_mbarrier(kv_finished.get_element_pointer(kv_index), kv_phase)
                k_dst = cl.map_shared_to_cluster(
                    kv_smem.get_element_pointer((kv_index, 0)), rank
                )
                k_bar = kv_arrived.get_element_pointer(kv_index)
                cl.copy_async_bulk_tensor_global_to_shared(
                    k_tmap,
                    (
                        0,
                        key_block * BLOCK_N + rank * (BLOCK_N // 2),
                        0,
                        head_idx,
                        batch_idx,
                    ),
                    k_dst,
                    cl.map_shared_to_leader_block(k_bar),
                    multicast_mask=cl.int16(1 << rank),
                    cta_group=cl.CTAGroup.CTA_2,
                )
                kv_index += 1
                if kv_index == LOAD_STAGES:
                    kv_index = 0
                    kv_phase ^= 1

                wait_mbarrier(kv_finished.get_element_pointer(kv_index), kv_phase)
                v_dst = cl.map_shared_to_cluster(
                    kv_smem.get_element_pointer((kv_index, 0)), rank
                )
                v_bar = kv_arrived.get_element_pointer(kv_index)
                cl.copy_async_bulk_tensor_global_to_shared(
                    v_tmap,
                    (0, key_block * BLOCK_N, rank, head_idx, batch_idx),
                    v_dst,
                    cl.map_shared_to_leader_block(v_bar),
                    multicast_mask=cl.int16(1 << rank),
                    cta_group=cl.CTAGroup.CTA_2,
                )
                kv_index += 1
                if kv_index == LOAD_STAGES:
                    kv_index = 0
                    kv_phase ^= 1

            q_phase ^= 1
            current_bid += NUM_SMS

    elif warp == 12 and rank == 0 and cl.elect_sync():
        qk_instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
            b_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
            n=BLOCK_N,
            m=BLOCK_M * CLUSTER_SIZE,
        ).encode()
        pv_instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
            b_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
            transpose_b=True,
            n=HEAD_DIM_V,
            m=BLOCK_M * CLUSTER_SIZE,
        ).encode()

        kv_index = 0
        kv_phase = 0
        current_bid = cl.block_index(0)
        task_number = 0
        while current_bid < total_bids:
            q_phase = task_number & 1
            norm_phase = (task_number * iterations) & 1
            for qid in cl.static_iter(range(2)):
                cl.mbarrier_arrive_expect_transaction(
                    q_arrived.get_element_pointer(qid),
                    CLUSTER_SIZE * Q_TILE_ELEMENTS * 2,
                    scope=cl.MbarrierScope.BLOCK,
                )
                wait_mbarrier(q_arrived.get_element_pointer(qid), q_phase)

            k_stage = kv_index
            cl.mbarrier_arrive_expect_transaction(
                kv_arrived.get_element_pointer(k_stage),
                CLUSTER_SIZE * KV_TILE_ELEMENTS * 2,
                scope=cl.MbarrierScope.BLOCK,
            )
            wait_mbarrier(kv_arrived.get_element_pointer(k_stage), kv_phase)
            cl.tcgen05_fence_after_thread_sync()
            for qid in cl.static_iter(range(2)):
                score_tmem = cl.tcgen05_tmem_offset(
                    tmem_storage[0], column_offset=qid * 256
                )
                for chunk in cl.static_iter(range(HEAD_DIM_QK // MMA_K)):
                    cl.tcgen05_mma(
                        cl.Tcgen05MMAKind.F16,
                        score_tmem,
                        qk_descriptor(
                            q_smem.get_element_pointer((qid, 0)), BLOCK_M, chunk
                        ),
                        qk_descriptor(
                            kv_smem.get_element_pointer((k_stage, 0)),
                            BLOCK_N // 2,
                            chunk,
                        ),
                        qk_instruction,
                        accumulate=chunk != 0,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                cl.tcgen05_commit(
                    kv_finished.get_element_pointer(k_stage),
                    multicast_mask=0b11,
                    cta_group=cl.CTAGroup.CTA_2,
                )
                cl.tcgen05_commit(
                    scores_arrived.get_element_pointer(qid),
                    multicast_mask=0b11,
                    cta_group=cl.CTAGroup.CTA_2,
                )
            if iterations == 1:
                for qid in cl.static_iter(range(2)):
                    cl.tcgen05_commit(
                        q_finished.get_element_pointer(qid),
                        multicast_mask=0b11,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
            kv_index += 1
            if kv_index == LOAD_STAGES:
                kv_index = 0
                kv_phase ^= 1

            for key_block in range(iterations):
                v_stage = kv_index
                cl.mbarrier_arrive_expect_transaction(
                    kv_arrived.get_element_pointer(v_stage),
                    CLUSTER_SIZE * BLOCK_N * (HEAD_DIM_V // 2) * 2,
                    scope=cl.MbarrierScope.BLOCK,
                )
                wait_mbarrier(kv_arrived.get_element_pointer(v_stage), kv_phase)

                for qid in cl.static_iter(range(2)):
                    wait_mbarrier(
                        norm_scores_arrived.get_element_pointer(qid), norm_phase
                    )
                    output_tmem = cl.tcgen05_tmem_offset(
                        tmem_storage[0], column_offset=qid * 256 + 128
                    )
                    for quarter in cl.static_iter(range(4)):
                        if quarter > 0:
                            wait_mbarrier(
                                norm_quarter_arrived.get_element_pointer(
                                    (quarter - 1, qid)
                                ),
                                norm_phase,
                            )
                        v_ptr = kv_smem.get_element_pointer(
                            (v_stage, quarter * 32 * 64)
                        )
                        for chunk in cl.static_iter(range(2)):
                            cl.tcgen05_mma(
                                cl.Tcgen05MMAKind.F16,
                                output_tmem,
                                cl.tcgen05_tmem_offset(
                                    tmem_storage[0],
                                    column_offset=qid * 256 + quarter * 16 + chunk * 8,
                                ),
                                v_descriptor(v_ptr, chunk),
                                pv_instruction,
                                accumulate=(
                                    key_block != 0 or quarter != 0 or chunk != 0
                                ),
                                cta_group=cl.CTAGroup.CTA_2,
                            )
                    cl.tcgen05_commit(
                        kv_finished.get_element_pointer(v_stage),
                        multicast_mask=0b11,
                        cta_group=cl.CTAGroup.CTA_2,
                    )

                kv_index += 1
                if kv_index == LOAD_STAGES:
                    kv_index = 0
                    kv_phase ^= 1

                if key_block + 1 < iterations:
                    k_stage = kv_index
                    cl.mbarrier_arrive_expect_transaction(
                        kv_arrived.get_element_pointer(k_stage),
                        CLUSTER_SIZE * KV_TILE_ELEMENTS * 2,
                        scope=cl.MbarrierScope.BLOCK,
                    )
                    wait_mbarrier(kv_arrived.get_element_pointer(k_stage), kv_phase)
                    cl.tcgen05_fence_after_thread_sync()
                    for qid in cl.static_iter(range(2)):
                        score_tmem = cl.tcgen05_tmem_offset(
                            tmem_storage[0], column_offset=qid * 256
                        )
                        for chunk in cl.static_iter(range(HEAD_DIM_QK // MMA_K)):
                            cl.tcgen05_mma(
                                cl.Tcgen05MMAKind.F16,
                                score_tmem,
                                qk_descriptor(
                                    q_smem.get_element_pointer((qid, 0)),
                                    BLOCK_M,
                                    chunk,
                                ),
                                qk_descriptor(
                                    kv_smem.get_element_pointer((k_stage, 0)),
                                    BLOCK_N // 2,
                                    chunk,
                                ),
                                qk_instruction,
                                accumulate=chunk != 0,
                                cta_group=cl.CTAGroup.CTA_2,
                            )
                        cl.tcgen05_commit(
                            kv_finished.get_element_pointer(k_stage),
                            multicast_mask=0b11,
                            cta_group=cl.CTAGroup.CTA_2,
                        )
                        cl.tcgen05_commit(
                            scores_arrived.get_element_pointer(qid),
                            multicast_mask=0b11,
                            cta_group=cl.CTAGroup.CTA_2,
                        )
                    if key_block + 2 == iterations:
                        for qid in cl.static_iter(range(2)):
                            cl.tcgen05_commit(
                                q_finished.get_element_pointer(qid),
                                multicast_mask=0b11,
                                cta_group=cl.CTAGroup.CTA_2,
                            )
                    kv_index += 1
                    if kv_index == LOAD_STAGES:
                        kv_index = 0
                        kv_phase ^= 1
                    norm_phase ^= 1
                else:
                    for qid in cl.static_iter(range(2)):
                        cl.tcgen05_commit(
                            tile_arrived.get_element_pointer(qid),
                            multicast_mask=0b11,
                            cta_group=cl.CTAGroup.CTA_2,
                        )
                    norm_phase ^= 1

            current_bid += NUM_SMS
            task_number += 1

    elif warpgroup < 2:
        qid = warpgroup
        score_phase = 0
        rescale_phase = 1
        current_bid = cl.block_index(0)
        while current_bid < total_bids:
            row_sum = cl.float32(0.0)
            row_max = cl.float32(-float("inf"))
            wait_mbarrier(rescale_finished.get_element_pointer(qid), rescale_phase)
            rescale_phase ^= 1

            for key_block in range(iterations):
                wait_mbarrier(scores_arrived.get_element_pointer(qid), score_phase)
                row_tmem = cl.tcgen05_tmem_offset(
                    tmem_storage[0],
                    lane_offset=warp_in_group * WARP_SIZE,
                    column_offset=qid * 256,
                )
                old_max = row_max
                for quarter in cl.static_iter(range(4)):
                    scores = cl.tcgen05_load(
                        cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                        cl.tcgen05_tmem_offset(
                            row_tmem, column_offset=quarter * 32
                        ),
                        count=32,
                    )
                    cl.tcgen05_wait_load()
                    row_max = max_vector32(scores, row_max)

                correction = cl.float32(1.0)
                if key_block > 0:
                    correction_log2 = (old_max - row_max) * cl.float32(SCALE_LOG2)
                    if correction_log2 >= cl.float32(-8.0):
                        row_max = old_max
                    else:
                        correction = fast_exp2(correction_log2)
                    max_vec[qid, lane_in_group] = correction
                cl.barrier_sync_warp()
                if cl.elect_sync():
                    cl.mbarrier_arrive(corr_arrived.get_element_pointer(qid))

                scale = cl.float32(SCALE_LOG2)
                offset = -row_max * scale
                # Reload one quarter at a time until the backend can keep all four
                # live without the 624 stack bytes per thread seen today.
                block_sum = cl.float32(0.0)
                for quarter in cl.static_iter(range(4)):
                    scores = cl.tcgen05_load(
                        cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                        cl.tcgen05_tmem_offset(
                            row_tmem, column_offset=quarter * 32
                        ),
                        count=32,
                    )
                    cl.tcgen05_wait_load()
                    probabilities, quarter_sum = probability_vector(
                        scores, scale, offset
                    )
                    if cl.static_eval(quarter == 0):
                        block_sum = quarter_sum
                    else:
                        block_sum += quarter_sum
                    cl.tcgen05_store(
                        cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                        cl.tcgen05_tmem_offset(
                            row_tmem, column_offset=quarter * 16
                        ),
                        probabilities,
                    )
                    cl.tcgen05_wait_store()
                    if cl.static_eval(quarter == 0):
                        if cl.elect_sync():
                            cl.mbarrier_arrive(
                                cl.map_shared_to_cluster(
                                    norm_scores_arrived.get_element_pointer(qid), 0
                                )
                            )
                    elif cl.elect_sync():
                        cl.mbarrier_arrive(
                            cl.map_shared_to_cluster(
                                norm_quarter_arrived.get_element_pointer(
                                    (quarter - 1, qid)
                                ),
                                0,
                            )
                        )

                wait_mbarrier(rescale_finished.get_element_pointer(qid), rescale_phase)
                rescale_phase ^= 1
                row_sum = row_sum * correction + block_sum
                score_phase ^= 1

            lse_vec[qid, lane_in_group] = row_max
            max_vec[qid, lane_in_group] = row_sum
            cl.barrier_sync_warp()
            if cl.elect_sync():
                cl.mbarrier_arrive(corr_arrived.get_element_pointer(qid))
            current_bid += NUM_SMS

    elif warpgroup == 2:
        correction_phase = 0
        end_phase = 0
        if warp == 8 and cl.elect_sync():
            for qid in cl.static_iter(range(2)):
                cl.mbarrier_arrive(
                    cl.map_shared_to_cluster(
                        norm_scores_arrived.get_element_pointer(qid), 0
                    )
                )

        current_bid = cl.block_index(0)
        while current_bid < total_bids:
            for qid in cl.static_iter(range(2)):
                wait_mbarrier(corr_arrived.get_element_pointer(qid), correction_phase)
                if warp == 8 and cl.elect_sync():
                    cl.mbarrier_arrive(rescale_finished.get_element_pointer(qid))
            correction_phase ^= 1

            for key_block in range(1, iterations):
                for qid in cl.static_iter(range(2)):
                    wait_mbarrier(
                        corr_arrived.get_element_pointer(qid), correction_phase
                    )
                    correction = max_vec[qid, lane_in_group]
                    needs_rescale = cl._nvvm.vote_any_sync(
                        cl.int32(-1), correction < cl.float32(1.0)
                    )
                    if needs_rescale:
                        for column in cl.static_iter(range(0, HEAD_DIM_V, 16)):
                            output_row = cl.tcgen05_tmem_offset(
                                tmem_storage[0],
                                lane_offset=warp_in_group * WARP_SIZE,
                                column_offset=qid * 256 + 128 + column,
                            )
                            values = cl.tcgen05_load(
                                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                                output_row,
                                count=16,
                            )
                            cl.tcgen05_wait_load()
                            cl.tcgen05_store(
                                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                                output_row,
                                scale_vector16(values, correction),
                            )
                        cl.tcgen05_wait_store()
                    cl.barrier_sync_block(
                        number_of_threads=WARPGROUP_SIZE, barrier_id=1
                    )
                    if warp == 8 and cl.elect_sync():
                        cl.mbarrier_arrive(
                            cl.map_shared_to_cluster(
                                norm_scores_arrived.get_element_pointer(qid), 0
                            )
                        )
                        cl.mbarrier_arrive(rescale_finished.get_element_pointer(qid))
                correction_phase ^= 1

            cluster_linear = current_bid // CLUSTER_SIZE
            clusters_m = (q_sequence // BLOCK_M) // 4
            clusters_per_batch = heads * clusters_m
            batch_idx = cluster_linear // clusters_per_batch
            rem = cluster_linear - batch_idx * clusters_per_batch
            head_idx = rem // clusters_m
            m_base = (rem - head_idx * clusters_m) * 4

            for qid in cl.static_iter(range(2)):
                wait_mbarrier(corr_arrived.get_element_pointer(qid), correction_phase)
                row_sum = max_vec[qid, lane_in_group]
                row_max = lse_vec[qid, lane_in_group]
                if warp == 8 and cl.elect_sync():
                    cl.mbarrier_arrive(rescale_finished.get_element_pointer(qid))
                invalid = row_sum == cl.float32(0.0) or row_sum != row_sum
                inv_norm = cl._nvvm.rcp_approx_ftz_f(
                    cl.float32(1.0) if invalid else row_sum
                )
                wait_mbarrier(tile_arrived.get_element_pointer(qid), end_phase)
                cl.copy_async_bulk_wait_group(0, read=True)
                cl.barrier_sync_block(number_of_threads=WARPGROUP_SIZE, barrier_id=1)
                for column in cl.static_iter(range(0, HEAD_DIM_V, 16)):
                    output_row = cl.tcgen05_tmem_offset(
                        tmem_storage[0],
                        lane_offset=warp_in_group * WARP_SIZE,
                        column_offset=qid * 256 + 128 + column,
                    )
                    values = cl.tcgen05_load(
                        cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                        output_row,
                        count=16,
                    )
                    cl.tcgen05_wait_load()
                    store_output_pairs(o_smem, values, inv_norm, lane_in_group, column)
                cl.barrier_sync_block(number_of_threads=WARPGROUP_SIZE, barrier_id=1)
                if warp == 8 and cl.elect_sync():
                    cl.fence_proxy(
                        cl.FenceProxyKind.ASYNC_SHARED,
                        space=cl.MemorySpace.SHARED,
                    )
                    m_tile = m_base + rank * 2 + qid
                    cache_hint = cl.create_fractional_cache_policy(
                        cl.CachePolicy.L2_EVICT_FIRST
                    )
                    cl.copy_async_bulk_tensor_shared_to_global(
                        o_smem.get_base_pointer(),
                        o_tmap,
                        (0, m_tile * BLOCK_M, 0, head_idx, batch_idx),
                        l2_cache_hint=cache_hint,
                    )
                    cl.copy_async_bulk_commit_group()
                    cl.mbarrier_arrive(
                        cl.map_shared_to_cluster(
                            norm_scores_arrived.get_element_pointer(qid), 0
                        )
                    )

                lse_value = cl.float32(-float("inf"))
                if not invalid:
                    scale = cl.float32(SCALE_LOG2)
                    lse_value = (row_max * scale + fast_log2(row_sum)) * cl.float32(LN2)
                m_tile = m_base + rank * 2 + qid
                lse_index = (
                    (batch_idx * heads + head_idx) * q_sequence
                    + m_tile * BLOCK_M
                    + lane_in_group
                )
                lse[lse_index] = lse_value

            correction_phase ^= 1
            end_phase ^= 1
            current_bid += NUM_SMS

    if warp == 8:
        cl.copy_async_bulk_wait_group(0)
    cl.barrier_sync_cluster(aligned=True)
    if warp == 0:
        cl.tcgen05_deallocate(tmem_storage[0], 512, cta_group=cl.CTAGroup.CTA_2)


def make_tma_view(x, segment=64):
    batch, sequence, heads, depth = x.shape
    assert depth % segment == 0
    return torch.as_strided(
        x,
        size=(segment, sequence, depth // segment, heads, batch),
        stride=(1, heads * depth, segment, depth, sequence * heads * depth),
    )


def run_mha(q, k, v):
    batch, q_sequence, heads, qk_dim = q.shape
    kv_sequence = k.shape[1]
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert qk_dim == HEAD_DIM_QK
    assert k.shape == (batch, kv_sequence, heads, HEAD_DIM_QK)
    assert v.shape == (batch, kv_sequence, heads, HEAD_DIM_V)
    assert q_sequence % 512 == 0 and kv_sequence % BLOCK_N == 0
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    output = torch.empty(
        (batch, q_sequence, heads, HEAD_DIM_V), dtype=v.dtype, device=v.device
    )
    lse = torch.empty(
        (batch, heads, 1, q_sequence), dtype=torch.float32, device=q.device
    )
    cl.launch(
        torch.cuda.current_stream(),
        (NUM_SMS,),
        (THREADS,),
        mha_kernel,
        (
            make_tma_view(q),
            make_tma_view(k),
            make_tma_view(v),
            make_tma_view(output),
            lse.reshape(-1),
            batch,
            q_sequence,
            kv_sequence,
            heads,
        ),
        block_in_cluster_count=(CLUSTER_SIZE, 1, 1),
    )
    return output, lse


def reference_mha(q, k, v):
    scale = 1.0 / math.sqrt(HEAD_DIM_QK)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    lse = torch.logsumexp(scores, dim=-1).unsqueeze(2)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, v.float())
    return output, lse


@pytest.mark.parametrize(
    "batch,q_sequence,kv_sequence,heads",
    ((1, 512, 512, 1), (1, 512, 1024, 1), (2, 512, 512, 2)),
)
def test_bf16_b200_mha_noncausal(batch, q_sequence, kv_sequence, heads):
    torch.manual_seed(0)
    q = torch.randn(
        (batch, q_sequence, heads, HEAD_DIM_QK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch, kv_sequence, heads, HEAD_DIM_QK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        (batch, kv_sequence, heads, HEAD_DIM_V),
        device="cuda",
        dtype=torch.bfloat16,
    )

    actual, actual_lse = run_mha(q, k, v)
    torch.cuda.synchronize()
    expected, expected_lse = reference_mha(q, k, v)

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    test_bf16_b200_mha_noncausal(1, 512, 512, 1)
    test_bf16_b200_mha_noncausal(1, 512, 1024, 1)
    test_bf16_b200_mha_noncausal(2, 512, 512, 2)
    print("success")
