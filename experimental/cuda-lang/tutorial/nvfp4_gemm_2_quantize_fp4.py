# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Lang port of ``nvfp4_gemm_2_quantize_fp4.py``.

The kernel performs a CTA_2 NVFP4 GEMM and supports either an FP16 epilogue or
fused per-row, block-16 E2M1 FP4 quantization with E4M3 output scales.  It keeps
the source tutorial's 256x256x256 collective tile, five-stage cluster-TMA
pipeline, eight-warp role assignment, bulk/interleaved scale-factor staging,
CTA_2 MXF4NVF4 MMA, FP32 accumulation, and packed FP4 byte order.

Like the CuTe DSL source, this port uses a CLC dynamic persistent scheduler,
two overlapped TMEM accumulator windows, and a shared-memory TMA output store.
The main shared-memory regions use one 1024-byte-aligned dynamic allocation;
the two CLC response tokens remain statically allocated to match CuTe's CLC
response allocation.
"""

from __future__ import annotations

import argparse
import os

import cuda.lang as cl
import torch


WARP_SIZE = 32
BLOCK_THREADS = 8 * WARP_SIZE
SCHEDULER_WARP = 0
TMA_WARP = 1
MMA_WARP = 2
EPILOGUE_WARP_BASE = 4
EPILOGUE_WARPS = 4

CLUSTER_M = 2
CTA_M = 128
CTA_N = 128
BLOCK_M = CLUSTER_M * CTA_M
BLOCK_N = CLUSTER_M * CTA_N
BLOCK_K = 256
PACKED_BLOCK_K = BLOCK_K // 2
MMA_K = 64
SF_VECTOR_SIZE = 16
SF_K_PER_TILE = BLOCK_K // SF_VECTOR_SIZE
AB_STAGES = 5
TMEM_COLUMNS = 512
SCHEDULER_PIPELINE_STAGES = 2

A_STAGE_BYTES = CTA_M * PACKED_BLOCK_K
B_STAGE_BYTES = CTA_N * PACKED_BLOCK_K
SFA_STAGE_BYTES = CTA_M * SF_K_PER_TILE
SFB_CTA_BYTES = CTA_N * SF_K_PER_TILE
SFB_STAGE_BYTES = CLUSTER_M * SFB_CTA_BYTES
TMA_STAGE_BYTES = (
    A_STAGE_BYTES + B_STAGE_BYTES + SFA_STAGE_BYTES + SFB_STAGE_BYTES
) * CLUSTER_M

SFB_BULK_TMEM_COLUMN = 464
SFA_BULK_TMEM_COLUMN = 496
SFB_INTERLEAVED_TMEM_COLUMN = 500
SFA_INTERLEAVED_TMEM_COLUMN = 508

EPILOGUE_CHUNK_COLUMNS = 64
ACC_WINDOW_STRIDE_COLUMNS = BLOCK_N - EPILOGUE_CHUNK_COLUMNS
OUTPUT_SCALE_BLOCK_N = 16
OUTPUT_SCALE_UNIT_N = 64
OUTPUT_SCALE_UNIT_BYTES = 512
FP4_MAX = 6.0
VEC_BYTES = 32
FP4_SMEM_BYTES_PER_ROW = BLOCK_N // 2
FP4_SMEM_BYTES_PER_WARP = WARP_SIZE * FP4_SMEM_BYTES_PER_ROW
FP4_SMEM_STORE_BYTES = EPILOGUE_WARPS * FP4_SMEM_BYTES_PER_WARP
TMEM_BARRIER_ID = 1
TMEM_DEALLOC_BARRIER_ID = 2
TMEM_BARRIER_THREADS = (EPILOGUE_WARPS + 1) * WARP_SIZE
ACC_EMPTY_THREADS = EPILOGUE_WARPS * WARP_SIZE * CLUSTER_M
SCHEDULER_CONSUMER_WARPS = 3 + EPILOGUE_WARPS
SCHEDULER_CONSUMERS = SCHEDULER_CONSUMER_WARPS * WARP_SIZE * CLUSTER_M
CLC_BYTES = cl.clusterlaunchcontrol_token.bitwidth // 8

_DEFAULT_MNKL = (512, 512, 256, 1)
_DEFAULT_TOLERANCE = 1.0e-1
_DEFAULT_FP4_PAYLOAD_ATOL = 1.05


def _initial_work_tile():
    return (
        cl.block_index(0) // CLUSTER_M,
        cl.block_index(1),
        cl.block_index(2),
    )


def _work_tile_from_clc_token(token):
    has_work = cl.clusterlaunchcontrol_is_canceled(token)
    tile_m = cl.clusterlaunchcontrol_get_first_block_index(token, axis=0) // CLUSTER_M
    tile_n = cl.clusterlaunchcontrol_get_first_block_index(token, axis=1)
    tile_l = cl.clusterlaunchcontrol_get_first_block_index(token, axis=2)
    return tile_m, tile_n, tile_l, has_work


def _consume_clc_response(clc_barriers, clc_tokens, iteration):
    slot = iteration % SCHEDULER_PIPELINE_STAGES
    phase = (iteration // SCHEDULER_PIPELINE_STAGES) & 1
    cl.mbarrier_wait_parity(
        clc_barriers.get_element_pointer(slot),
        phase,
        scope=cl.MbarrierScope.CLUSTER,
    )
    token = clc_tokens.get_element_pointer(slot).load()
    tile_m, tile_n, tile_l, has_work = _work_tile_from_clc_token(token)
    cl.fence_proxy(
        cl.FenceProxyKind.ASYNC_SHARED,
        space=cl.MemorySpace.SHARED,
    )
    return tile_m, tile_n, tile_l, has_work


def _release_clc_response(scheduler_consumed, iteration, count=1):
    slot = iteration % SCHEDULER_PIPELINE_STAGES
    leader_barrier = cl.map_shared_to_cluster(
        scheduler_consumed.get_element_pointer(slot), 0
    )
    cl.mbarrier_arrive(
        leader_barrier,
        count=count,
        scope=cl.MbarrierScope.BLOCK,
    )


def _as_float32_vector_64(regs):
    return cl.Vector(
        *tuple(cl.bitcast(regs[i], cl.float32) for i in cl.static_iter(range(64))),
        dtype=cl.float32,
    )


def _to_float16_vector(values, base, count):
    return cl.Vector(
        *tuple(values[base + i] for i in cl.static_iter(range(count))),
        dtype=cl.float32,
    ).astype(cl.float16)


def _pack_fp32x8_to_e2m1x8(values, base):
    """Pack eight normalized FP32 values into four E2M1 bytes."""
    mask = cl.uint32(0xFF)
    byte0 = cl.uint32(
        cl.uint16(cl._nvvm.ff_to_e2m1x2_rn_satfinite(values[base + 1], values[base]))
    )
    byte1 = cl.uint32(
        cl.uint16(
            cl._nvvm.ff_to_e2m1x2_rn_satfinite(values[base + 3], values[base + 2])
        )
    )
    byte2 = cl.uint32(
        cl.uint16(
            cl._nvvm.ff_to_e2m1x2_rn_satfinite(values[base + 5], values[base + 4])
        )
    )
    byte3 = cl.uint32(
        cl.uint16(
            cl._nvvm.ff_to_e2m1x2_rn_satfinite(values[base + 7], values[base + 6])
        )
    )
    return cl.uint32(
        (byte0 & mask)
        | ((byte1 & mask) << 8)
        | ((byte2 & mask) << 16)
        | ((byte3 & mask) << 24)
    )


def _quantize_fp4_block(values, base, alpha):
    scaled = cl.Vector(
        *tuple(
            values[base + i] * alpha
            for i in cl.static_iter(range(OUTPUT_SCALE_BLOCK_N))
        ),
        dtype=cl.float32,
    )
    absolute = cl.abs(scaled)
    reduce8 = tuple(
        cl.maximum(absolute[2 * i], absolute[2 * i + 1])
        for i in cl.static_iter(range(8))
    )
    reduce4 = tuple(
        cl.maximum(reduce8[2 * i], reduce8[2 * i + 1])
        for i in cl.static_iter(range(4))
    )
    reduce2 = tuple(
        cl.maximum(reduce4[2 * i], reduce4[2 * i + 1])
        for i in cl.static_iter(range(2))
    )
    amax = cl.maximum(reduce2[0], reduce2[1])
    inv_scale = cl.float32(FP4_MAX) / amax
    normalized = scaled * inv_scale
    scale = cl.truediv(cl.float32(1.0), inv_scale)
    return (
        scale,
        _pack_fp32x8_to_e2m1x8(normalized, 0),
        _pack_fp32x8_to_e2m1x8(normalized, 8),
    )


def _swizzle_128b(byte_offset):
    """Apply CuTe's Swizzle<3, 4, 3> to a byte address."""
    return byte_offset ^ ((byte_offset >> 3) & 0x70)


def _pack_e4m3_scales(scales):
    lo = cl.uint32(cl.uint16(cl._nvvm.ff_to_e4m3x2_rn(scales[1], scales[0])))
    hi = cl.uint32(cl.uint16(cl._nvvm.ff_to_e4m3x2_rn(scales[3], scales[2])))
    return cl.Vector(
        cl.uint8(lo & cl.uint32(0xFF)),
        cl.uint8((lo >> 8) & cl.uint32(0xFF)),
        cl.uint8(hi & cl.uint32(0xFF)),
        cl.uint8((hi >> 8) & cl.uint32(0xFF)),
        dtype=cl.uint8,
    )


@cl.kernel
def _kernel(
    a,
    b,
    sfa,
    sfb,
    c,
    c_scale,
    alpha,
    k: cl.Constant[int],
    output_fp4: cl.Constant[bool],
    scale_bulk_copy: cl.Constant[bool],
):
    """CTA_2 NVFP4 GEMM with FP16 or fused FP4 output."""
    cl.static_assert(k % BLOCK_K == 0, "K must be divisible by 256")
    packed_k, m, batch_count = a.shape
    n = b.shape[1]
    tid = cl.thread_index(0)
    warp = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    rank = cl.block_in_cluster_index(0)
    is_leader = rank == 0

    a_tmap = cl.tensor_map_tiled(
        a,
        (PACKED_BLOCK_K, CTA_M, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    b_tmap = cl.tensor_map_tiled(
        b,
        (PACKED_BLOCK_K, CTA_N, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    sfa_tmap = cl.tensor_map_tiled(sfa, (256, 4, 1, 1))
    sfb_tmap = cl.tensor_map_tiled(sfb, (256, 4, 1, 1))
    if output_fp4:
        c_tmap = cl.tensor_map_tiled(
            c,
            (FP4_SMEM_BYTES_PER_ROW, WARP_SIZE, 1),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )

    ab_full = cl.shared_array(AB_STAGES, cl.mbarrier, alignment=8, dynamic=True)
    ab_empty = cl.shared_array(AB_STAGES, cl.mbarrier, alignment=8, dynamic=True)
    acc_full = cl.shared_array(1, cl.mbarrier, alignment=8, dynamic=True)
    acc_empty = cl.shared_array(1, cl.mbarrier, alignment=8, dynamic=True)
    tmem_dealloc = cl.shared_array(1, cl.mbarrier, alignment=8, dynamic=True)
    clc_barriers = cl.shared_array(
        SCHEDULER_PIPELINE_STAGES,
        cl.mbarrier,
        alignment=8,
        dynamic=True,
    )
    scheduler_consumed = cl.shared_array(
        SCHEDULER_PIPELINE_STAGES,
        cl.mbarrier,
        alignment=8,
        dynamic=True,
    )
    clc_tokens = cl.shared_array(
        SCHEDULER_PIPELINE_STAGES,
        cl.clusterlaunchcontrol_token,
        alignment=16,
    )
    tmem_storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
        alignment=4,
        dynamic=True,
    )
    a_smem = cl.shared_array(
        (AB_STAGES, A_STAGE_BYTES),
        cl.uint8,
        alignment=128,
        dynamic=True,
    )
    b_smem = cl.shared_array(
        (AB_STAGES, B_STAGE_BYTES),
        cl.uint8,
        alignment=128,
        dynamic=True,
    )
    sfa_smem = cl.shared_array(
        (AB_STAGES, SFA_STAGE_BYTES),
        cl.uint8,
        alignment=128,
        dynamic=True,
    )
    sfb_smem = cl.shared_array(
        (AB_STAGES, SFB_STAGE_BYTES),
        cl.uint8,
        alignment=128,
        dynamic=True,
    )
    if output_fp4:
        c_smem = cl.shared_array(
            FP4_SMEM_STORE_BYTES,
            cl.uint8,
            alignment=128,
            dynamic=True,
        )

    if warp == SCHEDULER_WARP and lane < SCHEDULER_PIPELINE_STAGES:
        cl.mbarrier_initialize(clc_barriers.get_element_pointer(lane), 1)
        cl.mbarrier_initialize(
            scheduler_consumed.get_element_pointer(lane),
            SCHEDULER_CONSUMERS,
        )
    if warp == SCHEDULER_WARP and cl.elect_sync():
        for stage in cl.static_iter(range(AB_STAGES)):
            cl.mbarrier_initialize(ab_full.get_element_pointer(stage), 1)
            cl.mbarrier_initialize(ab_empty.get_element_pointer(stage), 1)
        cl.mbarrier_initialize(acc_full.get_base_pointer(), 1)
        cl.mbarrier_initialize(acc_empty.get_base_pointer(), ACC_EMPTY_THREADS)
        cl.mbarrier_initialize(tmem_dealloc.get_base_pointer(), WARP_SIZE)
    cl.fence_mbarrier_initialize()
    cl.barrier_arrive_cluster(aligned=False, memory_order=cl.MemoryOrder.RELAXED)
    cl.barrier_wait_cluster(aligned=False)

    if warp < EPILOGUE_WARP_BASE:
        cl.setmaxregister_decrease(24)

    if warp == MMA_WARP:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(),
            TMEM_COLUMNS,
            cta_group=cl.CTAGroup.CTA_2,
        )
        cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_2)
    if warp == MMA_WARP or warp >= EPILOGUE_WARP_BASE:
        cl.barrier_sync_block(
            number_of_threads=TMEM_BARRIER_THREADS,
            barrier_id=TMEM_BARRIER_ID,
        )

    k_tiles = k // BLOCK_K

    if warp == SCHEDULER_WARP:
        scheduler_has_work = True
        scheduler_iteration = cl.int32(0)
        while scheduler_has_work:
            slot = scheduler_iteration % SCHEDULER_PIPELINE_STAGES
            ring_phase = (scheduler_iteration // SCHEDULER_PIPELINE_STAGES) & 1
            if is_leader and scheduler_iteration >= SCHEDULER_PIPELINE_STAGES:
                cl.mbarrier_wait_parity(
                    scheduler_consumed.get_element_pointer(slot),
                    ring_phase ^ 1,
                )
            clc_barrier = clc_barriers.get_element_pointer(slot)
            clc_token = clc_tokens.get_element_pointer(slot)
            if is_leader and cl.elect_sync():
                cl.clusterlaunchcontrol_try_cancel(
                    clc_token, clc_barrier, multicast=True
                )
            if cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    clc_barrier,
                    CLC_BYTES,
                    scope=cl.MbarrierScope.CLUSTER,
                    memory_order=cl.MemoryOrder.RELAXED,
                )
            cl.mbarrier_wait_parity(
                clc_barrier,
                ring_phase,
                scope=cl.MbarrierScope.CLUSTER,
            )
            token = clc_token.load()
            tile_m, tile_n, tile_l, scheduler_has_work = _work_tile_from_clc_token(
                token
            )
            cl.fence_proxy(
                cl.FenceProxyKind.ASYNC_SHARED,
                space=cl.MemorySpace.SHARED,
            )
            _release_clc_response(scheduler_consumed, scheduler_iteration)
            scheduler_iteration += 1

    elif warp == TMA_WARP:
        local_mask = cl.int16(1 << rank)
        sfb_mask = cl.int16(0b11)
        leader_barrier_base = cl.map_shared_to_leader_block(ab_full.get_base_pointer())
        tile_m, tile_n, tile_l = _initial_work_tile()
        has_work = True
        ab_tma_iteration = cl.int32(0)
        tile_iteration = cl.int32(0)

        while has_work:
            coord_m = tile_m * BLOCK_M + rank * CTA_M
            coord_n_b = tile_n * BLOCK_N + rank * CTA_N
            scale_block_m = tile_m * CLUSTER_M + rank
            scale_block_n = tile_n * CLUSTER_M + rank
            for k_tile in range(k_tiles):
                stage = ab_tma_iteration % AB_STAGES
                empty_phase = ((ab_tma_iteration // AB_STAGES) & 1) ^ 1

                empty_bar = ab_empty.get_element_pointer(stage)
                full_bar = ab_full.get_element_pointer(stage)
                # Match CuTe: re-elect for each divergent collective region
                # instead of carrying one elected lane across K iterations.
                if cl.elect_sync():
                    cl.mbarrier_wait_parity(empty_bar, empty_phase)
                    if is_leader:
                        cl.mbarrier_arrive_expect_transaction(
                            full_bar,
                            TMA_STAGE_BYTES,
                            scope=cl.MbarrierScope.BLOCK,
                        )

                arrive_bar = leader_barrier_base + stage
                a_stage = cl.address_space_cast(
                    a_smem.get_element_pointer((stage, 0)),
                    cl.MemorySpace.SHARED_CLUSTER,
                )
                b_stage = cl.address_space_cast(
                    b_smem.get_element_pointer((stage, 0)),
                    cl.MemorySpace.SHARED_CLUSTER,
                )
                sfa_stage = cl.address_space_cast(
                    sfa_smem.get_element_pointer((stage, 0)),
                    cl.MemorySpace.SHARED_CLUSTER,
                )
                sfb_stage = cl.address_space_cast(
                    sfb_smem.get_element_pointer((stage, rank * SFB_CTA_BYTES)),
                    cl.MemorySpace.SHARED_CLUSTER,
                )
                packed_k_coord = k_tile * PACKED_BLOCK_K
                sf_k_atom = k_tile * 4

                if cl.elect_sync():
                    cl.copy_async_bulk_tensor_global_to_shared(
                        a_tmap,
                        (packed_k_coord, coord_m, tile_l),
                        a_stage,
                        arrive_bar,
                        multicast_mask=local_mask,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        b_tmap,
                        (packed_k_coord, coord_n_b, tile_l),
                        b_stage,
                        arrive_bar,
                        multicast_mask=local_mask,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        sfa_tmap,
                        (0, sf_k_atom, scale_block_m, tile_l),
                        sfa_stage,
                        arrive_bar,
                        multicast_mask=local_mask,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        sfb_tmap,
                        (0, sf_k_atom, scale_block_n, tile_l),
                        sfb_stage,
                        arrive_bar,
                        multicast_mask=sfb_mask,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                ab_tma_iteration += 1

            tile_m, tile_n, tile_l, has_work = _consume_clc_response(
                clc_barriers,
                clc_tokens,
                tile_iteration,
            )
            _release_clc_response(
                scheduler_consumed,
                tile_iteration,
            )
            tile_iteration += 1

    elif warp == MMA_WARP and cl.elect_sync():
        tmem_base = tmem_storage[0]
        instruction = cl.Tcgen05Mxf4InstructionDescriptor(
            a_type=cl.Tcgen05Mxf4InstructionDescriptor.Type.E2M1,
            b_type=cl.Tcgen05Mxf4InstructionDescriptor.Type.E2M1,
            scale_format=cl.Tcgen05Mxf4InstructionDescriptor.ScaleFormat.UE4M3,
            n=BLOCK_N,
            m=BLOCK_M,
        ).encode()
        tile_m, tile_n, tile_l = _initial_work_tile()
        has_work = True
        ab_mma_iteration = cl.int32(0)
        tile_iteration = cl.int32(0)

        while has_work:
            if is_leader:
                # Match CuTe's initial phase-1 token. The initialized barrier
                # makes the first accumulator window immediately available;
                # each epilogue arrival releases the next alternating window.
                acc_empty_phase = (tile_iteration + 1) & 1
                cl.mbarrier_wait_parity(
                    acc_empty.get_base_pointer(), acc_empty_phase
                )
                acc_tmem = cl.tcgen05_tmem_offset(
                    tmem_base,
                    column_offset=((tile_iteration & 1) * ACC_WINDOW_STRIDE_COLUMNS),
                )
                accumulate = False

                for k_tile in range(k_tiles):
                    stage = ab_mma_iteration % AB_STAGES
                    full_phase = (ab_mma_iteration // AB_STAGES) & 1

                    full_bar = ab_full.get_element_pointer(stage)
                    empty_bar = ab_empty.get_element_pointer(stage)
                    cl.mbarrier_wait_parity(full_bar, full_phase)

                    a_stage = a_smem.get_element_pointer((stage, 0))
                    b_stage = b_smem.get_element_pointer((stage, 0))
                    sfa_stage = sfa_smem.get_element_pointer((stage, 0))
                    sfb_stage = sfb_smem.get_element_pointer((stage, 0))
                    sfa_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=sfa_stage,
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_NONE,
                    ).encode()
                    sfb_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=sfb_stage,
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_NONE,
                    ).encode()
                    a_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=a_stage,
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=8 * 128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                    ).encode()
                    b_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=b_stage,
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=8 * 128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                    ).encode()

                    if scale_bulk_copy:
                        for copy_idx in cl.static_iter(range(4)):
                            cl.tcgen05_copy(
                                cl.tcgen05_tmem_offset(
                                    tmem_base,
                                    column_offset=(SFA_BULK_TMEM_COLUMN + copy_idx * 4),
                                ),
                                cl.int64(sfa_desc + 32 * copy_idx),
                                shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                                cta_group=cl.CTAGroup.CTA_2,
                                multicast=cl.Tcgen05CopyMulticast.WARPX4,
                            )
                        for copy_idx in cl.static_iter(range(8)):
                            smem_increment = 32 * (copy_idx // 2) + 128 * (copy_idx % 2)
                            cl.tcgen05_copy(
                                cl.tcgen05_tmem_offset(
                                    tmem_base,
                                    column_offset=(SFB_BULK_TMEM_COLUMN + copy_idx * 4),
                                ),
                                cl.int64(sfb_desc + smem_increment),
                                shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                                cta_group=cl.CTAGroup.CTA_2,
                                multicast=cl.Tcgen05CopyMulticast.WARPX4,
                            )

                    for k_block in cl.static_iter(range(BLOCK_K // MMA_K)):
                        if scale_bulk_copy:
                            scale_a = cl.tcgen05_tmem_offset(
                                tmem_base,
                                column_offset=(SFA_BULK_TMEM_COLUMN + k_block * 4),
                            )
                            scale_b = cl.tcgen05_tmem_offset(
                                tmem_base,
                                column_offset=(SFB_BULK_TMEM_COLUMN + k_block * 8),
                            )
                        else:
                            scale_a = cl.tcgen05_tmem_offset(
                                tmem_base,
                                column_offset=SFA_INTERLEAVED_TMEM_COLUMN,
                            )
                            scale_b = cl.tcgen05_tmem_offset(
                                tmem_base,
                                column_offset=SFB_INTERLEAVED_TMEM_COLUMN,
                            )
                            cl.tcgen05_copy(
                                scale_a,
                                cl.int64(sfa_desc + 32 * k_block),
                                shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                                cta_group=cl.CTAGroup.CTA_2,
                                multicast=cl.Tcgen05CopyMulticast.WARPX4,
                            )
                            for sfb_block in cl.static_iter(range(2)):
                                cl.tcgen05_copy(
                                    cl.tcgen05_tmem_offset(
                                        tmem_base,
                                        column_offset=(
                                            SFB_INTERLEAVED_TMEM_COLUMN + sfb_block * 4
                                        ),
                                    ),
                                    cl.int64(sfb_desc + 32 * k_block + 128 * sfb_block),
                                    shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                                    cta_group=cl.CTAGroup.CTA_2,
                                    multicast=cl.Tcgen05CopyMulticast.WARPX4,
                                )

                        cl.tcgen05_mma_block_scale(
                            cl.Tcgen05MMABlockScaleKind.MXF4NVF4,
                            acc_tmem,
                            cl.int64(a_desc + 2 * k_block),
                            cl.int64(b_desc + 2 * k_block),
                            cl.int32(instruction),
                            scale_a,
                            scale_b,
                            accumulate=accumulate,
                            cta_group=cl.CTAGroup.CTA_2,
                            scale_vector_size=(cl.Tcgen05MMAScaleVectorSize.BLOCK_16),
                        )
                        accumulate = True

                    cl.tcgen05_commit(
                        empty_bar,
                        multicast_mask=0b11,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                    ab_mma_iteration += 1

                cl.tcgen05_commit(
                    acc_full.get_base_pointer(),
                    multicast_mask=0b11,
                    cta_group=cl.CTAGroup.CTA_2,
                )

            tile_m, tile_n, tile_l, has_work = _consume_clc_response(
                clc_barriers,
                clc_tokens,
                tile_iteration,
            )
            _release_clc_response(
                scheduler_consumed,
                tile_iteration,
                count=WARP_SIZE,
            )
            tile_iteration += 1

    elif warp >= EPILOGUE_WARP_BASE:
        cl.setmaxregister_increase(232)
        tmem_base = tmem_storage[0]
        epi_warp = warp - EPILOGUE_WARP_BASE
        tile_m, tile_n, tile_l = _initial_work_tile()
        has_work = True
        tile_iteration = cl.int32(0)
        acc_full_phase = cl.int32(0)

        while has_work:
            cl.mbarrier_wait_parity(
                acc_full.get_base_pointer(), acc_full_phase
            )
            acc_full_phase ^= 1
            acc_window = tile_iteration & 1
            coord_m = tile_m * BLOCK_M + rank * CTA_M
            coord_n_c = tile_n * BLOCK_N
            row = coord_m + epi_warp * WARP_SIZE + lane

            for block_idx in range(cl.int32(BLOCK_N // EPILOGUE_CHUNK_COLUMNS)):
                chunk = (1 - acc_window) * (
                    BLOCK_N // EPILOGUE_CHUNK_COLUMNS - 1 - block_idx
                ) + acc_window * block_idx
                column = chunk * EPILOGUE_CHUNK_COLUMNS
                tmem = cl.tcgen05_tmem_offset(
                    tmem_base,
                    lane_offset=epi_warp * WARP_SIZE,
                    column_offset=(acc_window * ACC_WINDOW_STRIDE_COLUMNS + column),
                )
                regs = cl.tcgen05_load(
                    cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                    tmem,
                    count=EPILOGUE_CHUNK_COLUMNS,
                )
                if block_idx == 0:
                    cl.tcgen05_wait_load()
                    leader_acc_empty = cl.map_shared_to_cluster(
                        acc_empty.get_base_pointer(), 0
                    )
                    cl.mbarrier_arrive(leader_acc_empty, scope=cl.MbarrierScope.BLOCK)
                accumulators = _as_float32_vector_64(regs)

                if output_fp4:
                    alpha_value = cl.float32(alpha[tile_l])
                    scale0, word0, word1 = _quantize_fp4_block(
                        accumulators, 0, alpha_value
                    )
                    scale1, word2, word3 = _quantize_fp4_block(
                        accumulators, 16, alpha_value
                    )
                    scale2, word4, word5 = _quantize_fp4_block(
                        accumulators, 32, alpha_value
                    )
                    scale3, word6, word7 = _quantize_fp4_block(
                        accumulators, 48, alpha_value
                    )
                    words = (
                        word0,
                        word1,
                        word2,
                        word3,
                        word4,
                        word5,
                        word6,
                        word7,
                    )
                    for store_idx in cl.static_iter(range(2)):
                        logical_byte_offset = (
                            epi_warp * FP4_SMEM_BYTES_PER_WARP
                            + lane * FP4_SMEM_BYTES_PER_ROW
                            + chunk * (EPILOGUE_CHUNK_COLUMNS // 2)
                            + store_idx * 16
                        )
                        swizzled_byte_offset = _swizzle_128b(logical_byte_offset)
                        store_ptr = cl.bitcast(
                            c_smem.get_base_pointer() + swizzled_byte_offset,
                            cl.pointer_dtype(cl.uint32, cl.MemorySpace.SHARED),
                        )
                        store_ptr.store(
                            cl.Vector(
                                *tuple(
                                    words[store_idx * 4 + i]
                                    for i in cl.static_iter(range(4))
                                ),
                                dtype=cl.uint32,
                            ),
                            alignment=16,
                        )

                    scales = cl.Vector(scale0, scale1, scale2, scale3, dtype=cl.float32)
                    packed_scales = _pack_e4m3_scales(scales)
                    scale_units_n = n // OUTPUT_SCALE_UNIT_N
                    scale_units_m = m // CTA_M
                    unit_m = row // CTA_M
                    unit_n = (coord_n_c + column) // OUTPUT_SCALE_UNIT_N
                    row_in_unit = row % CTA_M
                    scale_offset = (
                        tile_l * scale_units_m * scale_units_n * OUTPUT_SCALE_UNIT_BYTES
                        + (unit_m * scale_units_n + unit_n) * OUTPUT_SCALE_UNIT_BYTES
                        + (row_in_unit % 32) * 16
                        + (row_in_unit // 32) * 4
                    )
                    (c_scale.get_base_pointer() + scale_offset).store(
                        packed_scales, alignment=4
                    )
                else:
                    vsize = VEC_BYTES // 2
                    for vector_idx in cl.static_iter(
                        range(EPILOGUE_CHUNK_COLUMNS // vsize)
                    ):
                        col = coord_n_c + column + vector_idx * vsize
                        values = _to_float16_vector(
                            accumulators, vector_idx * vsize, vsize
                        )
                        c.get_element_pointer((row, col, tile_l)).store(
                            values, alignment=VEC_BYTES
                        )

            if output_fp4:
                cl.fence_proxy(
                    cl.FenceProxyKind.ASYNC_SHARED,
                    space=cl.MemorySpace.SHARED,
                )
                if cl.elect_sync():
                    cl.copy_async_bulk_tensor_shared_to_global(
                        c_smem.get_base_pointer() + epi_warp * FP4_SMEM_BYTES_PER_WARP,
                        c_tmap,
                        (
                            coord_n_c // 2,
                            coord_m + epi_warp * WARP_SIZE,
                            tile_l,
                        ),
                    )
                cl.copy_async_bulk_commit_group()
                cl.copy_async_bulk_wait_group(0, read=True)

            tile_m, tile_n, tile_l, has_work = _consume_clc_response(
                clc_barriers,
                clc_tokens,
                tile_iteration,
            )
            _release_clc_response(scheduler_consumed, tile_iteration)
            tile_iteration += 1
        cl.tcgen05_fence_before_thread_sync()

    if warp == MMA_WARP or warp >= EPILOGUE_WARP_BASE:
        cl.barrier_sync_block(
            number_of_threads=TMEM_BARRIER_THREADS,
            barrier_id=TMEM_DEALLOC_BARRIER_ID,
        )
    if warp == MMA_WARP:
        peer_dealloc = cl.map_shared_to_cluster(
            tmem_dealloc.get_base_pointer(), rank ^ 1
        )
        cl.mbarrier_arrive(peer_dealloc, scope=cl.MbarrierScope.BLOCK)
        cl.mbarrier_wait_parity(tmem_dealloc.get_base_pointer(), 0)
        cl.tcgen05_deallocate(
            tmem_storage[0], TMEM_COLUMNS, cta_group=cl.CTAGroup.CTA_2
        )


def _parse_mnkl(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Expected exactly four MNKL values.")
    return values


def _validate_mnkl(mnkl: tuple[int, int, int, int]) -> None:
    if len(mnkl) != 4:
        raise ValueError("MNKL must contain exactly four values")
    m, n, k, batch = mnkl
    if min(m, n, k, batch) <= 0:
        raise ValueError("MNKL values must be positive")
    if m % BLOCK_M:
        raise ValueError(f"M must be a multiple of {BLOCK_M} (got {m})")
    if n % BLOCK_N:
        raise ValueError(f"N must be a multiple of {BLOCK_N} (got {n})")
    if k % BLOCK_K:
        raise ValueError(f"K must be a multiple of {BLOCK_K} (got {k})")


def to_blocked(scale: torch.Tensor) -> torch.Tensor:
    rows, columns = scale.shape
    if rows % 128 or columns % 4:
        raise ValueError("scale rows and columns must be multiples of 128 and 4")
    blocks = scale.view(rows // 128, 128, columns // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _make_scale_tensors(
    batch: int, rows: int, sf_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = (
        torch.randint(1, 3, (batch, rows, sf_k), dtype=torch.int8)
        .to(torch.float8_e4m3fn)
        .permute(1, 2, 0)
    )
    blocked = torch.stack(
        [to_blocked(reference[:, :, index]) for index in range(batch)]
    ).contiguous()
    tma = (
        blocked.view(torch.uint16)
        .reshape(batch, rows // 128, sf_k // 4, 256)
        .permute(3, 2, 1, 0)
        .cuda()
    )
    return reference, tma


def prepare_tensors(
    m: int,
    n: int,
    k: int,
    batch: int = 1,
    out_dtype: str = "fp4",
    **_,
) -> dict[str, torch.Tensor]:
    _validate_mnkl((m, n, k, batch))
    if out_dtype not in {"fp16", "fp4"}:
        raise ValueError("out_dtype must be 'fp16' or 'fp4'")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("this PyTorch build does not provide float4_e2m1fn_x2")

    torch.manual_seed(1111)
    a_storage = torch.randint(
        0, 2, (batch, m, k // 2), dtype=torch.uint8, device="cuda"
    )
    b_storage = torch.randint(
        0, 2, (batch, n, k // 2), dtype=torch.uint8, device="cuda"
    )
    a = a_storage.permute(2, 1, 0)
    b = b_storage.permute(2, 1, 0)
    a_ref = a_storage.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)
    b_ref = b_storage.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)

    sf_k = cl.cdiv(k, SF_VECTOR_SIZE)
    sfa_ref, sfa = _make_scale_tensors(batch, m, sf_k)
    sfb_ref, sfb = _make_scale_tensors(batch, n, sf_k)
    alpha = torch.randn((batch,), dtype=torch.float32, device="cuda")

    if out_dtype == "fp4":
        c_storage = torch.empty((batch, m, n // 2), dtype=torch.uint8, device="cuda")
        c = c_storage.permute(2, 1, 0)
        c_scale = torch.empty(
            batch * (m // CTA_M) * (n // OUTPUT_SCALE_UNIT_N) * OUTPUT_SCALE_UNIT_BYTES,
            dtype=torch.uint8,
            device="cuda",
        )
    else:
        c_storage = torch.empty((batch, m, n), dtype=torch.float16, device="cuda")
        c = c_storage.permute(1, 2, 0)
        c_scale = torch.empty((1,), dtype=torch.uint8, device="cuda")

    return {
        "a": a,
        "b": b,
        "a_ref": a_ref,
        "b_ref": b_ref,
        "sfa": sfa,
        "sfb": sfb,
        "sfa_ref": sfa_ref,
        "sfb_ref": sfb_ref,
        "c": c,
        "c_storage": c_storage,
        "c_scale": c_scale,
        "alpha": alpha,
    }


def run(
    tensors: dict[str, torch.Tensor],
    out_dtype: str = "fp4",
    scale_bulk_copy: bool = False,
    stream=None,
) -> None:
    a, b = tensors["a"], tensors["b"]
    sfa, sfb, c = tensors["sfa"], tensors["sfb"], tensors["c"]
    c_scale, alpha = tensors["c_scale"], tensors["alpha"]
    packed_k, m, batch = a.shape
    n = b.shape[1]
    k = packed_k * 2
    _validate_mnkl((m, n, k, batch))

    cuda_stream = torch.cuda.current_stream() if stream is None else stream
    cl.launch(
        cuda_stream,
        (m // CTA_M, n // BLOCK_N, batch),
        (BLOCK_THREADS, 1, 1),
        _kernel,
        (
            a,
            b,
            sfa,
            sfb,
            c,
            c_scale,
            alpha,
            k,
            out_dtype == "fp4",
            scale_bulk_copy,
        ),
        block_in_cluster_count=(CLUSTER_M, 1, 1),
    )


def _reference_accumulator(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    a_ref, b_ref = tensors["a_ref"], tensors["b_ref"]
    sfa_ref, sfb_ref = tensors["sfa_ref"], tensors["sfb_ref"]
    m, _, batch = a_ref.shape
    n = b_ref.shape[0]
    reference = torch.empty((batch, m, n), dtype=torch.float32, device="cuda")
    for batch_idx in range(batch):
        a_values = _unpack_e2m1_bytes_to_float(
            a_ref[:, :, batch_idx].view(torch.uint8)
        )
        b_values = _unpack_e2m1_bytes_to_float(
            b_ref[:, :, batch_idx].view(torch.uint8)
        )
        scale_a = sfa_ref[:, :, batch_idx].to(
            device=a_values.device, dtype=torch.float32
        ).repeat_interleave(SF_VECTOR_SIZE, dim=1)
        scale_b = sfb_ref[:, :, batch_idx].to(
            device=b_values.device, dtype=torch.float32
        ).repeat_interleave(SF_VECTOR_SIZE, dim=1)
        a_values *= scale_a
        b_values *= scale_b
        reference[batch_idx] = a_values @ b_values.T.contiguous()
    return reference


def _fp4_scale_reference(scaled_reference: torch.Tensor) -> torch.Tensor:
    batch, m, n = scaled_reference.shape
    blocks = scaled_reference.float().view(batch, m, n // 16, 16)
    plain = (blocks.abs().amax(dim=-1) / FP4_MAX).to(torch.float8_e4m3fn)
    scale_ref = torch.empty(
        (batch, m // CTA_M, n // OUTPUT_SCALE_UNIT_N, OUTPUT_SCALE_UNIT_BYTES),
        dtype=torch.float8_e4m3fn,
        device=scaled_reference.device,
    )
    scale_blocks = plain.view(batch, m // CTA_M, CTA_M, n // OUTPUT_SCALE_UNIT_N, 4)
    for row_mod in range(CTA_M):
        row_offset = (row_mod % 32) * 16 + (row_mod // 32) * 4
        scale_ref[:, :, :, row_offset: row_offset + 4] = scale_blocks[
            :, :, row_mod, :, :
        ]
    return scale_ref.flatten().view(torch.uint8)


def _normalized_fp4_reference_values(
    scaled_reference: torch.Tensor,
) -> torch.Tensor:
    batch, m, n = scaled_reference.shape
    blocks = scaled_reference.float().view(batch, m, n // 16, 16)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    inv_scale = torch.where(amax == 0, torch.zeros_like(amax), FP4_MAX / amax)
    normalized = (blocks * inv_scale).reshape(batch, m, n)
    return torch.nan_to_num(normalized, nan=0.0, posinf=FP4_MAX, neginf=-FP4_MAX).clamp(
        -FP4_MAX, FP4_MAX
    )


def _unpack_e2m1_bytes_to_float(fp4_bytes: torch.Tensor) -> torch.Tensor:
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


def verify_output(
    tensors: dict[str, torch.Tensor],
    out_dtype: str = "fp4",
    tolerance: float = _DEFAULT_TOLERANCE,
    fp4_payload_atol: float = _DEFAULT_FP4_PAYLOAD_ATOL,
) -> None:
    reference_acc = _reference_accumulator(tensors)
    if out_dtype == "fp16":
        torch.testing.assert_close(
            tensors["c_storage"],
            reference_acc.to(torch.float16),
            atol=tolerance,
            rtol=1.0e-2,
        )
        return

    scaled_reference = reference_acc * tensors["alpha"].view(-1, 1, 1)
    normalized = _normalized_fp4_reference_values(scaled_reference)
    actual_values = _unpack_e2m1_bytes_to_float(tensors["c_storage"])
    torch.testing.assert_close(
        actual_values,
        normalized,
        atol=fp4_payload_atol,
        rtol=0.0,
    )
    torch.testing.assert_close(
        tensors["c_scale"],
        _fp4_scale_reference(scaled_reference),
        atol=0,
        rtol=0,
    )


def run_nvfp4_gemm(
    mnkl: tuple[int, int, int, int] = _DEFAULT_MNKL,
    tolerance: float = _DEFAULT_TOLERANCE,
    warmup_iters: int = 0,
    iterations: int = 0,
    scale_bulk_copy: bool = False,
    out_dtype: str = "fp4",
    fp4_payload_atol: float = _DEFAULT_FP4_PAYLOAD_ATOL,
    keep_ptx: bool = False,
) -> None:
    _validate_mnkl(mnkl)
    if warmup_iters < 0 or iterations < 0:
        raise ValueError("warmup_iters and iterations must be non-negative")
    if keep_ptx:
        # CUDA Lang prints the retained PTX through its compiler log channel.
        os.environ["CUDA_LANG_LOGS"] = "PTX"
    m, n, k, batch = mnkl
    tensors = prepare_tensors(m=m, n=n, k=k, batch=batch, out_dtype=out_dtype)
    run(tensors, out_dtype=out_dtype, scale_bulk_copy=scale_bulk_copy)
    torch.cuda.synchronize()
    print(
        f"Run kernel (mnkl={mnkl}, out_dtype={out_dtype}, "
        f"scale_bulk_copy={scale_bulk_copy}) OK",
        flush=True,
    )
    verify_output(
        tensors,
        out_dtype=out_dtype,
        tolerance=tolerance,
        fp4_payload_atol=fp4_payload_atol,
    )
    print("Correctness: PASS")

    if iterations > 0:
        for _ in range(warmup_iters):
            run(tensors, out_dtype=out_dtype, scale_bulk_copy=scale_bulk_copy)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run(tensors, out_dtype=out_dtype, scale_bulk_copy=scale_bulk_copy)
        end.record()
        end.synchronize()
        average_us = start.elapsed_time(end) * 1000.0 / iterations
        tflops = (2 * m * n * k * batch) / (average_us * 1.0e-6) / 1.0e12
        print(
            f"Benchmark: {average_us:.3f} us, {tflops:.3f} TFLOP/s "
            f"({warmup_iters} warmup, {iterations} iterations)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CUDA Lang CTA_2 NVFP4 GEMM with fused FP4 quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mnkl", type=_parse_mnkl, default=_DEFAULT_MNKL, help="M,N,K,L dimensions"
    )
    parser.add_argument("--out-dtype", choices=("fp16", "fp4"), default="fp4")
    parser.add_argument("--scale-bulk-copy", action="store_true")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Benchmark warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Benchmark timing iterations; zero disables benchmarking",
    )
    parser.add_argument(
        "--keep-ptx",
        action="store_true",
        help="Emit retained PTX through CUDA_LANG_LOGS",
    )
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    parser.add_argument(
        "--fp4-payload-atol",
        type=float,
        default=_DEFAULT_FP4_PAYLOAD_ATOL,
    )
    args = parser.parse_args()
    if (
        args.tolerance < 0
        or args.fp4_payload_atol < 0
        or args.warmup_iters < 0
        or args.iterations < 0
    ):
        parser.error("tolerances and iteration counts must be non-negative")
    run_nvfp4_gemm(
        args.mnkl,
        tolerance=args.tolerance,
        scale_bulk_copy=args.scale_bulk_copy,
        out_dtype=args.out_dtype,
        fp4_payload_atol=args.fp4_payload_atol,
        warmup_iters=args.warmup_iters,
        iterations=args.iterations,
        keep_ptx=args.keep_ptx,
    )
