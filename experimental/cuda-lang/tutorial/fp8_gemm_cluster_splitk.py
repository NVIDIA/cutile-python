# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Warp-specialized FP8 GEMM with cluster split-K reduction.

Computes ``C[M, N, L] = A[M, K, L] @ B[K, N, L]`` with FP8 E4M3
inputs and output. The physical tensors use K-major A/B and M-major C layouts.

CUDA Lang notes beyond the preceding tutorial samples:

* CUDA Lang does not yet expose public wrappers for the byte-oriented
  ``stmatrix.m16n8`` operations used by the FP8 epilogue. The small STSM helper
  calls the corresponding ``cl._nvvm`` intrinsics. Public FP32-to-FP8 casts
  currently fail during IR-to-MLIR lowering, so the helper also uses the packed
  ``ff_to_e4m3x2_rn`` intrinsic.
* CUDA Lang does not yet expose the DSMEM transaction-counted
  ``st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32`` operation.
  The split-K handoff helper emits that instruction with ``cl._inline_ptx``.
* CUDA Lang's tensor-map builder does not currently accept FP8 element types.
  TMA transports these tensors as bytes, so the launch passes zero-copy UINT8
  views with identical shapes, strides, addresses, and underlying FP8 bits.
* The source guards TMA stores by epilogue warp but omits the required elected
  lane. CUDA Lang's public TMA-store operation follows the single-thread issue
  contract, so this port adds ``elect_sync()``; issuing from all 32 lanes causes
  a launch failure on B200.

The implementation retains the source's CTA_1 FP8 QMMA, six-warp role split,
round-robin K-tile partition, staged TMA mainloop, conditional split-K scratch,
two-phase owner reduction, FP8 STSM packing, and TMA output stores.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cuda.lang as cl
import torch


SMEM_CAPACITY = 232_448
MBAR_RESERVATION_BYTES = 1_024
TMEM_COLS = 512
EPILOGUE_STAGES = 2
REDUCTION_STAGES = 2
TILE_K = 128
MMA_K = 32

WARP_SIZE = 32
EPILOGUE_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5
BLOCK_THREADS = 6 * WARP_SIZE
TMEM_BARRIER_ID = 1
TMEM_BARRIER_THREADS = (EPILOGUE_WARPS + 1) * WARP_SIZE
EPILOGUE_BARRIER_ID = 2
EPILOGUE_THREADS = EPILOGUE_WARPS * WARP_SIZE

VALID_TILE_M = (64, 128)
VALID_TILE_N = (8, 16, 32, 64)
VALID_CLUSTER_SIZE = (1, 2, 3, 4, 5, 6, 7, 8)

_DEFAULT_MNKL = (4096, 8, 14336, 1)
_DEFAULT_TOLERANCE = 0.5


@dataclass(frozen=True)
class KernelSchedule:
    """Compile-time shared-memory schedule derived from tile/cluster choices."""

    num_bytes_a: int
    num_bytes_b: int
    num_ab_stages: int
    reduction_chunk_elems: int
    num_batches: int


def _compute_schedule(
    tile_m: int,
    tile_n: int,
    cluster_size: int,
) -> KernelSchedule:
    epilogue_subtile_n = min(32, tile_n)
    num_bytes_a = tile_m * TILE_K
    num_bytes_b = tile_n * TILE_K
    reduction_chunk_elems = tile_m * epilogue_subtile_n
    y_accum_bytes = (
        REDUCTION_STAGES
        * (cluster_size - 1)
        * reduction_chunk_elems
        * 4
        if cluster_size > 1
        else 0
    )
    stage_bytes = num_bytes_a + num_bytes_b
    num_ab_stages = (
        SMEM_CAPACITY - MBAR_RESERVATION_BYTES - y_accum_bytes
    ) // stage_bytes
    num_ab_stages = max(num_ab_stages, 2)

    subtile_count = cl.cdiv(tile_n, epilogue_subtile_n)
    num_batches = (
        cl.cdiv(subtile_count, cluster_size) if cluster_size > 1 else 0
    )

    total_smem = (
        MBAR_RESERVATION_BYTES + y_accum_bytes + num_ab_stages * stage_bytes
    )
    if total_smem > SMEM_CAPACITY:
        raise ValueError(
            f"shared-memory schedule requires {total_smem} bytes, "
            f"but SM100 provides {SMEM_CAPACITY}"
        )

    splitk_mbar_bytes = (
        8 * REDUCTION_STAGES * 2 if cluster_size > 1 else 0
    )
    mbar_bytes = 8 * num_ab_stages * 2 + 8 + splitk_mbar_bytes + 4
    if mbar_bytes > MBAR_RESERVATION_BYTES:
        raise ValueError("mbarrier schedule exceeds its reserved space")

    return KernelSchedule(
        num_bytes_a=num_bytes_a,
        num_bytes_b=num_bytes_b,
        num_ab_stages=num_ab_stages,
        reduction_chunk_elems=reduction_chunk_elems,
        num_batches=num_batches,
    )


def _wait_mbarrier(mbar, phase):
    while not cl.mbarrier_try_wait_parity(
        mbar, phase, time_hint=10_000_000
    ):
        pass


def _p3_to_u64(pointer):
    return cl.uint64(cl.bitcast(pointer, cl.uint32))


def _tmem_pointer(base, lane_offset, column_offset):
    pointer_dtype = cl.pointer_dtype(base.pointee_dtype, cl.MemorySpace.TENSOR)
    address = cl.bitcast(base, cl.uint32)
    offset = (cl.uint32(lane_offset) << 16) + cl.uint32(column_offset)
    return cl.bitcast(address + offset, pointer_dtype)


def _load_tmem(base, lane_offset, column_offset, repetitions):
    tmem = _tmem_pointer(base, lane_offset, column_offset)
    regs = cl.tcgen05_load(
        cl.Tcgen05LoadStoreShape.SHAPE_16X256B,
        tmem,
        count=repetitions,
    )
    return cl.Vector(
        *tuple(
            cl.bitcast(regs[i], cl.float32)
            for i in cl.static_iter(range(repetitions * 4))
        ),
        dtype=cl.float32,
    )


def _pack_e4m3x4(values, base):
    lo = cl.uint32(
        cl.uint16(
            cl._nvvm.ff_to_e4m3x2_rn(values[base + 1], values[base])
        )
    )
    hi = cl.uint32(
        cl.uint16(
            cl._nvvm.ff_to_e4m3x2_rn(values[base + 3], values[base + 2])
        )
    )
    return cl.int32(lo | (hi << 16))


def _store_fp8_stmatrix(
    smem,
    values,
    m_base,
    tile_m,
    repetitions,
    swizzle_mask,
    swizzle_shift,
    lane,
):
    """Convert one LDTM pass to E4M3 and issue the matching STSM shape."""
    if repetitions == 4:
        r0 = _pack_e4m3x4(values, 0)
        r1 = _pack_e4m3x4(values, 4)
        r2 = _pack_e4m3x4(values, 8)
        r3 = _pack_e4m3x4(values, 12)
        n_in_box = lane
        offset = n_in_box * tile_m + m_base
        offset = offset ^ (((offset >> swizzle_shift) & swizzle_mask) << 4)
        cl._nvvm.stmatrix_sync_aligned_m16n8_x4_trans_b8(
            smem + offset, r0, r1, r2, r3
        )
    elif repetitions == 2:
        r0 = _pack_e4m3x4(values, 0)
        r1 = _pack_e4m3x4(values, 4)
        n_in_box = lane % 16
        offset = n_in_box * tile_m + m_base
        offset = offset ^ (((offset >> swizzle_shift) & swizzle_mask) << 4)
        cl._nvvm.stmatrix_sync_aligned_m16n8_x2_trans_b8(
            smem + offset, r0, r1
        )
    else:
        r0 = _pack_e4m3x4(values, 0)
        n_in_box = lane % 16
        offset = n_in_box * tile_m + m_base
        offset = offset ^ (((offset >> swizzle_shift) & swizzle_mask) << 4)
        cl._nvvm.stmatrix_sync_aligned_m16n8_x1_trans_b8(smem + offset, r0)


def _st_async_v4_b32(dst, values, base, mbar):
    """Send four FP32 registers to peer DSMEM and complete 16 tx bytes."""
    cl._inline_ptx(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[%0], {%1, %2, %3, %4}, [%5];",
        ("C", dst),
        ("r", cl.bitcast(values[base], cl.int32)),
        ("r", cl.bitcast(values[base + 1], cl.int32)),
        ("r", cl.bitcast(values[base + 2], cl.int32)),
        ("r", cl.bitcast(values[base + 3], cl.int32)),
        ("C", mbar),
    )


def _peer_vector(y_accum, base, groups, lane):
    chunks = tuple(
        (y_accum + base + group * 128 + lane * 4).load(
            count=4, alignment=16
        )
        for group in cl.static_iter(range(groups))
    )
    return cl.Vector(
        *tuple(
            chunks[group][element]
            for group in cl.static_iter(range(groups))
            for element in cl.static_iter(range(4))
        ),
        dtype=cl.float32,
    )


@cl.kernel
def _kernel(
    a,
    b,
    c,
    tile_m: cl.Constant[int],
    tile_n: cl.Constant[int],
    cluster_size: cl.Constant[int],
    num_bytes_a: cl.Constant[int],
    num_bytes_b: cl.Constant[int],
    num_ab_stages: cl.Constant[int],
    reduction_chunk_elems: cl.Constant[int],
    num_batches: cl.Constant[int],
):
    m, n, batch_count = c.shape
    k = a.shape[0]

    tid = cl.thread_index(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    tile_coord_m = cl.block_index(0)
    tile_coord_n = cl.block_index(1)
    tile_coord_l = cl.block_index(2) // cluster_size
    cta_rank = cl.block_in_cluster_index(2)

    epilogue_subtile_n = min(32, tile_n)
    if tile_m == 128:
        c_tmap = cl.tensor_map_tiled(
            c,
            (tile_m, epilogue_subtile_n, 1),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )
    else:
        c_tmap = cl.tensor_map_tiled(
            c,
            (tile_m, epilogue_subtile_n, 1),
            swizzle=cl.SwizzleMode.SWIZZLE_64B,
        )
    a_tmap = cl.tensor_map_tiled(
        a,
        (TILE_K, tile_m, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    b_tmap = cl.tensor_map_tiled(
        b,
        (TILE_K, tile_n, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )

    ab_full = cl.shared_array(num_ab_stages, cl.mbarrier, alignment=8)
    ab_empty = cl.shared_array(num_ab_stages, cl.mbarrier, alignment=8)
    acc_full = cl.shared_array(1, cl.mbarrier, alignment=8)
    tmem_storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
        alignment=4,
    )

    splitk_stages = REDUCTION_STAGES if cluster_size > 1 else 0
    y_reduce_full = cl.shared_array(
        splitk_stages, cl.mbarrier, alignment=8
    )
    y_reduce_empty = cl.shared_array(
        splitk_stages, cl.mbarrier, alignment=8
    )
    y_accum = cl.shared_array(
        splitk_stages * (cluster_size - 1) * reduction_chunk_elems,
        cl.float32,
        alignment=128,
    )

    a_smem = cl.shared_array(
        num_bytes_a * num_ab_stages, cl.int8, alignment=1024
    )
    b_smem = cl.shared_array(
        num_bytes_b * num_ab_stages, cl.int8, alignment=1024
    )

    ab_full_base = ab_full.get_base_pointer()
    ab_empty_base = ab_empty.get_base_pointer()
    acc_full_ptr = acc_full.get_base_pointer()
    tmem_storage_ptr = tmem_storage.get_base_pointer()
    a_smem_base = a_smem.get_base_pointer()
    b_smem_base = b_smem.get_base_pointer()
    s_c_base = a_smem_base

    k_tile_count = cl.cdiv(k, TILE_K)
    local_k_count = (
        k_tile_count - cta_rank + cluster_size - 1
    ) // cluster_size

    rows_per_warp = tile_m // EPILOGUE_WARPS
    num_epilogue_passes = rows_per_warp // 16
    epilogue_tile_elems = tile_m * epilogue_subtile_n
    subtile_count = cl.cdiv(tile_n, epilogue_subtile_n)
    tmem_repetitions = epilogue_subtile_n // 8
    tmem_registers = tmem_repetitions * 4

    if warp < EPILOGUE_WARPS:
        if warp == 0 and lane < num_ab_stages:
            cl.mbarrier_initialize(ab_full.get_element_pointer(lane), 1)
        if warp == 1 and lane < num_ab_stages:
            cl.mbarrier_initialize(ab_empty.get_element_pointer(lane), 1)
        if warp == 2 and cluster_size > 1 and lane < REDUCTION_STAGES:
            cl.mbarrier_initialize(
                y_reduce_full.get_element_pointer(lane), 1
            )
            cl.mbarrier_initialize(
                y_reduce_empty.get_element_pointer(lane), cluster_size - 1
            )
        if warp == 3 and cl.elect_sync():
            cl.mbarrier_initialize(acc_full_ptr, 1)

    cl.fence_mbarrier_initialize()
    cl.barrier_sync_block()

    if cluster_size > 1:
        cl.barrier_arrive_cluster(
            aligned=False, memory_order=cl.MemoryOrder.RELAXED
        )

    coord_m = tile_coord_m * tile_m
    coord_n = tile_coord_n * tile_n

    if warp == TMA_WARP:
        stage = 0
        k_tile = cta_rank
        empty_phase = 1
        remaining_after_tile = local_k_count - 1
        peek_empty = cl.mbarrier_test_wait_parity(ab_empty_base, empty_phase)

        for _ in range(local_k_count):
            empty_stage = ab_empty.get_element_pointer(stage)
            full_stage = ab_full.get_element_pointer(stage)
            if not peek_empty:
                _wait_mbarrier(empty_stage, empty_phase)

            if cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    full_stage, num_bytes_a + num_bytes_b
                )

            if cl.elect_sync():
                coord_k = k_tile * TILE_K
                cl.copy_async_bulk_tensor_global_to_shared(
                    a_tmap,
                    (coord_k, coord_m, tile_coord_l),
                    a_smem_base + num_bytes_a * stage,
                    full_stage,
                )
                cl.copy_async_bulk_tensor_global_to_shared(
                    b_tmap,
                    (coord_k, coord_n, tile_coord_l),
                    b_smem_base + num_bytes_b * stage,
                    full_stage,
                )

            next_stage = stage + 1
            next_phase = empty_phase
            if next_stage == num_ab_stages:
                next_stage = 0
                next_phase = empty_phase ^ 1
            peek_empty = True
            if remaining_after_tile != 0:
                peek_empty = cl.mbarrier_test_wait_parity(
                    ab_empty.get_element_pointer(next_stage), next_phase
                )
            stage = next_stage
            empty_phase = next_phase
            k_tile += cluster_size
            remaining_after_tile -= 1

    elif warp == MMA_WARP:
        cl.tcgen05_allocate(
            tmem_storage_ptr, TMEM_COLS, cta_group=cl.CTAGroup.CTA_1
        )
        cl.barrier_sync_block(
            number_of_threads=TMEM_BARRIER_THREADS,
            barrier_id=TMEM_BARRIER_ID,
        )
        tmem_base = tmem_storage[0]

        instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=cl.Tcgen05InstructionDescriptor.F8F6F4Type.E4M3,
            b_type=cl.Tcgen05InstructionDescriptor.F8F6F4Type.E4M3,
            n=tile_n,
            m=tile_m,
        ).encode()
        a_descriptor_base = cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=_p3_to_u64(a_smem_base),
            leading_dimension_byte_offset=16,
            stride_dimension_byte_offset=1024,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
        b_descriptor_base = cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=_p3_to_u64(b_smem_base),
            leading_dimension_byte_offset=16,
            stride_dimension_byte_offset=1024,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
        a_stage_offset = num_bytes_a >> 4
        b_stage_offset = num_bytes_b >> 4

        stage = 0
        full_phase = 0
        remaining_after_tile = local_k_count - 1
        accumulate = False
        peek_full = cl.mbarrier_test_wait_parity(ab_full_base, full_phase)

        for _ in range(local_k_count):
            full_stage = ab_full.get_element_pointer(stage)
            if not peek_full:
                _wait_mbarrier(full_stage, full_phase)

            a_descriptor = a_descriptor_base + cl.uint64(
                stage * a_stage_offset
            )
            b_descriptor = b_descriptor_base + cl.uint64(
                stage * b_stage_offset
            )
            for kblock in cl.static_iter(range(TILE_K // MMA_K)):
                descriptor_increment = 2 * kblock
                if cl.elect_sync():
                    cl.tcgen05_mma(
                        cl.Tcgen05MMAKind.F8F6F4,
                        tmem_base,
                        a_descriptor + cl.uint64(descriptor_increment),
                        b_descriptor + cl.uint64(descriptor_increment),
                        instruction,
                        accumulate=accumulate,
                        cta_group=cl.CTAGroup.CTA_1,
                    )
                accumulate = True

            if cl.elect_sync():
                cl.tcgen05_commit(ab_empty.get_element_pointer(stage))

            next_stage = stage + 1
            next_phase = full_phase
            if next_stage == num_ab_stages:
                next_stage = 0
                next_phase = full_phase ^ 1
            peek_full = True
            if remaining_after_tile != 0:
                peek_full = cl.mbarrier_test_wait_parity(
                    ab_full.get_element_pointer(next_stage), next_phase
                )
            stage = next_stage
            full_phase = next_phase
            remaining_after_tile -= 1

        if cl.elect_sync():
            cl.tcgen05_commit(acc_full_ptr)

    elif warp < EPILOGUE_WARPS:
        cl.barrier_sync_block(
            number_of_threads=TMEM_BARRIER_THREADS,
            barrier_id=TMEM_BARRIER_ID,
        )
        tmem_base = tmem_storage[0]
        lane_offset = warp * rows_per_warp if tile_m == 128 else 0
        coord_c_m = tile_coord_m * tile_m
        coord_c_n = tile_coord_n * tile_n
        epilogue_stage = 0
        swizzle_mask = tile_m // 16 - 1
        swizzle_shift = 7

        _wait_mbarrier(acc_full_ptr, 0)

        if cluster_size == 1:
            for subtile in range(subtile_count):
                epilogue_stage = (epilogue_stage + 1) % EPILOGUE_STAGES
                s_c_stage = (
                    s_c_base + epilogue_stage * epilogue_tile_elems
                )
                for pass_idx in cl.static_iter(range(num_epilogue_passes)):
                    values = _load_tmem(
                        tmem_base,
                        lane_offset + pass_idx * 16,
                        subtile * epilogue_subtile_n,
                        tmem_repetitions,
                    )
                    m_base = warp * rows_per_warp + pass_idx * 16
                    _store_fp8_stmatrix(
                        s_c_stage,
                        values,
                        m_base,
                        tile_m,
                        tmem_repetitions,
                        swizzle_mask,
                        swizzle_shift,
                        lane,
                    )

                cl.fence_proxy(
                    cl.FenceProxyKind.ASYNC_SHARED,
                    space=cl.MemorySpace.SHARED,
                )
                cl.barrier_sync_block(
                    number_of_threads=EPILOGUE_THREADS,
                    barrier_id=EPILOGUE_BARRIER_ID,
                )
                if warp == 0 and cl.elect_sync():
                    cl.copy_async_bulk_tensor_shared_to_global(
                        s_c_stage,
                        c_tmap,
                        (
                            coord_c_m,
                            coord_c_n + subtile * epilogue_subtile_n,
                            tile_coord_l,
                        ),
                    )
                    cl.copy_async_bulk_commit_group()
                    cl.copy_async_bulk_wait_group(
                        EPILOGUE_STAGES - 1, read=True
                    )
                cl.barrier_sync_block(
                    number_of_threads=EPILOGUE_THREADS,
                    barrier_id=EPILOGUE_BARRIER_ID,
                )

        else:
            cl.barrier_wait_cluster(aligned=False)

            peer_count = cluster_size - 1
            n_remaining = n - coord_c_n
            valid_subtiles_raw = cl.cdiv(n_remaining, epilogue_subtile_n)
            valid_subtiles = (
                valid_subtiles_raw
                if valid_subtiles_raw < subtile_count
                else subtile_count
            )
            store_groups = tmem_registers // 4
            group_elems = WARP_SIZE * 4
            if tile_m == 128:
                warp_base = warp * 2 * store_groups * group_elems
            else:
                warp_base = warp * store_groups * group_elems
            y_accum_base = y_accum.get_base_pointer()

            for batch_idx in range(num_batches):
                batch_start = batch_idx * cluster_size
                buffer_idx = batch_idx % REDUCTION_STAGES
                full_phase = (batch_idx // REDUCTION_STAGES) % 2
                empty_phase = (1 + batch_idx // REDUCTION_STAGES) % 2
                buffer_offset = (
                    buffer_idx * peer_count * reduction_chunk_elems
                )

                if batch_idx >= REDUCTION_STAGES:
                    _wait_mbarrier(
                        y_reduce_empty.get_element_pointer(buffer_idx),
                        empty_phase,
                    )

                for offset in cl.static_iter(range(cluster_size)):
                    subtile = batch_start + offset
                    owner = offset
                    if cta_rank != owner and subtile < valid_subtiles:
                        values0 = _load_tmem(
                            tmem_base,
                            lane_offset,
                            subtile * epilogue_subtile_n,
                            tmem_repetitions,
                        )
                        if tile_m == 128:
                            values1 = _load_tmem(
                                tmem_base,
                                lane_offset + 16,
                                subtile * epilogue_subtile_n,
                                tmem_repetitions,
                            )

                        peer_slot = (
                            cta_rank if cta_rank < owner else cta_rank - 1
                        )
                        remote_y = cl.map_shared_to_cluster(
                            y_accum_base, owner
                        )
                        remote_full = cl.map_shared_to_cluster(
                            y_reduce_full.get_element_pointer(buffer_idx), owner
                        )

                        for pass_idx in cl.static_iter(
                            range(num_epilogue_passes)
                        ):
                            values = values0
                            if pass_idx == 1:
                                values = values1
                            pass_base = (
                                warp_base
                                + pass_idx * store_groups * group_elems
                            )
                            for group in cl.static_iter(range(store_groups)):
                                dst_offset = (
                                    buffer_offset
                                    + peer_slot * reduction_chunk_elems
                                    + pass_base
                                    + group * group_elems
                                    + lane * 4
                                )
                                _st_async_v4_b32(
                                    remote_y + dst_offset,
                                    values,
                                    group * 4,
                                    remote_full,
                                )

                my_subtile = batch_start + cta_rank
                if my_subtile < valid_subtiles:
                    reduced0 = _load_tmem(
                        tmem_base,
                        lane_offset,
                        my_subtile * epilogue_subtile_n,
                        tmem_repetitions,
                    )
                    if tile_m == 128:
                        reduced1 = _load_tmem(
                            tmem_base,
                            lane_offset + 16,
                            my_subtile * epilogue_subtile_n,
                            tmem_repetitions,
                        )

                    if warp == 0 and cl.elect_sync():
                        cl.mbarrier_arrive_expect_transaction(
                            y_reduce_full.get_element_pointer(buffer_idx),
                            peer_count * reduction_chunk_elems * 4,
                        )
                    _wait_mbarrier(
                        y_reduce_full.get_element_pointer(buffer_idx),
                        full_phase,
                    )

                    for peer in cl.static_iter(range(peer_count)):
                        peer_base = (
                            buffer_offset
                            + peer * reduction_chunk_elems
                            + warp_base
                        )
                        for pass_idx in cl.static_iter(
                            range(num_epilogue_passes)
                        ):
                            pass_base = (
                                peer_base
                                + pass_idx * store_groups * group_elems
                            )
                            peer_values = _peer_vector(
                                y_accum_base, pass_base, store_groups, lane
                            )
                            if pass_idx == 0:
                                reduced0 = reduced0 + peer_values
                            else:
                                reduced1 = reduced1 + peer_values

                    epilogue_stage = (
                        epilogue_stage + 1
                    ) % EPILOGUE_STAGES
                    s_c_stage = (
                        s_c_base + epilogue_stage * epilogue_tile_elems
                    )
                    for pass_idx in cl.static_iter(
                        range(num_epilogue_passes)
                    ):
                        values = reduced0
                        if pass_idx == 1:
                            values = reduced1
                        m_base = warp * rows_per_warp + pass_idx * 16
                        _store_fp8_stmatrix(
                            s_c_stage,
                            values,
                            m_base,
                            tile_m,
                            tmem_repetitions,
                            swizzle_mask,
                            swizzle_shift,
                            lane,
                        )

                    cl.fence_proxy(
                        cl.FenceProxyKind.ASYNC_SHARED,
                        space=cl.MemorySpace.SHARED,
                    )
                    cl.barrier_sync_block(
                        number_of_threads=EPILOGUE_THREADS,
                        barrier_id=EPILOGUE_BARRIER_ID,
                    )
                    if warp == 0 and cl.elect_sync():
                        cl.copy_async_bulk_tensor_shared_to_global(
                            s_c_stage,
                            c_tmap,
                            (
                                coord_c_m,
                                coord_c_n
                                + my_subtile * epilogue_subtile_n,
                                tile_coord_l,
                            ),
                        )
                        cl.copy_async_bulk_commit_group()
                        cl.copy_async_bulk_wait_group(
                            EPILOGUE_STAGES - 1, read=True
                        )
                    cl.barrier_sync_block(
                        number_of_threads=EPILOGUE_THREADS,
                        barrier_id=EPILOGUE_BARRIER_ID,
                    )

                    if batch_idx + REDUCTION_STAGES < num_batches:
                        if warp == 0 and lane < peer_count:
                            target = lane if lane < cta_rank else lane + 1
                            peer_empty = cl.map_shared_to_cluster(
                                y_reduce_empty.get_element_pointer(buffer_idx),
                                target,
                            )
                            cl.mbarrier_arrive(peer_empty)

        cl.barrier_sync_block()
        if warp == 0:
            cl.tcgen05_deallocate(
                tmem_base, TMEM_COLS, cta_group=cl.CTAGroup.CTA_1
            )


def _validate_configuration(
    mnkl: tuple[int, int, int, int],
    tile_m: int,
    tile_n: int,
    cluster_size: int,
) -> None:
    if len(mnkl) != 4:
        raise ValueError("MNKL must contain exactly four values")
    m, n, k, batch = mnkl
    if min(m, n, k, batch) <= 0:
        raise ValueError("MNKL values must be positive")
    if m % 16 != 0:
        raise ValueError("M must be divisible by 16 for TMA alignment")
    if k % 16 != 0:
        raise ValueError("K must be divisible by 16 for TMA alignment")
    if tile_m not in VALID_TILE_M:
        raise ValueError(f"tile_m must be one of {VALID_TILE_M}")
    if tile_n not in VALID_TILE_N:
        raise ValueError(f"tile_n must be one of {VALID_TILE_N}")
    if cluster_size not in VALID_CLUSTER_SIZE:
        raise ValueError(
            f"cluster_size must be one of {VALID_CLUSTER_SIZE}"
        )
    _compute_schedule(tile_m, tile_n, cluster_size)


def prepare_tensors(m: int, n: int, k: int, batch: int):
    """Create K-major FP8 A/B and M-major FP8 C tensors."""
    torch.manual_seed(1111)
    a = (
        torch.randint(
            -3, 4, (batch, m, k), dtype=torch.float32, device="cuda"
        )
        .to(torch.float8_e4m3fn)
        .permute(2, 1, 0)
    )
    b = (
        torch.randint(
            -3, 4, (batch, n, k), dtype=torch.float32, device="cuda"
        )
        .to(torch.float8_e4m3fn)
        .permute(2, 1, 0)
    )
    c = torch.zeros(
        (batch, n, m), dtype=torch.float8_e4m3fn, device="cuda"
    ).permute(2, 1, 0)
    return {"a": a, "b": b, "c": c}


def run(
    tensors: dict[str, torch.Tensor],
    tile_m: int = 128,
    tile_n: int = 8,
    cluster_size: int = 2,
    stream=None,
) -> None:
    a, b, c = tensors["a"], tensors["b"], tensors["c"]
    if a.ndim != 3 or b.ndim != 3 or c.ndim != 3:
        raise ValueError("A, B, and C must be rank-3 tensors")
    k, m, batch = a.shape
    if b.shape != (k, c.shape[1], batch):
        raise ValueError("B must have shape (K, N, L)")
    if c.shape != (m, b.shape[1], batch):
        raise ValueError("C must have shape (M, N, L)")
    if any(
        tensor.dtype != torch.float8_e4m3fn for tensor in (a, b, c)
    ):
        raise ValueError("A, B, and C must have dtype torch.float8_e4m3fn")

    n = b.shape[1]
    _validate_configuration((m, n, k, batch), tile_m, tile_n, cluster_size)
    schedule = _compute_schedule(tile_m, tile_n, cluster_size)
    cuda_stream = torch.cuda.current_stream() if stream is None else stream
    # Tensor maps currently reject FP8 dtype metadata. These one-byte views do
    # not change storage, shape, strides, or the bits consumed by QMMA.
    a_bytes = a.view(torch.uint8)
    b_bytes = b.view(torch.uint8)
    c_bytes = c.view(torch.uint8)
    cl.launch(
        cuda_stream,
        (cl.cdiv(m, tile_m), cl.cdiv(n, tile_n), batch * cluster_size),
        (BLOCK_THREADS, 1, 1),
        _kernel,
        (
            a_bytes,
            b_bytes,
            c_bytes,
            tile_m,
            tile_n,
            cluster_size,
            schedule.num_bytes_a,
            schedule.num_bytes_b,
            schedule.num_ab_stages,
            schedule.reduction_chunk_elems,
            schedule.num_batches,
        ),
        block_in_cluster_count=(1, 1, cluster_size),
    )


def verify_output(
    tensors: dict[str, torch.Tensor], tolerance: float = _DEFAULT_TOLERANCE
) -> None:
    a, b, c = tensors["a"], tensors["b"], tensors["c"]
    _, _, batch = c.shape
    reference = torch.empty_like(c)
    for batch_idx in range(batch):
        result = a[:, :, batch_idx].float().T @ b[:, :, batch_idx].float().contiguous()
        reference[:, :, batch_idx] = result.clamp(-448.0, 448.0).to(
            torch.float8_e4m3fn
        )
    torch.testing.assert_close(
        c.float(), reference.float(), atol=tolerance, rtol=0.0
    )


def verify(
    mnkl: tuple[int, int, int, int] = _DEFAULT_MNKL,
    tile_m: int = 128,
    tile_n: int = 8,
    cluster_size: int = 2,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> None:
    _validate_configuration(mnkl, tile_m, tile_n, cluster_size)
    m, n, k, batch = mnkl
    tensors = prepare_tensors(m, n, k, batch)
    run(tensors, tile_m=tile_m, tile_n=tile_n, cluster_size=cluster_size)
    torch.cuda.synchronize()
    verify_output(tensors, tolerance=tolerance)
    print(
        f"verify (mnkl={mnkl}, tile=({tile_m}, {tile_n}, {TILE_K}), "
        f"cluster_size={cluster_size}): PASS"
    )


def _parse_mnkl(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected four comma-separated integers"
        ) from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("MNKL must contain exactly four values")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnkl", type=_parse_mnkl, default=_DEFAULT_MNKL)
    parser.add_argument("--tile_m", type=int, choices=VALID_TILE_M, default=128)
    parser.add_argument("--tile_n", type=int, choices=VALID_TILE_N, default=8)
    parser.add_argument(
        "--cluster_size", type=int, choices=VALID_CLUSTER_SIZE, default=2
    )
    parser.add_argument(
        "--tolerance", type=float, default=_DEFAULT_TOLERANCE
    )
    args = parser.parse_args()
    verify(
        args.mnkl,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        cluster_size=args.cluster_size,
        tolerance=args.tolerance,
    )
