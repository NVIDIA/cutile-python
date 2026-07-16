# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CTA2 FP8 GEMM port of ThunderKittens' ``fp8_b200_gemm.cu``.
"""

from dataclasses import dataclass

import pytest
import torch

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability

if tuple(get_compute_capability()) != (10, 0):
    pytest.skip("Requires Blackwell", True)

WARP_SIZE = 32
MMA_K = 32


@dataclass(frozen=True)
class FP8B200GemmConfig:
    tile_n: int
    tile_k: int
    supergroup_size: int
    overlap_mma_epilogue: bool
    load_stages: int
    epilogue_stages: int

    def __post_init__(self):
        assert 16 <= self.tile_n <= 256 and self.tile_n % 16 == 0
        assert self.tile_k >= 32 and self.tile_k % 32 == 0
        assert 1 <= self.supergroup_size <= 16
        assert 1 <= self.load_stages <= 16
        assert 1 <= self.epilogue_stages <= 16
        assert self.tile_n % self.epilogue_stages == 0
        assert self.tile_k % 128 == 0

    @property
    def num_consumers(self):
        return 1 if self.overlap_mma_epilogue else 2

    @property
    def accumulator_stages(self):
        return 2 if self.overlap_mma_epilogue else 1

    @property
    def num_warps(self):
        return (self.num_consumers + 1) * 4

    def __str__(self):
        return (
            f"tile_n={self.tile_n}, tile_k={self.tile_k}, "
            f"supergroup_size={self.supergroup_size}, "
            f"overlap_mma_epilogue={self.overlap_mma_epilogue}, "
            f"load_stages={self.load_stages}, "
            f"epilogue_stages={self.epilogue_stages}"
        )


CONFIGS = (
    FP8B200GemmConfig(64, 256, 4, True, 5, 2),
    FP8B200GemmConfig(256, 128, 8, True, 6, 4),
    FP8B200GemmConfig(256, 128, 4, True, 6, 4),
    FP8B200GemmConfig(256, 128, 8, False, 4, 8),
)

BENCHMARK_CONFIGS = (
    (1024, CONFIGS[0]),
    (2048, CONFIGS[1]),
    (4096, CONFIGS[2]),
    (8192, CONFIGS[3]),
    (16384, CONFIGS[3]),
)


def fence_clc_acquire():
    cl.fence_proxy_sync_restrict(cl.MemoryOrder.ACQUIRE)


def fence_clc_release():
    cl.fence_proxy_sync_restrict(cl.MemoryOrder.RELEASE)


def swizzle_program_id(tile, tiles_m, tiles_n, width):
    full_groups = tiles_m // width
    full_rows = full_groups * width
    full_tiles = full_rows * tiles_n
    group = tile // (width * tiles_n)
    if tile < full_tiles:
        pid_m = group * width + tile % width
        pid_n = (tile // width) % tiles_n
    else:
        partial_width = tiles_m - full_rows
        if partial_width == 0:
            partial_width = 1
        partial = tile - full_tiles
        pid_m = full_rows + partial % partial_width
        pid_n = (partial // partial_width) % tiles_n
    if group % 2 != 0:
        pid_n = tiles_n - 1 - pid_n
    return pid_m, pid_n


def consume_scheduled_tile(ready, consumed, next_tile, next_has_work, phase):
    cl.mbarrier_wait_parity(ready, phase)
    tile = next_tile[0]
    has_work = next_has_work[0] != 0
    if cl.elect_sync():
        cl.mbarrier_arrive(consumed)
    return tile, has_work


def sync_consumer_warpgroup(consumer):
    cl.barrier_sync_block(4 * WARP_SIZE, 1 if consumer == 0 else 2, aligned=False)


def store_bf16_tmem_tile(dst, tmem_base, warp, column, row, output_column, n):
    tmem = cl.tcgen05_tmem_offset(
        tmem_base,
        lane_offset=warp * WARP_SIZE,
        column_offset=column,
    )
    regs = cl.tcgen05_load(cl.Tcgen05LoadStoreShape.SHAPE_32X32B, tmem, count=16)
    cl.tcgen05_wait_load()
    for pair in cl.static_iter(range(8)):
        lo = cl.bitcast(regs[pair * 2], cl.float32)
        hi = cl.bitcast(regs[pair * 2 + 1], cl.float32)
        packed = cl._nvvm.ff2bf16x2_rn(hi, lo)
        (dst + row * n + output_column + pair * 2).store(packed, alignment=4)


def store_bf16_tmem_subtile(dst, tmem_base, warp, column, row, width):
    for chunk in cl.static_iter(range(width // 32)):
        tmem = cl.tcgen05_tmem_offset(
            tmem_base,
            lane_offset=warp * WARP_SIZE,
            column_offset=column + chunk * 32,
        )
        regs = cl.tcgen05_load(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            tmem,
            count=32,
        )
        cl.tcgen05_wait_load()
        for pair in cl.static_iter(range(16)):
            lo = cl.bitcast(regs[pair * 2], cl.float32)
            hi = cl.bitcast(regs[pair * 2 + 1], cl.float32)
            packed = cl._nvvm.ff2bf16x2_rn(hi, lo)
            offset = row * width + chunk * 32 + pair * 2
            (dst + offset).store(packed, alignment=4)


def store_persistent_partition(
    c_tmap,
    c_smem,
    tmem_base,
    acc_ready,
    acc_empty,
    phase,
    pid_m,
    pid_n,
    rank,
    consumer,
    local_warp,
    lane,
    tile_n,
    num_consumers,
    epilogue_stages,
    num_d_tiles,
    has_next,
):
    cta_m = 128
    consumer_tile_m = 256
    subtile_n = tile_n // epilogue_stages

    cl.mbarrier_wait_parity(acc_ready, phase)
    cl.tcgen05_fence_after_thread_sync()
    row = local_warp * WARP_SIZE + lane
    for subtile in cl.static_iter(range(epilogue_stages)):
        stage = subtile % num_d_tiles
        stage_smem = c_smem.get_element_pointer((consumer, stage, 0))
        if local_warp == 0 and cl.elect_sync():
            cl.copy_async_bulk_wait_group(num_d_tiles - 1, read=True)
        sync_consumer_warpgroup(consumer)

        store_bf16_tmem_subtile(
            stage_smem,
            tmem_base,
            rank * 4 + local_warp,
            subtile * subtile_n,
            row,
            subtile_n,
        )

        if subtile == epilogue_stages - 1:
            if cl.elect_sync():
                empty = cl.map_shared_to_cluster(acc_empty, 0)
                cl.mbarrier_arrive(empty, scope=cl.MbarrierScope.BLOCK)
            if not has_next and cl.elect_sync():
                cl.grid_dependency_control_launch_dependents()
        sync_consumer_warpgroup(consumer)

        if local_warp == 0 and cl.elect_sync():
            cl.fence_proxy(
                cl.FenceProxyKind.ASYNC_SHARED,
                space=cl.MemorySpace.SHARED,
            )
            cache_hint = cl.create_fractional_cache_policy(
                cl.CachePolicy.L2_EVICT_FIRST
            )
            cl.copy_async_bulk_tensor_shared_to_global(
                stage_smem,
                c_tmap,
                (
                    pid_n * tile_n + subtile * subtile_n,
                    pid_m * consumer_tile_m * num_consumers
                    + (rank * num_consumers + consumer) * cta_m,
                ),
                l2_cache_hint=cache_hint,
            )
            cl.copy_async_bulk_commit_group()


@cl.kernel
def fp8_b200_gemm_kernel(
    a,
    b,
    c,
    m: cl.Constant[int],
    n: cl.Constant[int],
    k: cl.Constant[int],
):
    cta_m = 128
    cta_n = 128
    tile_m = 256
    tile_n = 256
    tile_k = 128
    stages = 3
    num_warps = 4

    tid = cl.thread_index(0)
    warp = tid // WARP_SIZE
    rank = cl.block_in_cluster_index(0)
    tile = cl.cluster_index(0)
    tiles_n = n // tile_n
    pid_m = tile // tiles_n
    pid_n = tile % tiles_n

    # The first dimension is contiguous so each TMA atom contains 128 FP8
    # values.  cuda.lang encodes all byte-sized FP8 tensor-map elements as U8;
    # the tcgen05 instruction descriptor supplies the numerical interpretation.
    a_tmap = cl.tensor_map_tiled(
        a,
        (128, cta_m, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    b_tmap = cl.tensor_map_tiled(
        b,
        (128, cta_n, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    c_tmap = cl.tensor_map_tiled(c, (tile_n, cta_m), order="F")

    a_smem = cl.shared_array(
        (stages, cta_m * tile_k),
        cl.uint8,
        alignment=512,
    )
    b_smem = cl.shared_array(
        (stages, cta_n * tile_k),
        cl.uint8,
        alignment=512,
    )
    c_smem = cl.shared_array(cta_m * tile_n, cl.bfloat16, alignment=128)
    load_ready = cl.shared_array(stages, cl.mbarrier, alignment=8)
    load_empty = cl.shared_array(stages, cl.mbarrier, alignment=8)
    acc_ready = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    tmem_storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR),
        alignment=4,
    )

    if warp == 0 and cl.elect_sync():
        for stage in cl.static_iter(range(stages)):
            cl.mbarrier_initialize(load_ready.get_element_pointer(stage), 2)
            cl.mbarrier_initialize(load_empty.get_element_pointer(stage), 1)
        cl.mbarrier_initialize(acc_ready, 1)
        cl.fence_mbarrier_initialize()
    cl.barrier_sync_cluster(aligned=True)

    if warp == 2:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(),
            tile_n,
            cta_group=cl.CTAGroup.CTA_2,
        )
    cl.barrier_sync_cluster(aligned=True)

    if warp == 1:
        for k_tile in range(k // tile_k):
            stage = k_tile % stages
            phase = (k_tile // stages) & 1
            if k_tile >= stages:
                cl.mbarrier_wait_parity(
                    load_empty.get_element_pointer(stage), phase ^ 1
                )
            if cl.elect_sync():
                a_stage = a_smem.get_element_pointer((stage, 0))
                b_stage = b_smem.get_element_pointer((stage, 0))
                a_dst = cl.map_shared_to_cluster(a_stage, rank)
                b_dst = cl.map_shared_to_cluster(b_stage, rank)
                ready = load_ready.get_element_pointer(stage)
                arrive_bar = cl.map_shared_to_leader_block(ready)
                expect_bar = cl.map_shared_to_cluster(ready, 0)
                cl.copy_async_bulk_tensor_global_to_shared(
                    a_tmap,
                    (0, pid_m * tile_m + rank * cta_m, k_tile),
                    a_dst,
                    arrive_bar,
                    cta_group=cl.CTAGroup.CTA_2,
                )
                cl.copy_async_bulk_tensor_global_to_shared(
                    b_tmap,
                    (0, pid_n * tile_n + rank * cta_n, k_tile),
                    b_dst,
                    arrive_bar,
                    cta_group=cl.CTAGroup.CTA_2,
                )
                cl.mbarrier_arrive_expect_transaction(
                    expect_bar,
                    (cta_m + cta_n) * tile_k,
                    scope=cl.MbarrierScope.BLOCK,
                )

    elif warp == 2:
        instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=cl.Tcgen05InstructionDescriptor.F8F6F4Type.E4M3,
            b_type=cl.Tcgen05InstructionDescriptor.F8F6F4Type.E4M3,
            n=tile_n,
            m=tile_m,
        ).encode()
        for k_tile in range(k // tile_k):
            stage = k_tile % stages
            phase = (k_tile // stages) & 1
            if rank == 0 and cl.elect_sync():
                cl.mbarrier_wait_parity(load_ready.get_element_pointer(stage), phase)
                cl.tcgen05_fence_after_thread_sync()
                a_desc = cl.Tcgen05SharedMemoryDescriptor(
                    matrix_start_address=a_smem.get_element_pointer((stage, 0)),
                    leading_dimension_byte_offset=16,
                    stride_dimension_byte_offset=8 * 128,
                    swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                ).encode()
                b_desc = cl.Tcgen05SharedMemoryDescriptor(
                    matrix_start_address=b_smem.get_element_pointer((stage, 0)),
                    leading_dimension_byte_offset=16,
                    stride_dimension_byte_offset=8 * 128,
                    swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                ).encode()
                for kk in cl.static_iter(range(tile_k // MMA_K)):
                    cl.tcgen05_mma(
                        cl.Tcgen05MMAKind.F8F6F4,
                        tmem_storage[0],
                        a_desc + (MMA_K >> 4) * kk,
                        b_desc + (MMA_K >> 4) * kk,
                        instruction,
                        accumulate=k_tile != 0 or kk != 0,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                cl.tcgen05_commit(
                    load_empty.get_element_pointer(stage),
                    multicast_mask=0b11,
                    cta_group=cl.CTAGroup.CTA_2,
                )
        if rank == 0 and cl.elect_sync():
            cl.tcgen05_commit(
                acc_ready,
                multicast_mask=0b11,
                cta_group=cl.CTAGroup.CTA_2,
            )

    cl.mbarrier_wait_parity(acc_ready, 0)
    cl.tcgen05_fence_after_thread_sync()
    row = tid
    for column in cl.static_iter(range(0, tile_n, 16)):
        store_bf16_tmem_tile(
            c_smem.get_base_pointer(),
            tmem_storage[0],
            rank * num_warps + warp,
            column,
            row,
            column,
            tile_n,
        )
    cl.barrier_sync_block()

    if warp == 0 and cl.elect_sync():
        cl.fence_proxy(cl.FenceProxyKind.ASYNC_SHARED, space=cl.MemorySpace.SHARED)
        cl.copy_async_bulk_tensor_shared_to_global(
            c_smem.get_base_pointer(),
            c_tmap,
            (pid_n * tile_n, pid_m * tile_m + rank * cta_m),
        )
        cl.copy_async_bulk_commit_group()
        cl.copy_async_bulk_wait_group(0)

    cl.barrier_sync_cluster(aligned=True)
    if warp == 2:
        cl.tcgen05_deallocate(
            tmem_storage[0],
            tile_n,
            cta_group=cl.CTAGroup.CTA_2,
        )


def _fp8_b200_gemm_persistent_kernel(
    a,
    b,
    c,
    m: cl.Constant[int],
    n: cl.Constant[int],
    k: cl.Constant[int],
    tile_n: cl.Constant[int],
    tile_k: cl.Constant[int],
    supergroup_size: cl.Constant[int],
    num_consumers: cl.Constant[int],
    load_stages: cl.Constant[int],
    epilogue_stages: cl.Constant[int],
):
    cta_m = 128
    consumer_tile_m = 256
    cta_n = tile_n // 2
    acc_stages = cl.static_eval(2 if num_consumers == 1 else 1)
    num_d_tiles = cl.static_eval(2 if epilogue_stages > 1 else 1)
    subtile_n = tile_n // epilogue_stages
    producer_base = num_consumers * 4
    scheduler_warp = producer_base + 2
    loader_warp = producer_base + 3

    tid = cl.thread_index(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    rank = cl.block_in_cluster_index(0)
    tiles_m = m // (consumer_tile_m * num_consumers)
    tiles_n = n // tile_n

    a_tmap = cl.tensor_map_tiled(
        a,
        (128, cta_m, tile_k // 128),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    b_tmap = cl.tensor_map_tiled(
        b,
        (128, cta_n, tile_k // 128),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    c_tmap = cl.tensor_map_tiled(c, (subtile_n, cta_m), order="F")
    if tid == 0:
        cl.prefetch_tensor_map(a_tmap)
        cl.prefetch_tensor_map(b_tmap)
        cl.prefetch_tensor_map(c_tmap)

    a_smem = cl.shared_array(
        (load_stages, num_consumers, cta_m * tile_k),
        cl.uint8,
        alignment=512,
    )
    b_smem = cl.shared_array(
        (load_stages, cta_n * tile_k),
        cl.uint8,
        alignment=512,
    )
    c_smem = cl.shared_array(
        (num_consumers, num_d_tiles, cta_m * subtile_n),
        cl.bfloat16,
        alignment=128,
    )
    load_ready = cl.shared_array(load_stages, cl.mbarrier, alignment=8)
    load_empty = cl.shared_array(load_stages, cl.mbarrier, alignment=8)
    acc_ready = cl.shared_array(
        acc_stages * num_consumers,
        cl.mbarrier,
        alignment=8,
    )
    acc_empty = cl.shared_array(
        acc_stages * num_consumers,
        cl.mbarrier,
        alignment=8,
    )
    clc_bar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    schedule_ready = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    schedule_consumed = cl.shared_array(
        1,
        cl.mbarrier,
        alignment=8,
    ).get_base_pointer()
    clc_token = cl.shared_array(
        1,
        cl.clusterlaunchcontrol_token,
        alignment=16,
    ).get_base_pointer()
    next_tile = cl.shared_array(1, cl.int32)
    next_has_work = cl.shared_array(1, cl.int32)
    tmem_storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR),
        alignment=4,
    )

    if warp == 0 and cl.elect_sync():
        for stage in cl.static_iter(range(load_stages)):
            cl.mbarrier_initialize(load_ready.get_element_pointer(stage), 2)
            cl.mbarrier_initialize(
                load_empty.get_element_pointer(stage),
                num_consumers,
            )
        for stage in cl.static_iter(range(acc_stages * num_consumers)):
            cl.mbarrier_initialize(acc_ready.get_element_pointer(stage), 1)
            cl.mbarrier_initialize(acc_empty.get_element_pointer(stage), 8)
        cl.mbarrier_initialize(clc_bar, 1)
        cl.mbarrier_initialize(schedule_ready, 1)
        cl.mbarrier_initialize(schedule_consumed, 1 + 2 * num_consumers)
        cl.fence_mbarrier_initialize()
    cl.barrier_sync_cluster(aligned=True)

    if warp >= producer_base:
        cl.setmaxregister_decrease(56)
    elif num_consumers == 2:
        cl.setmaxregister_increase(224)

    tmem_columns = tile_n * num_consumers * acc_stages
    if warp == producer_base:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(),
            tmem_columns,
            cta_group=cl.CTAGroup.CTA_2,
        )
    cl.barrier_sync_cluster(aligned=True)

    if warp == scheduler_warp:
        iteration = 0
        scheduling = True
        while scheduling:
            phase = iteration & 1
            if iteration != 0:
                cl.mbarrier_wait_parity(schedule_consumed, phase ^ 1)
            if rank == 0 and cl.elect_sync():
                fence_clc_acquire()
                cl.clusterlaunchcontrol_try_cancel(
                    clc_token,
                    clc_bar,
                    multicast=True,
                )
            if cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    clc_bar,
                    clc_token.pointee_dtype.bitwidth // 8,
                    scope=cl.MbarrierScope.CLUSTER,
                    memory_order=cl.MemoryOrder.RELAXED,
                )
            cl.mbarrier_wait_parity(clc_bar, phase, scope=cl.MbarrierScope.CLUSTER)
            token = clc_token.load()
            scheduling = cl.clusterlaunchcontrol_is_canceled(token)
            if cl.elect_sync():
                next_has_work[0] = cl.int32(scheduling)
                if scheduling:
                    block = cl.clusterlaunchcontrol_get_first_block_index(
                        token,
                        axis=0,
                    )
                    next_tile[0] = block // 2
                fence_clc_release()
                cl.mbarrier_arrive(schedule_ready)
            cl.barrier_sync_warp()
            iteration += 1

    elif warp == loader_warp:
        if cl.elect_sync():
            cl.grid_dependency_control_wait()
        tile = cl.cluster_index(0)
        has_work = True
        load_index = 0
        task = 0
        while has_work:
            pid_m, pid_n = swizzle_program_id(
                tile,
                tiles_m,
                tiles_n,
                supergroup_size,
            )
            for k_tile in range(k // tile_k):
                stage = load_index % load_stages
                phase = (load_index // load_stages) & 1
                if load_index >= load_stages:
                    cl.mbarrier_wait_parity(
                        load_empty.get_element_pointer(stage),
                        phase ^ 1,
                    )
                if cl.elect_sync():
                    ready = load_ready.get_element_pointer(stage)
                    arrive_bar = cl.map_shared_to_leader_block(ready)
                    expect_bar = cl.map_shared_to_cluster(ready, 0)
                    for consumer in cl.static_iter(range(num_consumers)):
                        a_stage = a_smem.get_element_pointer((stage, consumer, 0))
                        a_dst = cl.map_shared_to_cluster(a_stage, rank)
                        cl.copy_async_bulk_tensor_global_to_shared(
                            a_tmap,
                            (
                                0,
                                pid_m * consumer_tile_m * num_consumers
                                + (rank * num_consumers + consumer) * cta_m,
                                k_tile * (tile_k // 128),
                            ),
                            a_dst,
                            arrive_bar,
                            cta_group=cl.CTAGroup.CTA_2,
                        )
                    b_stage = b_smem.get_element_pointer((stage, 0))
                    b_dst = cl.map_shared_to_cluster(b_stage, rank)
                    cl.copy_async_bulk_tensor_global_to_shared(
                        b_tmap,
                        (
                            0,
                            pid_n * tile_n + rank * cta_n,
                            k_tile * (tile_k // 128),
                        ),
                        b_dst,
                        arrive_bar,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                    cl.mbarrier_arrive_expect_transaction(
                        expect_bar,
                        (num_consumers * cta_m + cta_n) * tile_k,
                        scope=cl.MbarrierScope.BLOCK,
                    )
                load_index += 1
            tile, has_work = consume_scheduled_tile(
                schedule_ready,
                schedule_consumed,
                next_tile.get_base_pointer(),
                next_has_work.get_base_pointer(),
                task & 1,
            )
            task += 1

    elif warp >= producer_base and warp < producer_base + num_consumers:
        consumer = warp - producer_base
        tile = cl.cluster_index(0)
        has_work = True
        load_index = 0
        task = 0
        instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=cl.Tcgen05InstructionDescriptor.F8F6F4Type.E4M3,
            b_type=cl.Tcgen05InstructionDescriptor.F8F6F4Type.E4M3,
            n=tile_n,
            m=consumer_tile_m,
        ).encode()
        while has_work:
            scheduled_tile, scheduled_work = consume_scheduled_tile(
                schedule_ready,
                schedule_consumed,
                next_tile.get_base_pointer(),
                next_has_work.get_base_pointer(),
                task & 1,
            )
            acc_slot = task % acc_stages
            acc_phase = (task // acc_stages) & 1
            acc_index = acc_slot * num_consumers + consumer
            acc_tmem = cl.tcgen05_tmem_offset(
                tmem_storage[0],
                column_offset=acc_index * tile_n,
            )
            if task >= acc_stages and rank == 0 and cl.elect_sync():
                cl.mbarrier_wait_parity(
                    acc_empty.get_element_pointer(acc_index),
                    acc_phase ^ 1,
                )
            for k_tile in range(k // tile_k):
                stage = load_index % load_stages
                phase = (load_index // load_stages) & 1
                if rank == 0 and cl.elect_sync():
                    cl.mbarrier_wait_parity(
                        load_ready.get_element_pointer(stage), phase
                    )
                    cl.tcgen05_fence_after_thread_sync()
                    a_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=a_smem.get_element_pointer(
                            (stage, consumer, 0)
                        ),
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=8 * 128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                    ).encode()
                    b_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=b_smem.get_element_pointer((stage, 0)),
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=8 * 128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                    ).encode()
                    for atom in cl.static_iter(range(tile_k // 128)):
                        a_atom = a_desc + atom * (cta_m * 128 >> 4)
                        b_atom = b_desc + atom * (cta_n * 128 >> 4)
                        for kk in cl.static_iter(range(128 // MMA_K)):
                            cl.tcgen05_mma(
                                cl.Tcgen05MMAKind.F8F6F4,
                                acc_tmem,
                                a_atom + (MMA_K >> 4) * kk,
                                b_atom + (MMA_K >> 4) * kk,
                                instruction,
                                accumulate=(k_tile != 0 or atom != 0 or kk != 0),
                                cta_group=cl.CTAGroup.CTA_2,
                            )
                    cl.tcgen05_commit(
                        load_empty.get_element_pointer(stage),
                        multicast_mask=0b11,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                load_index += 1
            if rank == 0 and cl.elect_sync():
                cl.tcgen05_commit(
                    acc_ready.get_element_pointer(acc_index),
                    multicast_mask=0b11,
                    cta_group=cl.CTAGroup.CTA_2,
                )
            tile = scheduled_tile
            has_work = scheduled_work
            task += 1

    elif warp < producer_base:
        consumer = warp // 4
        local_warp = warp % 4
        tile = cl.cluster_index(0)
        has_work = True
        task = 0
        while has_work:
            pid_m, pid_n = swizzle_program_id(
                tile,
                tiles_m,
                tiles_n,
                supergroup_size,
            )
            cl.mbarrier_wait_parity(schedule_ready, task & 1)
            scheduled_tile = next_tile[0]
            scheduled_work = next_has_work[0] != 0
            sync_consumer_warpgroup(consumer)
            if local_warp == 0 and cl.elect_sync():
                cl.mbarrier_arrive(schedule_consumed)
            acc_slot = task % acc_stages
            acc_phase = (task // acc_stages) & 1
            acc_index = acc_slot * num_consumers + consumer
            acc_tmem = cl.tcgen05_tmem_offset(
                tmem_storage[0],
                column_offset=acc_index * tile_n,
            )
            store_persistent_partition(
                c_tmap,
                c_smem,
                acc_tmem,
                acc_ready.get_element_pointer(acc_index),
                acc_empty.get_element_pointer(acc_index),
                acc_phase,
                pid_m,
                pid_n,
                rank,
                consumer,
                local_warp,
                lane,
                tile_n,
                num_consumers,
                epilogue_stages,
                num_d_tiles,
                scheduled_work,
            )
            tile = scheduled_tile
            has_work = scheduled_work
            task += 1

    cl.barrier_sync_cluster(aligned=True)
    if warp == producer_base:
        cl.tcgen05_deallocate(
            tmem_storage[0],
            tmem_columns,
            cta_group=cl.CTAGroup.CTA_2,
        )


fp8_b200_gemm_persistent_kernel = cl.kernel(
    max_threads_per_block=(8 * WARP_SIZE,),
    max_blocks_per_cluster=2,
    min_blocks_per_sm=1,
    max_registers_per_thread=256,
)(_fp8_b200_gemm_persistent_kernel)

fp8_b200_gemm_persistent_two_consumer_kernel = cl.kernel(
    max_threads_per_block=(12 * WARP_SIZE,),
    max_blocks_per_cluster=2,
    min_blocks_per_sm=1,
)(_fp8_b200_gemm_persistent_kernel)


def make_fp8_tma_view(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    assert cols % 128 == 0
    return torch.as_strided(
        x,
        size=(128, rows, cols // 128),
        stride=(1, cols, 128),
    )


def prepare_fp8_b200_gemm_b(b):
    return make_fp8_tma_view(b.T.contiguous())


def launch_fp8_b200_gemm(a, b, c, config, stream=None, b_tma_view=None):
    m, k = a.shape
    bk, n = b.shape
    assert bk == k
    assert c.shape == (m, n)
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert c.dtype == torch.bfloat16
    assert m % (256 * config.num_consumers) == 0
    assert n % config.tile_n == 0
    assert k % config.tile_k == 0

    tasks = (m // (256 * config.num_consumers)) * (n // config.tile_n)
    if stream is None:
        stream = torch.cuda.current_stream()
    if b_tma_view is None:
        b_tma_view = prepare_fp8_b200_gemm_b(b)
    kernel = (
        fp8_b200_gemm_persistent_kernel
        if config.num_consumers == 1
        else fp8_b200_gemm_persistent_two_consumer_kernel
    )
    cl.launch(
        stream,
        (tasks * 2,),
        (config.num_warps * WARP_SIZE,),
        kernel,
        (
            make_fp8_tma_view(a),
            b_tma_view,
            c,
            m,
            n,
            k,
            config.tile_n,
            config.tile_k,
            config.supergroup_size,
            config.num_consumers,
            config.load_stages,
            config.epilogue_stages,
        ),
        block_in_cluster_count=(2, 1, 1),
        programmatic_dependent_launch=True,
    )


def benchmark_fp8_b200_gemm(n, config, warmups=5, iterations=10):
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    argument_bytes = 4 * n * n
    eviction_bytes = 3 * properties.L2_cache_size
    groups = (
        1 if argument_bytes > eviction_bytes else eviction_bytes // argument_bytes + 1
    )

    torch.manual_seed(2024)
    arguments = []
    for _ in range(groups):
        a = torch.randn((n, n), dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
        b = torch.randn((n, n), dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
        c = torch.empty((n, n), dtype=torch.bfloat16, device="cuda")
        arguments.append((a, b, c, prepare_fp8_b200_gemm_b(b)))

    for iteration in range(warmups):
        a, b, c, b_tma_view = arguments[iteration % groups]
        launch_fp8_b200_gemm(a, b, c, config, b_tma_view=b_tma_view)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for iteration in range(iterations):
        a, b, c, b_tma_view = arguments[iteration % groups]
        launch_fp8_b200_gemm(a, b, c, config, b_tma_view=b_tma_view)
    stop.record()
    stop.synchronize()

    microseconds = start.elapsed_time(stop) * 1000.0 / iterations
    tflops = 2.0 * n * n * n / microseconds / 1.0e6
    return microseconds, tflops


def main():
    for n, config in BENCHMARK_CONFIGS:
        print(f"N={n}: {config}")
        microseconds, tflops = benchmark_fp8_b200_gemm(n, config)
        print(f"{microseconds:.4f} us, {tflops:.4f} TFLOPs", flush=True)
        torch.cuda.empty_cache()


@pytest.mark.parametrize("m,n,k", ((256, 256, 128), (512, 512, 384)))
def test_fp8_b200_gemm(m, n, k):

    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((k, n), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    c = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    tiles = (m // 256) * (n // 256)

    cl.launch(
        torch.cuda.current_stream(),
        (tiles * 2,),
        (4 * WARP_SIZE,),
        fp8_b200_gemm_kernel,
        (
            make_fp8_tma_view(a),
            make_fp8_tma_view(b.T.contiguous()),
            c,
            m,
            n,
            k,
        ),
        block_in_cluster_count=(2, 1, 1),
    )
    torch.cuda.synchronize()

    reference = a.float() @ b.float()
    torch.testing.assert_close(c.float(), reference, atol=1.0, rtol=2e-2)


@pytest.mark.parametrize(
    "config,m,n,k",
    (
        (CONFIGS[0], 512, 256, 1536),
        (CONFIGS[1], 512, 512, 896),
        (CONFIGS[3], 1024, 512, 640),
    ),
)
def test_fp8_b200_gemm_persistent(config, m, n, k):
    check_fp8_b200_gemm_persistent(config, m, n, k)


def check_fp8_b200_gemm_persistent(config, m, n, k):
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((k, n), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    c = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    launch_fp8_b200_gemm(a, b, c, config)
    torch.cuda.synchronize()

    reference = a.float() @ b.float()
    torch.testing.assert_close(c.float(), reference, atol=1.0, rtol=2e-2)


@pytest.mark.parametrize(
    "n,config",
    BENCHMARK_CONFIGS,
)
def test_fp8_b200_gemm_benchmark_case(n, config):
    check_fp8_b200_gemm_persistent(config, n, n, n)


def test_fp8_b200_gemm_clc_work_stealing():
    config = CONFIGS[1]
    multiprocessors = torch.cuda.get_device_properties(0).multi_processor_count
    tasks = multiprocessors // 2 + 16
    m, n, k = tasks * 256, 256, 128
    torch.manual_seed(1)
    a = torch.randn((m, k), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((k, n), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    c = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    launch_fp8_b200_gemm(a, b, c, config)
    torch.cuda.synchronize()

    reference = a.float() @ b.float()
    torch.testing.assert_close(c.float(), reference, atol=1.0, rtol=2e-2)


if __name__ == "__main__":
    main()
