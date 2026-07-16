# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CTA2 MXFP8 GEMM port of ThunderKittens' ``mxfp8_b200_gemm.cu``."""

from dataclasses import dataclass

import pytest
import torch

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability

if tuple(get_compute_capability()) != (10, 0):
    pytest.skip("Requires Blackwell", allow_module_level=True)

WARP_SIZE = 32
WARPGROUP_SIZE = 4 * WARP_SIZE
TILE_M = 256
CTA_M = TILE_M // 2
TILE_K = 128
MMA_K = 32
SCALE_ROWS = 32
SCALE_COLUMNS = 16
SCALE_TILE_BYTES = SCALE_ROWS * SCALE_COLUMNS
TMEM_COLUMNS = 512
TMEM_A_SCALE_COLUMN = 256
TMEM_B_SCALE_COLUMN = 384


@dataclass(frozen=True)
class MXFP8B200GemmConfig:
    tile_n: int
    load_stages: int
    epilogue_stages: int
    supergroup_size: int
    output_stages: int
    overlap_epilogue: bool

    def __post_init__(self):
        assert self.tile_n in (128, 256)
        assert self.load_stages > 0
        assert self.epilogue_stages > 0
        assert self.supergroup_size > 0
        assert self.output_stages > 0
        assert self.tile_n % self.epilogue_stages == 0
        assert self.epilogue_stages <= 1 or self.output_stages >= 2

    @property
    def num_warps(self):
        return 8

    def __str__(self):
        return (
            f"tile_n={self.tile_n}, load_stages={self.load_stages}, "
            f"epilogue_stages={self.epilogue_stages}, "
            f"supergroup_size={self.supergroup_size}, "
            f"output_stages={self.output_stages}, "
            f"overlap_epilogue={self.overlap_epilogue}"
        )


CONFIGS = (
    MXFP8B200GemmConfig(128, 5, 4, 12, 2, True),
    MXFP8B200GemmConfig(256, 5, 8, 12, 2, True),
    MXFP8B200GemmConfig(256, 5, 8, 8, 2, False),
    MXFP8B200GemmConfig(256, 6, 16, 16, 4, False),
    MXFP8B200GemmConfig(256, 4, 8, 8, 2, False),
)

BENCHMARKS = tuple(zip((1024, 2048, 4096, 8192, 16384), CONFIGS))


def pack_mxfp8_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack logical ``[rows, K/32]`` UE8M0 scales into PTX's 1X layout."""
    rows, k_blocks = scales.shape
    assert scales.dtype == torch.float8_e8m0fnu
    assert rows % 128 == 0
    assert k_blocks % 4 == 0
    return (
        scales.reshape(rows // 128, 4, 32, k_blocks // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(rows // 128, k_blocks // 4, 32, 16)
        .contiguous()
    )


def unpack_mxfp8_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert packed PTX 1X scale storage back to logical scales."""
    row_tiles, k_tiles, scale_rows, scale_columns = scales.shape
    assert scales.dtype == torch.float8_e8m0fnu
    assert (scale_rows, scale_columns) == (32, 16)
    return (
        scales.reshape(row_tiles, k_tiles, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(row_tiles * 128, k_tiles * 4)
        .contiguous()
    )


def make_fp8_tma_view(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    assert columns % 128 == 0
    return torch.as_strided(
        x,
        size=(128, rows, columns // 128),
        stride=(1, columns, 128),
    )


def make_mxfp8_scale_tma_view(scales: torch.Tensor) -> torch.Tensor:
    row_tiles, k_tiles, scale_rows, scale_columns = scales.shape
    assert scales.dtype == torch.float8_e8m0fnu
    assert (scale_rows, scale_columns) == (32, 16)
    return torch.as_strided(
        scales,
        size=(16, 32, k_tiles, row_tiles),
        stride=(1, 16, 32 * 16, k_tiles * 32 * 16),
    )


def swizzle_program_id(tile, tiles_m, tiles_n, width):
    tiles_per_group = width * tiles_n
    group = tile // tiles_per_group
    within_group = tile % tiles_per_group
    rows = cl.minimum(width, tiles_m - group * width)
    pid_m = group * width + within_group % rows
    pid_n = within_group // rows
    return pid_m, pid_n


def sync_consumer_warpgroup():
    cl.barrier_sync_block(WARPGROUP_SIZE, 1, aligned=False)


def load_tmem_bf16_subtile(registers, register_offset, tmem, warp, column, width):
    source = cl.tcgen05_tmem_offset(
        tmem,
        lane_offset=warp * WARP_SIZE,
        column_offset=column,
    )
    values = cl.tcgen05_load(
        cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
        source,
        count=width,
    )
    cl.tcgen05_wait_load()
    for pair in cl.static_iter(range(width // 2)):
        lo = cl.bitcast(values[pair * 2], cl.float32)
        hi = cl.bitcast(values[pair * 2 + 1], cl.float32)
        packed = cl._nvvm.ff2bf16x2_rn(hi, lo)
        registers[register_offset + pair * 2] = packed[0]
        registers[register_offset + pair * 2 + 1] = packed[1]


def store_bf16_register_subtile(dst, registers, register_offset, row, width):
    for pair in cl.static_iter(range(width // 2)):
        packed = cl.Vector(
            registers[register_offset + pair * 2],
            registers[register_offset + pair * 2 + 1],
            dtype=cl.bfloat16,
        )
        (dst + row * width + pair * 2).store(
            packed,
            alignment=4,
        )


def issue_output_store(c_tmap, c_smem, pid_m, pid_n, rank, subtile, width):
    if cl.elect_sync():
        cl.fence_proxy(
            cl.FenceProxyKind.ASYNC_SHARED,
            space=cl.MemorySpace.SHARED,
        )
        cache_hint = cl.create_fractional_cache_policy(cl.CachePolicy.L2_EVICT_FIRST)
        cl.copy_async_bulk_tensor_shared_to_global(
            c_smem,
            c_tmap,
            (
                pid_n + subtile * width,
                pid_m + rank * CTA_M,
            ),
            l2_cache_hint=cache_hint,
        )
        cl.copy_async_bulk_commit_group()


def store_output_tile(
    c_tmap,
    c_smem,
    tmem,
    output_ready,
    output_empty,
    phase,
    pid_m,
    pid_n,
    rank,
    local_warp,
    lane,
    tile_n,
    epilogue_stages,
    output_stages,
    overlap_epilogue,
):
    width = tile_n // epilogue_stages
    row = local_warp * WARP_SIZE + lane
    cl.mbarrier_wait_parity(output_ready, phase)
    cl.tcgen05_fence_after_thread_sync()

    if overlap_epilogue:
        with cl.local_array(width, cl.bfloat16) as registers:
            for subtile in cl.static_iter(range(epilogue_stages)):
                stage = subtile % output_stages
                stage_smem = c_smem.get_element_pointer((stage, 0))
                if local_warp == 0 and cl.elect_sync():
                    cl.copy_async_bulk_wait_group(output_stages - 1, read=True)
                sync_consumer_warpgroup()
                load_tmem_bf16_subtile(
                    registers,
                    0,
                    tmem,
                    rank * 4 + local_warp,
                    subtile * width,
                    width,
                )
                store_bf16_register_subtile(stage_smem, registers, 0, row, width)
                if subtile == epilogue_stages - 1 and cl.elect_sync():
                    empty = cl.map_shared_to_cluster(output_empty, 0)
                    cl.mbarrier_arrive(empty, scope=cl.MbarrierScope.BLOCK)
                sync_consumer_warpgroup()
                if local_warp == 0:
                    issue_output_store(
                        c_tmap,
                        stage_smem,
                        pid_m,
                        pid_n,
                        rank,
                        subtile,
                        width,
                    )
    else:
        with cl.local_array(
            epilogue_stages * width,
            cl.bfloat16,
        ) as registers:
            for subtile in cl.static_iter(range(epilogue_stages)):
                load_tmem_bf16_subtile(
                    registers,
                    subtile * width,
                    tmem,
                    rank * 4 + local_warp,
                    subtile * width,
                    width,
                )
            if cl.elect_sync():
                empty = cl.map_shared_to_cluster(output_empty, 0)
                cl.mbarrier_arrive(empty, scope=cl.MbarrierScope.BLOCK)
            sync_consumer_warpgroup()
            for subtile in cl.static_iter(range(epilogue_stages)):
                stage = subtile % output_stages
                stage_smem = c_smem.get_element_pointer((stage, 0))
                if local_warp == 0 and cl.elect_sync():
                    cl.copy_async_bulk_wait_group(output_stages - 1, read=True)
                sync_consumer_warpgroup()
                store_bf16_register_subtile(
                    stage_smem,
                    registers,
                    subtile * width,
                    row,
                    width,
                )
                sync_consumer_warpgroup()
                if local_warp == 0:
                    issue_output_store(
                        c_tmap,
                        stage_smem,
                        pid_m,
                        pid_n,
                        rank,
                        subtile,
                        width,
                    )


@cl.kernel(
    max_threads_per_block=(8 * WARP_SIZE,),
    max_blocks_per_cluster=2,
    min_blocks_per_sm=1,
    max_registers_per_thread=256,
)
def mxfp8_b200_gemm_kernel(
    a,
    a_scales,
    b,
    b_scales,
    c,
    m: cl.Constant[int],
    n: cl.Constant[int],
    k: cl.Constant[int],
    tile_n: cl.Constant[int],
    load_stages: cl.Constant[int],
    epilogue_stages: cl.Constant[int],
    supergroup_size: cl.Constant[int],
    output_stages: cl.Constant[int],
    overlap_epilogue: cl.Constant[bool],
):
    tid = cl.thread_index(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    rank = cl.block_in_cluster_index(0)
    cta_n = tile_n // 2
    b_scale_tiles = tile_n // 128
    output_width = tile_n // epilogue_stages
    tiles_m = m // TILE_M
    tiles_n = n // tile_n
    tasks = tiles_m * tiles_n
    mma_warp = 4
    scale_loader_warp = 6
    data_loader_warp = 7

    a_tmap = cl.tensor_map_tiled(
        a,
        (128, CTA_M, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    b_tmap = cl.tensor_map_tiled(
        b,
        (128, cta_n, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    a_scale_tmap = cl.tensor_map_tiled(a_scales, (16, 32, 1, 1))
    b_scale_tmap = cl.tensor_map_tiled(b_scales, (16, 32, 1, 1))
    c_tmap = cl.tensor_map_tiled(c, (output_width, CTA_M), order="F")
    if tid == 0:
        cl.prefetch_tensor_map(a_tmap)
        cl.prefetch_tensor_map(a_scale_tmap)
        cl.prefetch_tensor_map(b_tmap)
        cl.prefetch_tensor_map(b_scale_tmap)
        cl.prefetch_tensor_map(c_tmap)

    a_smem = cl.shared_array(
        (load_stages, CTA_M * TILE_K),
        cl.uint8,
        alignment=512,
    )
    b_smem = cl.shared_array(
        (load_stages, cta_n * TILE_K),
        cl.uint8,
        alignment=512,
    )
    a_scale_smem = cl.shared_array(
        (load_stages, SCALE_TILE_BYTES),
        cl.uint8,
        alignment=128,
    )
    b_scale_smem = cl.shared_array(
        (load_stages, b_scale_tiles, SCALE_TILE_BYTES),
        cl.uint8,
        alignment=128,
    )
    c_smem = cl.shared_array(
        (output_stages, CTA_M * output_width),
        cl.bfloat16,
        alignment=128,
    )
    data_ready = cl.shared_array(load_stages, cl.mbarrier, alignment=8)
    scale_ready = cl.shared_array(load_stages, cl.mbarrier, alignment=8)
    input_empty = cl.shared_array(load_stages, cl.mbarrier, alignment=8)
    output_ready = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    output_empty = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    tmem_storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR),
        alignment=4,
    )

    if warp == 0 and cl.elect_sync():
        for stage in cl.static_iter(range(load_stages)):
            cl.mbarrier_initialize(data_ready.get_element_pointer(stage), 2)
            cl.mbarrier_initialize(scale_ready.get_element_pointer(stage), 2)
            cl.mbarrier_initialize(input_empty.get_element_pointer(stage), 1)
        cl.mbarrier_initialize(output_ready, 1)
        cl.mbarrier_initialize(output_empty, 8)
        cl.fence_mbarrier_initialize()
    cl.barrier_sync_cluster(aligned=True)

    if warp == mma_warp:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(),
            TMEM_COLUMNS,
            cta_group=cl.CTAGroup.CTA_2,
        )
    cl.barrier_sync_cluster(aligned=True)
    tmem = tmem_storage[0]

    if warp == data_loader_warp:
        if cl.elect_sync():
            cl.grid_dependency_control_wait()
        task = cl.cluster_index(0)
        load_index = 0
        while task < tasks:
            pid_m, pid_n = swizzle_program_id(
                task,
                tiles_m,
                tiles_n,
                supergroup_size,
            )
            for k_tile in range(k // TILE_K):
                stage = load_index % load_stages
                phase = (load_index // load_stages) & 1
                if load_index >= load_stages:
                    cl.mbarrier_wait_parity(
                        input_empty.get_element_pointer(stage),
                        phase ^ 1,
                    )
                if cl.elect_sync():
                    ready = data_ready.get_element_pointer(stage)
                    arrive = cl.map_shared_to_leader_block(ready)
                    expect = cl.map_shared_to_cluster(ready, 0)
                    a_dst = cl.map_shared_to_cluster(
                        a_smem.get_element_pointer((stage, 0)),
                        rank,
                    )
                    b_dst = cl.map_shared_to_cluster(
                        b_smem.get_element_pointer((stage, 0)),
                        rank,
                    )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        a_tmap,
                        (0, pid_m * TILE_M + rank * CTA_M, k_tile),
                        a_dst,
                        arrive,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        b_tmap,
                        (0, pid_n * tile_n + rank * cta_n, k_tile),
                        b_dst,
                        arrive,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                    cl.mbarrier_arrive_expect_transaction(
                        expect,
                        (CTA_M + cta_n) * TILE_K,
                        scope=cl.MbarrierScope.BLOCK,
                    )
                load_index += 1
            task += cl.cluster_count(0)

    elif warp == scale_loader_warp:
        if cl.elect_sync():
            cl.grid_dependency_control_wait()
        task = cl.cluster_index(0)
        load_index = 0
        while task < tasks:
            pid_m, pid_n = swizzle_program_id(
                task,
                tiles_m,
                tiles_n,
                supergroup_size,
            )
            for k_tile in range(k // TILE_K):
                stage = load_index % load_stages
                phase = (load_index // load_stages) & 1
                if load_index >= load_stages:
                    cl.mbarrier_wait_parity(
                        input_empty.get_element_pointer(stage),
                        phase ^ 1,
                    )
                if cl.elect_sync():
                    ready = scale_ready.get_element_pointer(stage)
                    arrive = cl.map_shared_to_leader_block(ready)
                    expect = cl.map_shared_to_cluster(ready, 0)
                    a_dst = cl.map_shared_to_cluster(
                        a_scale_smem.get_element_pointer((stage, 0)),
                        rank,
                    )
                    cl.copy_async_bulk_tensor_global_to_shared(
                        a_scale_tmap,
                        (0, 0, k_tile, pid_m * 2 + rank),
                        a_dst,
                        arrive,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                    expected_bytes = SCALE_TILE_BYTES
                    if tile_n == 256:
                        b_dst = cl.map_shared_to_cluster(
                            b_scale_smem.get_element_pointer((stage, rank, 0)),
                            rank,
                        )
                        cl.copy_async_bulk_tensor_global_to_shared(
                            b_scale_tmap,
                            (0, 0, k_tile, pid_n * 2 + rank),
                            b_dst,
                            arrive,
                            cta_group=cl.CTAGroup.CTA_2,
                            multicast_mask=0b11,
                        )
                        expected_bytes += 2 * SCALE_TILE_BYTES
                    elif rank == 0:
                        b_dst = cl.map_shared_to_cluster(
                            b_scale_smem.get_element_pointer((stage, 0, 0)),
                            0,
                        )
                        cl.copy_async_bulk_tensor_global_to_shared(
                            b_scale_tmap,
                            (0, 0, k_tile, pid_n),
                            b_dst,
                            arrive,
                            cta_group=cl.CTAGroup.CTA_2,
                            multicast_mask=0b11,
                        )
                        expected_bytes += 2 * SCALE_TILE_BYTES
                    cl.mbarrier_arrive_expect_transaction(
                        expect,
                        expected_bytes,
                        scope=cl.MbarrierScope.BLOCK,
                    )
                load_index += 1
            task += cl.cluster_count(0)

    elif warp == mma_warp and rank == 0:
        task = cl.cluster_index(0)
        iteration = 0
        load_index = 0
        while task < tasks:
            output_phase = iteration & 1
            if iteration != 0 and cl.elect_sync():
                cl.mbarrier_wait_parity(output_empty, output_phase ^ 1)
            for k_tile in range(k // TILE_K):
                stage = load_index % load_stages
                phase = (load_index // load_stages) & 1
                if cl.elect_sync():
                    cl.mbarrier_wait_parity(
                        scale_ready.get_element_pointer(stage),
                        phase,
                    )
                    a_scale_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=a_scale_smem.get_element_pointer(
                            (stage, 0)
                        ),
                        leading_dimension_byte_offset=128,
                        stride_dimension_byte_offset=128,
                    ).encode()
                    b_scale_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=b_scale_smem.get_element_pointer(
                            (stage, 0, 0)
                        ),
                        leading_dimension_byte_offset=128,
                        stride_dimension_byte_offset=128,
                    ).encode()
                    a_scale_tmem = cl.tcgen05_tmem_offset(
                        tmem,
                        column_offset=TMEM_A_SCALE_COLUMN + stage * 4,
                    )
                    b_scale_tmem = cl.tcgen05_tmem_offset(
                        tmem,
                        column_offset=TMEM_B_SCALE_COLUMN + stage * 8,
                    )
                    cl.tcgen05_copy(
                        a_scale_tmem,
                        a_scale_desc,
                        shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                        cta_group=cl.CTAGroup.CTA_2,
                        multicast=cl.Tcgen05CopyMulticast.WARPX4,
                    )
                    cl.tcgen05_copy(
                        b_scale_tmem,
                        b_scale_desc,
                        shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                        cta_group=cl.CTAGroup.CTA_2,
                        multicast=cl.Tcgen05CopyMulticast.WARPX4,
                    )
                    if b_scale_tiles == 2:
                        b_scale_desc_1 = cl.Tcgen05SharedMemoryDescriptor(
                            matrix_start_address=b_scale_smem.get_element_pointer(
                                (stage, 1, 0)
                            ),
                            leading_dimension_byte_offset=128,
                            stride_dimension_byte_offset=128,
                        ).encode()
                        cl.tcgen05_copy(
                            cl.tcgen05_tmem_offset(b_scale_tmem, column_offset=4),
                            b_scale_desc_1,
                            shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                            cta_group=cl.CTAGroup.CTA_2,
                            multicast=cl.Tcgen05CopyMulticast.WARPX4,
                        )

                    cl.mbarrier_wait_parity(
                        data_ready.get_element_pointer(stage),
                        phase,
                    )
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
                    for kk in cl.static_iter(range(TILE_K // MMA_K)):
                        instruction = cl.Tcgen05Mxf8f6f4InstructionDescriptor(
                            a_type=(cl.Tcgen05Mxf8f6f4InstructionDescriptor.Type.E4M3),
                            b_type=(cl.Tcgen05Mxf8f6f4InstructionDescriptor.Type.E4M3),
                            n=tile_n,
                            m=TILE_M,
                            a_scale_id=kk,
                            b_scale_id=kk,
                        ).encode()
                        cl.tcgen05_mma_block_scale(
                            cl.Tcgen05MMABlockScaleKind.MXF8F6F4,
                            tmem,
                            a_desc + (MMA_K >> 4) * kk,
                            b_desc + (MMA_K >> 4) * kk,
                            instruction,
                            a_scale_tmem,
                            b_scale_tmem,
                            accumulate=k_tile != 0 or kk != 0,
                            cta_group=cl.CTAGroup.CTA_2,
                        )
                    cl.tcgen05_commit(
                        input_empty.get_element_pointer(stage),
                        multicast_mask=0b11,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                load_index += 1
            if cl.elect_sync():
                cl.tcgen05_commit(
                    output_ready,
                    multicast_mask=0b11,
                    cta_group=cl.CTAGroup.CTA_2,
                )
            task += cl.cluster_count(0)
            iteration += 1

    elif warp < 4:
        local_warp = warp
        task = cl.cluster_index(0)
        iteration = 0
        while task < tasks:
            pid_m, pid_n = swizzle_program_id(
                task,
                tiles_m,
                tiles_n,
                supergroup_size,
            )
            store_output_tile(
                c_tmap,
                c_smem,
                tmem,
                output_ready,
                output_empty,
                iteration & 1,
                pid_m * TILE_M,
                pid_n * tile_n,
                rank,
                local_warp,
                lane,
                tile_n,
                epilogue_stages,
                output_stages,
                overlap_epilogue,
            )
            task += cl.cluster_count(0)
            iteration += 1
        sync_consumer_warpgroup()
        if cl.elect_sync():
            cl.grid_dependency_control_launch_dependents()

    cl.barrier_sync_cluster(aligned=True)
    if warp == mma_warp:
        cl.tcgen05_deallocate(
            tmem,
            TMEM_COLUMNS,
            cta_group=cl.CTAGroup.CTA_2,
        )


def launch_mxfp8_b200_gemm(
    a,
    a_scales,
    b,
    b_scales,
    c,
    config,
    stream=None,
):
    m, k = a.shape
    n, bk = b.shape
    assert bk == k
    assert c.shape == (m, n)
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a_scales.dtype == b_scales.dtype == torch.float8_e8m0fnu
    assert a_scales.shape == (m // 128, k // 128, 32, 16)
    assert b_scales.shape == (n // 128, k // 128, 32, 16)
    assert c.dtype == torch.bfloat16
    assert m % TILE_M == 0
    assert n % config.tile_n == 0
    assert k % TILE_K == 0

    tasks = (m // TILE_M) * (n // config.tile_n)
    multiprocessors = torch.cuda.get_device_properties(a.device).multi_processor_count
    blocks = min(tasks * 2, multiprocessors - multiprocessors % 2)
    if stream is None:
        stream = torch.cuda.current_stream()
    cl.launch(
        stream,
        (blocks,),
        (config.num_warps * WARP_SIZE,),
        mxfp8_b200_gemm_kernel,
        (
            make_fp8_tma_view(a),
            make_mxfp8_scale_tma_view(a_scales),
            make_fp8_tma_view(b),
            make_mxfp8_scale_tma_view(b_scales),
            c,
            m,
            n,
            k,
            config.tile_n,
            config.load_stages,
            config.epilogue_stages,
            config.supergroup_size,
            config.output_stages,
            config.overlap_epilogue,
        ),
        block_in_cluster_count=(2, 1, 1),
        programmatic_dependent_launch=True,
    )


def reference_mxfp8_gemm(a, a_scales, b, b_scales):
    a_values = a.float() * a_scales.float().repeat_interleave(32, dim=1)
    b_values = b.float() * b_scales.float().repeat_interleave(32, dim=1)
    return a_values @ b_values.T


def make_test_scales(rows, k, nonuniform=False):
    if nonuniform:
        row = torch.arange(rows, device="cuda")[:, None]
        block = torch.arange(k // 32, device="cuda")[None, :]
        exponents = (row + block) % 5 - 2
        return torch.exp2(exponents.float()).to(torch.float8_e8m0fnu)
    return torch.ones((rows, k // 32), dtype=torch.float8_e8m0fnu, device="cuda")


def check_mxfp8_b200_gemm(config, m, n, k, nonuniform=False):
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((n, k), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    a_scales = make_test_scales(m, k, nonuniform)
    b_scales = make_test_scales(n, k, nonuniform)
    packed_a_scales = pack_mxfp8_scales(a_scales)
    packed_b_scales = pack_mxfp8_scales(b_scales)
    c = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    launch_mxfp8_b200_gemm(
        a,
        packed_a_scales,
        b,
        packed_b_scales,
        c,
        config,
    )
    torch.cuda.synchronize()

    reference = reference_mxfp8_gemm(a, a_scales, b, b_scales)
    torch.testing.assert_close(c.float(), reference, atol=2.0, rtol=2e-2)


def benchmark_mxfp8_b200_gemm(n, config, warmups=5, iterations=10):
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    argument_bytes = 4 * n * n + 2 * n * n // 32
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
        a_scale_values = torch.randint(-3, 4, (n, n // 32), device="cuda")
        b_scale_values = torch.randint(-3, 4, (n, n // 32), device="cuda")
        a_scales = pack_mxfp8_scales(
            torch.exp2(a_scale_values.float()).to(torch.float8_e8m0fnu)
        )
        b_scales = pack_mxfp8_scales(
            torch.exp2(b_scale_values.float()).to(torch.float8_e8m0fnu)
        )
        c = torch.empty((n, n), dtype=torch.bfloat16, device="cuda")
        arguments.append((a, a_scales, b, b_scales, c))

    for iteration in range(warmups):
        launch_mxfp8_b200_gemm(*arguments[iteration % groups], config)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for iteration in range(iterations):
        launch_mxfp8_b200_gemm(*arguments[iteration % groups], config)
    stop.record()
    stop.synchronize()

    microseconds = start.elapsed_time(stop) * 1000.0 / iterations
    tflops = 2.0 * n * n * n / microseconds / 1.0e6
    return microseconds, tflops


def main():
    for n, config in BENCHMARKS:
        print(f"N={n}: {config}")
        microseconds, tflops = benchmark_mxfp8_b200_gemm(n, config)
        print(f"{microseconds:.4f} us, {tflops:.4f} TFLOPs", flush=True)
        torch.cuda.empty_cache()


def test_mxfp8_scale_layout_round_trip():
    logical = make_test_scales(256, 256, nonuniform=True)
    torch.testing.assert_close(unpack_mxfp8_scales(pack_mxfp8_scales(logical)), logical)


@pytest.mark.parametrize(
    "config,n",
    (
        (CONFIGS[0], 128),
        (CONFIGS[1], 256),
    ),
)
def test_mxfp8_b200_gemm_nonuniform_scales(config, n):
    check_mxfp8_b200_gemm(config, 256, n, 256, True)


@pytest.mark.parametrize(
    "config,m,n,k",
    (
        (CONFIGS[0], 512, 256, 768),
        (CONFIGS[1], 512, 512, 896),
        (CONFIGS[2], 512, 512, 640),
        (CONFIGS[3], 512, 512, 896),
        (CONFIGS[4], 512, 512, 640),
    ),
)
def test_mxfp8_b200_gemm_pipeline(config, m, n, k):
    check_mxfp8_b200_gemm(config, m, n, k)


@pytest.mark.parametrize(
    "n,config",
    BENCHMARKS,
)
def test_mxfp8_b200_gemm_benchmark_case(n, config):
    check_mxfp8_b200_gemm(config, n, n, n)


if __name__ == "__main__":
    main()
