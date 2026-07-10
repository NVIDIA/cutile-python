# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._compile import get_compute_capability
import cuda.lang as cl
import torch
import pytest

__doc__ = """
Based on:
https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/14-multicta.py

There are some notable differences between these tests and the Gluon tutorial
mostly due to the fact that Gluon borrows some higher level concepts from
triton.
Users must handle these themselves in cuda.lang at the time of writing.

- Gluon uses layouts like gl.BlockedLayout or gl.NVMMASharedLayout.
- gl.max and gl.sum handles distributed reductions.
- gl.warp_specialize.

"""

cc = get_compute_capability()
if tuple(cc) != (10, 0):
    pytest.skip("Requires blackwell", True)

WARP_SIZE = 32
MMA_K = 16


def warp_reduce_max(value):
    for offset in cl.static_iter((16, 8, 4, 2, 1)):
        value = cl._libdevice.fmaxf(value, cl.shfl_down_sync(value, offset))
    return value


def warp_reduce_sum(value):
    for offset in cl.static_iter((16, 8, 4, 2, 1)):
        value += cl.shfl_down_sync(value, offset)
    return value


def block_reduce_max(value, warp_values, num_warps):
    tid = cl.thread_index(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    value = warp_reduce_max(value)
    if lane == 0:
        warp_values[warp] = value
    cl.barrier_sync_block()
    if tid == 0:
        value = warp_values[0]
        for index in cl.static_iter(range(1, num_warps)):
            value = cl._libdevice.fmaxf(value, warp_values[index])
        warp_values[0] = value
    cl.barrier_sync_block()
    return warp_values[0]


def block_reduce_sum(value, warp_values, num_warps):
    tid = cl.thread_index(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    value = warp_reduce_sum(value)
    if lane == 0:
        warp_values[warp] = value
    cl.barrier_sync_block()
    if tid == 0:
        value = warp_values[0]
        for index in cl.static_iter(range(1, num_warps)):
            value += warp_values[index]
        warp_values[0] = value
    cl.barrier_sync_block()
    return warp_values[0]


@cl.kernel
def multicta_softmax_kernel(
    x,
    out,
    n: cl.Constant[int],
    num_ctas: cl.Constant[int],
    num_warps: cl.Constant[int],
):
    block_threads = num_warps * WARP_SIZE
    warp_values = cl.shared_array(num_warps, cl.float32)
    cluster_values = cl.shared_array(2, cl.float32)

    tid = cl.thread_index(0)
    row = cl.cluster_index(0)
    rank = cl.block_in_cluster_index(0)
    col_start = rank * block_threads + tid
    col_stride = num_ctas * block_threads

    local_max = cl.float32(-float("inf"))
    for col in range(col_start, n, col_stride):
        local_max = cl._libdevice.fmaxf(local_max, x[row, col])
    cta_max = block_reduce_max(local_max, warp_values, num_warps)
    if tid == 0:
        cluster_values[0] = cta_max
    cl.barrier_sync_block()
    cl.barrier_sync_cluster()

    if rank == 0 and tid == 0:
        row_max = cl.float32(-float("inf"))
        for peer in cl.static_iter(range(num_ctas)):
            peer_values = cl.map_shared_to_cluster(
                cluster_values.get_base_pointer(), peer
            )
            row_max = cl._libdevice.fmaxf(row_max, peer_values.load())
        cluster_values[0] = row_max
    cl.barrier_sync_cluster()
    if tid == 0:
        root_values = cl.map_shared_to_cluster(cluster_values.get_base_pointer(), 0)
        cluster_values[1] = root_values.load()
    cl.barrier_sync_block()
    row_max = cluster_values[1]

    local_sum = cl.float32(0.0)
    for col in range(col_start, n, col_stride):
        value = cl._libdevice.expf(x[row, col] - row_max)
        out[row, col] = value
        local_sum += value
    cta_sum = block_reduce_sum(local_sum, warp_values, num_warps)
    if tid == 0:
        cluster_values[0] = cta_sum
    cl.barrier_sync_block()
    cl.barrier_sync_cluster()

    if rank == 0 and tid == 0:
        row_sum = cl.float32(0.0)
        for peer in cl.static_iter(range(num_ctas)):
            peer_values = cl.map_shared_to_cluster(
                cluster_values.get_base_pointer(), peer
            )
            row_sum += peer_values.load()
        cluster_values[0] = row_sum
    cl.barrier_sync_cluster()
    if tid == 0:
        root_values = cl.map_shared_to_cluster(cluster_values.get_base_pointer(), 0)
        cluster_values[1] = root_values.load()
    cl.barrier_sync_block()
    row_sum = cluster_values[1]

    for col in range(col_start, n, col_stride):
        out[row, col] /= row_sum


def pick_multicta_softmax_config(n):
    num_warps = 8 if n >= 64 * 1024 else 4
    thresholds = ((16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8))
    num_ctas = next((value for limit, value in thresholds if n <= limit), 16)
    return num_warps, num_ctas


@pytest.mark.parametrize("m,n", ((64, 64), (64, 256), (16, 2**16)))
def test_multicta_softmax(m, n):
    torch.manual_seed(0)
    num_warps, num_ctas = pick_multicta_softmax_config(n)
    x = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    cl.launch(
        torch.cuda.current_stream(),
        (m * num_ctas,),
        (num_warps * WARP_SIZE,),
        multicta_softmax_kernel,
        (x, out, n, num_ctas, num_warps),
        block_in_cluster_count=(num_ctas, 1, 1),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.cpu(), torch.softmax(x.cpu(), dim=1), rtol=1e-5, atol=1e-5
    )


@cl.kernel
def tma_multicast_copy_kernel(
    inp,
    out,
    tile_m: cl.Constant[int],
    tile_n: cl.Constant[int],
    num_ctas: cl.Constant[int],
):
    tile_elements = tile_m * tile_n
    tile_bytes = tile_elements * 2
    smem = cl.shared_array(tile_elements, cl.float16, alignment=128)
    mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    in_tmap = cl.tensor_map_tiled(inp, (tile_n, tile_m), order="F")
    out_tmap = cl.tensor_map_tiled(out, (tile_n, tile_m), order="F")

    tid = cl.thread_index(0)
    rank = cl.block_in_cluster_index(0)
    if tid == 0:
        cl.mbarrier_initialize(mbar, 1)
    cl.fence_mbarrier_initialize()
    cl.barrier_sync_cluster()

    if tid == 0:
        cl.mbarrier_arrive_expect_transaction(mbar, tile_bytes)
    cl.barrier_sync_cluster()

    if rank == 0 and tid == 0:
        cluster_smem = cl.map_shared_to_cluster(smem.get_base_pointer(), 0)
        cl.copy_async_bulk_tensor_global_to_shared(
            in_tmap,
            (0, 0),
            cluster_smem,
            mbar,
            multicast_mask=(1 << num_ctas) - 1,
        )

    while not cl.mbarrier_try_wait_parity(mbar, 0, time_hint=10_000):
        pass
    cl.barrier_sync_block()

    if rank == 0 and tid == 0:
        cl.copy_async_bulk_tensor_shared_to_global(
            smem.get_base_pointer(), out_tmap, (0, 0)
        )
        cl.copy_async_bulk_commit_group()
        cl.copy_async_bulk_wait_group(0)


def test_tma_multicast_copy():
    tile_m = tile_n = 128
    num_ctas = 2
    torch.manual_seed(0)
    inp = torch.randn((tile_m, tile_n), dtype=torch.float16, device="cuda")
    out = torch.empty_like(inp)
    cl.launch(
        torch.cuda.current_stream(),
        (num_ctas,),
        (128,),
        tma_multicast_copy_kernel,
        (inp, out, tile_m, tile_n, num_ctas),
        block_in_cluster_count=(num_ctas, 1, 1),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, inp, rtol=0, atol=0)


def wait_mbarrier(mbar, phase):
    while not cl.mbarrier_try_wait_parity(mbar, phase, time_hint=10_000):
        pass


def tensor_memory_pointer(base, lane_offset, column_offset):
    dtype = base.pointee_dtype
    pointer_dtype = cl.pointer_dtype(dtype, cl.MemorySpace.TENSOR)
    address = cl.bitcast(base, cl.uint32)
    offset = (cl.uint32(lane_offset) << 16) + cl.uint32(column_offset)
    return cl.bitcast(address + offset, pointer_dtype)


def p3_to_u64(pointer):
    return cl.uint64(cl.bitcast(pointer, cl.uint32))


def store_fp16_tmem_tile(dst, tmem_base, warp, column, row, output_column, n):
    tmem = tensor_memory_pointer(tmem_base, warp * WARP_SIZE, column)
    regs = cl.tcgen05_load(cl.Tcgen05LoadStoreShape.SHAPE_32X32B, tmem, count=16)
    cl.tcgen05_wait_load()
    for pair in cl.static_iter(range(8)):
        lo = cl.bitcast(regs[pair * 2], cl.float32)
        hi = cl.bitcast(regs[pair * 2 + 1], cl.float32)
        packed = cl._nvvm.ff2f16x2_rn(hi, lo)
        (dst + row * n + output_column + pair * 2).store(packed, alignment=4)


@cl.kernel
def two_cta_tcgen05_kernel(a, b, c):
    cta_m = 128
    cta_n = 64
    tile_n = 128
    tile_k = 64
    num_warps = 4

    tid = cl.thread_index(0)
    warp = tid // WARP_SIZE
    rank = cl.block_in_cluster_index(0)

    a_tmap = cl.tensor_map_tiled(a, (64, cta_m, 1), swizzle=cl.SwizzleMode.SWIZZLE_128B)
    b_tmap = cl.tensor_map_tiled(b, (64, cta_n, 1), swizzle=cl.SwizzleMode.SWIZZLE_128B)
    c_tmap = cl.tensor_map_tiled(c, (tile_n, cta_m), order="F")
    a_smem = cl.shared_array(cta_m * tile_k, cl.float16, alignment=512)
    b_smem = cl.shared_array(cta_n * tile_k, cl.float16, alignment=512)
    c_smem = cl.shared_array(cta_m * tile_n, cl.float16, alignment=128)
    tma_bar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    mma_bar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    tmem_storage = cl.shared_array(
        1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
    )

    if warp == 0 and cl.elect_sync():
        cl.mbarrier_initialize(tma_bar, 2)
        cl.mbarrier_initialize(mma_bar, 1)
        cl.fence_mbarrier_initialize()
    cl.barrier_sync_cluster(aligned=True)

    if warp == 1 and cl.elect_sync():
        a_dst = cl.map_shared_to_cluster(a_smem.get_base_pointer(), rank)
        b_dst = cl.map_shared_to_cluster(b_smem.get_base_pointer(), rank)
        tma_bar_address = cl.bitcast(tma_bar, cl.uint32) & 0xFEFFFFFF
        tma_arrive_bar = cl.bitcast(
            tma_bar_address,
            cl.pointer_dtype(cl.mbarrier, cl.MemorySpace.SHARED),
        )
        tma_expect_bar = cl.map_shared_to_cluster(tma_bar, 0)
        cl.copy_async_bulk_tensor_global_to_shared(
            a_tmap,
            (0, rank * cta_m, 0),
            a_dst,
            tma_arrive_bar,
            cta_group=cl.CTAGroup.CTA_2,
        )
        cl.copy_async_bulk_tensor_global_to_shared(
            b_tmap,
            (0, rank * cta_n, 0),
            b_dst,
            tma_arrive_bar,
            cta_group=cl.CTAGroup.CTA_2,
        )
        cl.mbarrier_arrive_expect_transaction(
            tma_expect_bar,
            (cta_m + cta_n) * tile_k * 2,
            scope=cl.MbarrierScope.BLOCK,
        )

    if warp == 0:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(),
            tile_n,
            cta_group=cl.CTAGroup.CTA_2,
        )
        if rank == 0 and cl.elect_sync():
            wait_mbarrier(tma_bar, 0)
            cl.tcgen05_fence_after_thread_sync()
            a_desc = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=p3_to_u64(a_smem.get_base_pointer()),
                leading_dimension_byte_offset=16,
                stride_dimension_byte_offset=8 * 128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
            ).encode()
            b_desc = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=p3_to_u64(b_smem.get_base_pointer()),
                leading_dimension_byte_offset=16,
                stride_dimension_byte_offset=8 * 128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
            ).encode()
            instruction = cl.Tcgen05InstructionDescriptor(
                d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
                a_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
                b_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
                n=tile_n,
                m=2 * cta_m,
            ).encode()
            for kk in cl.static_iter(range(tile_k // MMA_K)):
                cl.tcgen05_mma(
                    cl.Tcgen05MMAKind.F16,
                    tmem_storage[0],
                    a_desc + (32 >> 4) * kk,
                    b_desc + (32 >> 4) * kk,
                    instruction,
                    accumulate=kk != 0,
                    cta_group=cl.CTAGroup.CTA_2,
                )
            cl.tcgen05_commit(
                mma_bar,
                multicast_mask=0b11,
                cta_group=cl.CTAGroup.CTA_2,
            )

    wait_mbarrier(mma_bar, 0)
    cl.barrier_sync_block()
    cl.tcgen05_fence_after_thread_sync()
    for column in cl.static_iter(range(0, tile_n, 16)):
        store_fp16_tmem_tile(
            c_smem.get_base_pointer(),
            tmem_storage[0],
            rank * num_warps + warp,
            column,
            tid,
            column,
            tile_n,
        )
    cl.barrier_sync_block()
    if tid == 0:
        cl.fence_proxy(cl.FenceProxyKind.ASYNC_SHARED, space=cl.MemorySpace.SHARED)
        cl.copy_async_bulk_tensor_shared_to_global(
            c_smem.get_base_pointer(), c_tmap, (0, rank * cta_m)
        )
        cl.copy_async_bulk_commit_group()
        cl.copy_async_bulk_wait_group(0)

    cl.barrier_sync_cluster(aligned=True)
    if warp == 0:
        cl.tcgen05_deallocate(tmem_storage[0], tile_n, cta_group=cl.CTAGroup.CTA_2)


def test_two_cta_tcgen05():
    m, n, k = 256, 128, 64
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    c = torch.empty((m, n), dtype=torch.float16, device="cuda")
    cl.launch(
        torch.cuda.current_stream(),
        (2,),
        (4 * WARP_SIZE,),
        two_cta_tcgen05_kernel,
        (make_3d_view(a), make_3d_view(b.T.contiguous()), c),
        block_in_cluster_count=(2, 1, 1),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b, atol=1e-1, rtol=1e-2)


@cl.kernel
def tma_tcgen05_kernel(a, b, c):
    cta_m = 128
    cta_n = 64
    tile_n = 128
    tile_k = 64
    num_k_tiles = 2
    num_warps = 4

    tid = cl.thread_index(0)
    warp = tid // WARP_SIZE
    rank = cl.block_in_cluster_index(0)
    pair_rank = rank % 2

    a_tmap = cl.tensor_map_tiled(a, (64, cta_m, 1), swizzle=cl.SwizzleMode.SWIZZLE_128B)
    b_tmap = cl.tensor_map_tiled(b, (64, cta_n, 1), swizzle=cl.SwizzleMode.SWIZZLE_128B)
    c_tmap = cl.tensor_map_tiled(c, (tile_n, cta_m), order="F")
    a_smem = cl.shared_array(cta_m * tile_k, cl.float16, alignment=512)
    b_smem = cl.shared_array(cta_n * tile_k, cl.float16, alignment=512)
    c_smem = cl.shared_array(cta_m * tile_n, cl.float16, alignment=128)
    a_bar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    b_bar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    mma_bar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    tmem_storage = cl.shared_array(
        1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
    )

    if warp == 0 and cl.elect_sync():
        cl.mbarrier_initialize(a_bar, 1)
        cl.mbarrier_initialize(b_bar, 1)
        # Both CTA pairs consume each multicast B tile.
        cl.mbarrier_initialize(mma_bar, 2)
        cl.fence_mbarrier_initialize()
    cl.barrier_sync_cluster(aligned=True)

    if warp == 0:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(),
            tile_n,
            cta_group=cl.CTAGroup.CTA_2,
        )

    a_desc = cl.Tcgen05SharedMemoryDescriptor(
        matrix_start_address=p3_to_u64(a_smem.get_base_pointer()),
        leading_dimension_byte_offset=16,
        stride_dimension_byte_offset=8 * 128,
        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
    ).encode()
    b_desc = cl.Tcgen05SharedMemoryDescriptor(
        matrix_start_address=p3_to_u64(b_smem.get_base_pointer()),
        leading_dimension_byte_offset=16,
        stride_dimension_byte_offset=8 * 128,
        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
    ).encode()
    instruction = cl.Tcgen05InstructionDescriptor(
        d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
        a_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
        b_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
        n=tile_n,
        m=2 * cta_m,
    ).encode()

    for k_tile in cl.static_iter(range(num_k_tiles)):
        phase = k_tile & 1
        if tid == 0:
            cl.mbarrier_arrive_expect_transaction(a_bar, cta_m * tile_k * 2)
            cl.mbarrier_arrive_expect_transaction(b_bar, cta_n * tile_k * 2)
        cl.barrier_sync_cluster(aligned=True)

        if tid == 0:
            cl.copy_async_bulk_tensor_global_to_shared(
                a_tmap,
                (0, rank * cta_m, k_tile),
                a_smem.get_base_pointer(),
                a_bar,
            )
            if rank < 2:
                b_dst = cl.map_shared_to_cluster(b_smem.get_base_pointer(), rank)
                cl.copy_async_bulk_tensor_global_to_shared(
                    b_tmap,
                    (0, rank * cta_n, k_tile),
                    b_dst,
                    b_bar,
                    multicast_mask=cl.int16(0b0101 << rank),
                )

        wait_mbarrier(a_bar, phase)
        wait_mbarrier(b_bar, phase)
        cl.barrier_sync_block()

        if pair_rank == 0 and warp == 0 and cl.elect_sync():
            cl.tcgen05_fence_after_thread_sync()
            for kk in cl.static_iter(range(tile_k // MMA_K)):
                cl.tcgen05_mma(
                    cl.Tcgen05MMAKind.F16,
                    tmem_storage[0],
                    a_desc + (32 >> 4) * kk,
                    b_desc + (32 >> 4) * kk,
                    instruction,
                    accumulate=k_tile != 0 or kk != 0,
                    cta_group=cl.CTAGroup.CTA_2,
                )
            cl.tcgen05_commit(
                mma_bar,
                multicast_mask=0b1111,
                cta_group=cl.CTAGroup.CTA_2,
            )

        wait_mbarrier(mma_bar, phase)
        cl.barrier_sync_block()

    cl.tcgen05_fence_after_thread_sync()
    for column in cl.static_iter(range(0, tile_n, 16)):
        store_fp16_tmem_tile(
            c_smem.get_base_pointer(),
            tmem_storage[0],
            pair_rank * num_warps + warp,
            column,
            tid,
            column,
            tile_n,
        )
    cl.barrier_sync_block()
    if tid == 0:
        cl.fence_proxy(cl.FenceProxyKind.ASYNC_SHARED, space=cl.MemorySpace.SHARED)
        cl.copy_async_bulk_tensor_shared_to_global(
            c_smem.get_base_pointer(), c_tmap, (0, rank * cta_m)
        )
        cl.copy_async_bulk_commit_group()
        cl.copy_async_bulk_wait_group(0)

    cl.barrier_sync_cluster(aligned=True)
    if warp == 0:
        cl.tcgen05_deallocate(tmem_storage[0], tile_n, cta_group=cl.CTAGroup.CTA_2)


def test_tma_tcgen05():
    m, n, k = 512, 128, 128
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    c = torch.empty((m, n), dtype=torch.float16, device="cuda")
    cl.launch(
        torch.cuda.current_stream(),
        (4,),
        (4 * WARP_SIZE,),
        tma_tcgen05_kernel,
        (make_3d_view(a), make_3d_view(b.T.contiguous()), c),
        block_in_cluster_count=(4, 1, 1),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b, atol=1e-1, rtol=1e-2)


CLC_BYTES = cl.clusterlaunchcontrol_token.bitwidth // 8


def fence_clc_acquire():
    cl.fence_proxy_sync_restrict(cl.MemoryOrder.ACQUIRE)


def fence_clc_release():
    cl.fence_proxy_sync_restrict(cl.MemoryOrder.RELEASE)


def wait_mbarrier_cluster(mbar, phase):
    while not cl.mbarrier_try_wait_parity(
        mbar, phase, scope=cl.MbarrierScope.CLUSTER, time_hint=10_000
    ):
        pass


def swizzle_program_id(tile, tiles_m, tiles_n, width):
    full_tiles = tiles_m // width
    full_size = full_tiles * width
    full_elements = full_size * tiles_n
    tile_group = tile // (width * tiles_n)
    if tile < full_elements:
        pid_m = tile_group * width + tile % width
        pid_n = (tile // width) % tiles_n
    else:
        partial_width = tiles_m - full_size
        if partial_width == 0:
            partial_width = 1
        partial = tile - full_elements
        pid_m = full_size + partial % partial_width
        pid_n = (partial // partial_width) % tiles_n
    if tile_group % 2 != 0:
        pid_n = tiles_n - 1 - pid_n
    return pid_m, pid_n


def consume_scheduled_tile(ready, consumed, next_tile, next_has_work, phase):
    wait_mbarrier(ready, phase)
    tile = next_tile[0]
    has_work = next_has_work[0] != 0
    if cl.elect_sync():
        cl.mbarrier_arrive(consumed)
    return tile, has_work


def store_matmul_partition(
    c_tmap,
    c_smem,
    tmem_base,
    acc_ready,
    acc_empty,
    phase,
    pid_m,
    pid_n,
    rank,
    warp,
    lane,
    m,
    n,
):
    tile_m = 256
    tile_n = 256
    subtile_n = 32
    subtile_stages = 4

    wait_mbarrier(acc_ready, phase)
    cl.tcgen05_fence_after_thread_sync()
    valid_tile = pid_m < m // tile_m and pid_n < n // tile_n
    row = warp * WARP_SIZE + lane
    for subtile in cl.static_iter(range(tile_n // subtile_n)):
        stage = subtile % subtile_stages
        stage_smem = c_smem.get_element_pointer((stage, 0))
        if warp == 0 and cl.elect_sync():
            cl.copy_async_bulk_wait_group(subtile_stages - 1, read=True)
        cl.barrier_sync_block()

        if valid_tile:
            for column in cl.static_iter(range(0, subtile_n, 16)):
                store_fp16_tmem_tile(
                    stage_smem,
                    tmem_base,
                    rank * 4 + warp,
                    subtile * subtile_n + column,
                    row,
                    column,
                    subtile_n,
                )
        cl.barrier_sync_block()

        if warp == 0 and cl.elect_sync():
            cl.fence_proxy(
                cl.FenceProxyKind.ASYNC_SHARED,
                space=cl.MemorySpace.SHARED,
            )
            cl.copy_async_bulk_tensor_shared_to_global(
                stage_smem,
                c_tmap,
                (
                    pid_n * tile_n + subtile * subtile_n,
                    pid_m * tile_m + rank * 128,
                ),
                predicate=valid_tile,
            )
            cl.copy_async_bulk_commit_group()
    if cl.elect_sync():
        empty_bar = cl.map_shared_to_cluster(acc_empty, 0)
        cl.mbarrier_arrive(empty_bar, scope=cl.MbarrierScope.BLOCK)


@cl.kernel
def matmul_multicta_kernel(
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
    tile_k = 64
    snake_width = 16
    stages = 6
    acc_stages = 2
    subtile_n = 32
    subtile_stages = 4

    tid = cl.thread_index(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    rank = cl.block_in_cluster_index(0)
    tiles_m = m // tile_m
    tiles_n = n // tile_n

    a_tmap = cl.tensor_map_tiled(a, (64, cta_m, 1), swizzle=cl.SwizzleMode.SWIZZLE_128B)
    b_tmap = cl.tensor_map_tiled(b, (64, cta_n, 1), swizzle=cl.SwizzleMode.SWIZZLE_128B)
    c_tmap = cl.tensor_map_tiled(c, (subtile_n, cta_m), order="F")
    a_smem = cl.shared_array((stages, cta_m * tile_k), cl.float16, alignment=512)
    b_smem = cl.shared_array((stages, cta_n * tile_k), cl.float16, alignment=512)
    c_smem = cl.shared_array(
        (subtile_stages, cta_m * subtile_n),
        cl.float16,
        alignment=128,
    )
    load_ready = cl.shared_array(stages, cl.mbarrier, alignment=8)
    load_empty = cl.shared_array(stages, cl.mbarrier, alignment=8)
    acc_ready = cl.shared_array(acc_stages, cl.mbarrier, alignment=8)
    acc_empty = cl.shared_array(acc_stages, cl.mbarrier, alignment=8)
    clc_bar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
    scheduler_ready = cl.shared_array(acc_stages, cl.mbarrier, alignment=8)
    scheduler_consumed = cl.shared_array(acc_stages, cl.mbarrier, alignment=8)
    clc_token = cl.shared_array(
        1, cl.clusterlaunchcontrol_token, alignment=16
    ).get_base_pointer()
    next_tile = cl.shared_array(acc_stages, cl.int32)
    next_has_work = cl.shared_array(acc_stages, cl.int32)
    tmem_storage = cl.shared_array(
        1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
    )

    if warp == 0 and cl.elect_sync():
        for stage in cl.static_iter(range(stages)):
            cl.mbarrier_initialize(load_ready.get_element_pointer(stage), 2)
            cl.mbarrier_initialize(load_empty.get_element_pointer(stage), 1)
        for stage in cl.static_iter(range(acc_stages)):
            cl.mbarrier_initialize(acc_ready.get_element_pointer(stage), 1)
            cl.mbarrier_initialize(acc_empty.get_element_pointer(stage), 8)
            cl.mbarrier_initialize(scheduler_ready.get_element_pointer(stage), 1)
            cl.mbarrier_initialize(scheduler_consumed.get_element_pointer(stage), 3)
        cl.mbarrier_initialize(clc_bar, 1)
        cl.fence_mbarrier_initialize()
    cl.barrier_sync_cluster(aligned=True)

    if warp == 2:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(),
            tile_n * acc_stages,
            cta_group=cl.CTAGroup.CTA_2,
        )
    cl.barrier_sync_cluster(aligned=True)

    if warp == 3:
        clc_phase = 0
        iteration = 0
        scheduler_has_work = True
        tile = cl.cluster_index(0)
        while scheduler_has_work:
            slot = iteration % acc_stages
            ring_phase = (iteration // acc_stages) & 1
            if iteration >= acc_stages:
                wait_mbarrier(
                    scheduler_consumed.get_element_pointer(slot), ring_phase ^ 1
                )
            if rank == 0 and cl.elect_sync():
                fence_clc_acquire()
                cl.clusterlaunchcontrol_try_cancel(clc_token, clc_bar, multicast=True)
            if cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    clc_bar,
                    CLC_BYTES,
                    scope=cl.MbarrierScope.CLUSTER,
                    memory_order=cl.MemoryOrder.RELAXED,
                )
            wait_mbarrier_cluster(clc_bar, clc_phase)
            token = clc_token.load()
            scheduler_has_work = cl.clusterlaunchcontrol_is_canceled(token)
            if cl.elect_sync():
                next_has_work[slot] = cl.int32(scheduler_has_work)
                if scheduler_has_work:
                    block = cl.clusterlaunchcontrol_get_first_block_index(token, axis=0)
                    stolen_tile = block // 2
                    next_tile[slot] = stolen_tile
                    fence_clc_release()
                cl.mbarrier_arrive(scheduler_ready.get_element_pointer(slot))
            cl.barrier_sync_warp()
            pid_m, pid_n = swizzle_program_id(tile, tiles_m, tiles_n, snake_width)
            store_matmul_partition(
                c_tmap,
                c_smem,
                tensor_memory_pointer(tmem_storage[0], 0, slot * tile_n),
                acc_ready.get_element_pointer(slot),
                acc_empty.get_element_pointer(slot),
                ring_phase,
                pid_m,
                pid_n,
                rank,
                warp,
                lane,
                m,
                n,
            )
            if scheduler_has_work:
                tile = next_tile[slot]
            clc_phase ^= 1
            iteration += 1

    elif warp == 1:
        tile = cl.cluster_index(0)
        has_work = True
        load_index = 0
        tile_index = 0
        while has_work:
            pid_m, pid_n = swizzle_program_id(tile, tiles_m, tiles_n, snake_width)
            for k_tile in range(k // tile_k):
                stage = load_index % stages
                stage_phase = (load_index // stages) & 1
                if load_index >= stages:
                    wait_mbarrier(
                        load_empty.get_element_pointer(stage), stage_phase ^ 1
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
                        (cta_m + cta_n) * tile_k * 2,
                        scope=cl.MbarrierScope.BLOCK,
                    )
                load_index += 1
            slot = tile_index % acc_stages
            ring_phase = (tile_index // acc_stages) & 1
            store_matmul_partition(
                c_tmap,
                c_smem,
                tensor_memory_pointer(tmem_storage[0], 0, slot * tile_n),
                acc_ready.get_element_pointer(slot),
                acc_empty.get_element_pointer(slot),
                ring_phase,
                pid_m,
                pid_n,
                rank,
                warp,
                lane,
                m,
                n,
            )
            tile, has_work = consume_scheduled_tile(
                scheduler_ready.get_element_pointer(slot),
                scheduler_consumed.get_element_pointer(slot),
                next_tile.get_element_pointer(slot),
                next_has_work.get_element_pointer(slot),
                ring_phase,
            )
            tile_index += 1

    elif warp == 2:
        tile = cl.cluster_index(0)
        has_work = True
        load_index = 0
        acc_index = 0
        instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
            b_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
            n=tile_n,
            m=tile_m,
        ).encode()
        while has_work:
            acc_slot = acc_index % acc_stages
            acc_phase = (acc_index // acc_stages) & 1
            acc_tmem = tensor_memory_pointer(tmem_storage[0], 0, acc_slot * tile_n)
            if acc_index >= acc_stages and rank == 0 and cl.elect_sync():
                wait_mbarrier(acc_empty.get_element_pointer(acc_slot), acc_phase ^ 1)
            for k_tile in range(k // tile_k):
                stage = load_index % stages
                stage_phase = (load_index // stages) & 1
                if rank == 0 and cl.elect_sync():
                    wait_mbarrier(load_ready.get_element_pointer(stage), stage_phase)
                    cl.tcgen05_fence_after_thread_sync()
                    a_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=p3_to_u64(
                            a_smem.get_element_pointer((stage, 0))
                        ),
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=8 * 128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                    ).encode()
                    b_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=p3_to_u64(
                            b_smem.get_element_pointer((stage, 0))
                        ),
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=8 * 128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                    ).encode()
                    for kk in cl.static_iter(range(tile_k // MMA_K)):
                        cl.tcgen05_mma(
                            cl.Tcgen05MMAKind.F16,
                            acc_tmem,
                            a_desc + (32 >> 4) * kk,
                            b_desc + (32 >> 4) * kk,
                            instruction,
                            accumulate=k_tile != 0 or kk != 0,
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
                    acc_ready.get_element_pointer(acc_slot),
                    multicast_mask=0b11,
                    cta_group=cl.CTAGroup.CTA_2,
                )
            pid_m, pid_n = swizzle_program_id(tile, tiles_m, tiles_n, snake_width)
            store_matmul_partition(
                c_tmap,
                c_smem,
                acc_tmem,
                acc_ready.get_element_pointer(acc_slot),
                acc_empty.get_element_pointer(acc_slot),
                acc_phase,
                pid_m,
                pid_n,
                rank,
                warp,
                lane,
                m,
                n,
            )
            acc_index += 1
            tile, has_work = consume_scheduled_tile(
                scheduler_ready.get_element_pointer(acc_slot),
                scheduler_consumed.get_element_pointer(acc_slot),
                next_tile.get_element_pointer(acc_slot),
                next_has_work.get_element_pointer(acc_slot),
                acc_phase,
            )
        if rank == 0 and cl.elect_sync():
            for stage in cl.static_iter(range(acc_stages)):
                if acc_index > stage:
                    last_use = acc_index - 1
                    if last_use % acc_stages != stage:
                        last_use -= 1
                    last_phase = (last_use // acc_stages) & 1
                    wait_mbarrier(acc_empty.get_element_pointer(stage), last_phase)
        cl.tcgen05_deallocate(
            tmem_storage[0],
            tile_n * acc_stages,
            cta_group=cl.CTAGroup.CTA_2,
        )

    else:
        tile = cl.cluster_index(0)
        has_work = True
        acc_index = 0
        while has_work:
            pid_m, pid_n = swizzle_program_id(tile, tiles_m, tiles_n, snake_width)
            slot = acc_index % acc_stages
            ring_phase = (acc_index // acc_stages) & 1
            store_matmul_partition(
                c_tmap,
                c_smem,
                tensor_memory_pointer(tmem_storage[0], 0, slot * tile_n),
                acc_ready.get_element_pointer(slot),
                acc_empty.get_element_pointer(slot),
                ring_phase,
                pid_m,
                pid_n,
                rank,
                warp,
                lane,
                m,
                n,
            )
            acc_index += 1
            tile, has_work = consume_scheduled_tile(
                scheduler_ready.get_element_pointer(slot),
                scheduler_consumed.get_element_pointer(slot),
                next_tile.get_element_pointer(slot),
                next_has_work.get_element_pointer(slot),
                ring_phase,
            )

    if warp == 0 and cl.elect_sync():
        cl.copy_async_bulk_wait_group(0)
    cl.barrier_sync_cluster(aligned=True)


def test_matmul_multicta():
    m, n, k = 1024, 1024, 512
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    c = torch.empty((m, n), dtype=torch.float16, device="cuda")
    tiles = (m // 256) * (n // 256)
    cl.launch(
        torch.cuda.current_stream(),
        (tiles * 2,),
        (4 * WARP_SIZE,),
        matmul_multicta_kernel,
        (
            make_3d_view(a),
            make_3d_view(b.T.contiguous()),
            c,
            m,
            n,
            k,
        ),
        block_in_cluster_count=(2, 1, 1),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b, atol=1e-1, rtol=1e-2)


def make_3d_view(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    assert cols % 64 == 0
    return torch.as_strided(
        x,
        size=(64, rows, cols // 64),
        stride=(1, cols, 64),
    )
