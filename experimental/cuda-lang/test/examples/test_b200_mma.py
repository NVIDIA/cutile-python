# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability
import torch
import pytest

cc = get_compute_capability()
if tuple(cc) != (10, 0):
    pytest.skip("requires blackwell", True)


WARP_SIZE = 32
NUM_WARPS = 6
TB_SIZE = NUM_WARPS * WARP_SIZE

BLOCK_M = 128
BLOCK_K = 64
MMA_K = 16


def wait_mbarrier(mbar, phase):
    while not cl.mbarrier_try_wait_parity(mbar, phase, time_hint=10_000):
        pass


def p3_to_u64(pointer):
    return cl.uint64(cl.bitcast(pointer, cl.uint32))


def epilogue_store_tile(c_ptr, tmem_base, warp, base_col, g_row, g_col, n):
    tmem_ptr = cl.tcgen05_tmem_offset(
        tmem_base,
        lane_offset=warp * WARP_SIZE,
        column_offset=base_col,
    )
    regs = cl.tcgen05_load(cl.Tcgen05LoadStoreShape.SHAPE_32X32B, tmem_ptr, count=16)
    cl.tcgen05_wait_load()

    for pair_idx in cl.static_iter(range(8)):
        lo = cl.bitcast(regs[pair_idx * 2], cl.float32)
        hi = cl.bitcast(regs[pair_idx * 2 + 1], cl.float32)
        packed = cl._nvvm.ff2bf16x2_rn(hi, lo)
        offset = g_row * n + g_col + pair_idx * 2
        c_ptr.get_element_pointer(offset).store(packed, alignment=4)


def make_mma_kernel(
    block_n: int,
    cta_group_kind: cl.CTAGroup,
    num_stages: int,
):
    cta_group = 2 if cta_group_kind is cl.CTAGroup.CTA_2 else 1

    @cl.kernel
    def mma_kernel(
        a,
        b,
        c,
        m: cl.Constant[int],
        n: cl.Constant[int],
        k_total: cl.Constant[int],
    ):
        tid = cl.thread_index(0)
        bid = cl.block_index(0)
        num_bids = cl.block_count(0)
        warp_id = tid // WARP_SIZE
        cta_rank = cl.block_in_cluster_index(0)

        grid_m = m // BLOCK_M
        grid_n = n // block_n
        num_tiles = grid_m * grid_n
        num_iters = k_total // BLOCK_K

        a_tmap = cl.tensor_map_tiled(
            a,
            (64, BLOCK_M, BLOCK_K // 64),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )
        b_tmap = cl.tensor_map_tiled(
            b,
            (64, block_n // cta_group, BLOCK_K // 64),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )

        a_smem = cl.shared_array(
            (num_stages, BLOCK_M * BLOCK_K), cl.bfloat16, alignment=512
        )
        b_smem = cl.shared_array(
            (num_stages, (block_n // cta_group) * BLOCK_K), cl.bfloat16, alignment=512
        )

        tma_mbars = cl.shared_array(num_stages, cl.mbarrier, alignment=8)
        mma_mbars = cl.shared_array(num_stages, cl.mbarrier, alignment=8)
        mainloop_mbars = cl.shared_array(2, cl.mbarrier, alignment=8)
        epilogue_mbars = cl.shared_array(2, cl.mbarrier, alignment=8)

        tmem_storage = cl.shared_array(
            1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
        )

        if warp_id == 0 and cl.elect_sync():
            for stage in cl.static_iter(range(num_stages)):
                cl.mbarrier_initialize(tma_mbars.get_element_pointer(stage), cta_group)
                cl.mbarrier_initialize(mma_mbars.get_element_pointer(stage), 1)

            for stage in cl.static_iter(range(2)):
                cl.mbarrier_initialize(mainloop_mbars.get_element_pointer(stage), 1)
                cl.mbarrier_initialize(
                    epilogue_mbars.get_element_pointer(stage),
                    4 * cta_group * WARP_SIZE,
                )

            cl.fence_mbarrier_initialize()

        if cta_group > 1:
            cl.barrier_sync_cluster(aligned=True)
        else:
            cl.barrier_sync_block()

        if warp_id == NUM_WARPS - 2:
            if cl.elect_sync():
                tma_stage = 0
                mma_phase = 1

                this_bid = bid
                while this_bid < num_tiles:
                    bid_m = this_bid // (grid_n * 2) * 2 + (this_bid % 2)
                    bid_n = (this_bid // 2) % grid_n
                    off_m = bid_m * BLOCK_M
                    off_n = bid_n * block_n + cta_rank * (block_n // cta_group)

                    for iter_k in range(num_iters):
                        a_stage_ptr = a_smem.get_element_pointer((tma_stage, 0))
                        b_stage_ptr = b_smem.get_element_pointer((tma_stage, 0))
                        a_tma_dst = a_stage_ptr
                        b_tma_dst = b_stage_ptr
                        if cta_group > 1:
                            a_tma_dst = cl.map_shared_to_cluster(a_stage_ptr, cta_rank)
                            b_tma_dst = cl.map_shared_to_cluster(b_stage_ptr, cta_rank)
                        tma_mbar = tma_mbars.get_element_pointer(tma_stage)
                        mma_mbar = mma_mbars.get_element_pointer(tma_stage)
                        tma_arrive_mbar = tma_mbar
                        tma_expect_mbar = tma_mbar
                        if cta_group > 1:
                            tma_mbar_addr = cl.bitcast(tma_mbar, cl.uint32) & 0xFEFFFFFF
                            tma_arrive_mbar = cl.bitcast(
                                tma_mbar_addr,
                                cl.pointer_dtype(cl.mbarrier, cl.MemorySpace.SHARED),
                            )
                            tma_expect_mbar = cl.map_shared_to_cluster(tma_mbar, 0)

                        wait_mbarrier(mma_mbar, mma_phase)

                        if cta_group > 1:
                            cl.copy_async_bulk_tensor_global_to_shared(
                                a_tmap,
                                (0, off_m, iter_k),
                                a_tma_dst,
                                tma_arrive_mbar,
                                cta_group=cta_group_kind,
                            )
                            cl.copy_async_bulk_tensor_global_to_shared(
                                b_tmap,
                                (0, off_n, iter_k),
                                b_tma_dst,
                                tma_arrive_mbar,
                                cta_group=cta_group_kind,
                            )
                        else:
                            cl.copy_async_bulk_tensor_global_to_shared(
                                a_tmap,
                                (0, off_m, iter_k),
                                a_tma_dst,
                                tma_arrive_mbar,
                            )
                            cl.copy_async_bulk_tensor_global_to_shared(
                                b_tmap,
                                (0, off_n, iter_k),
                                b_tma_dst,
                                tma_arrive_mbar,
                            )
                        cl.mbarrier_arrive_expect_transaction(
                            tma_expect_mbar,
                            a_tmap.get_transaction_bytes() + b_tmap.get_transaction_bytes(),
                            scope=cl.MbarrierScope.BLOCK,
                        )

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            mma_phase = mma_phase ^ 1

                    this_bid = this_bid + num_bids

        elif warp_id == NUM_WARPS - 1:
            cl.tcgen05_allocate(
                tmem_storage.get_base_pointer(),
                block_n * 2,
                cta_group=cta_group_kind,
            )
            tmem_ptr = tmem_storage[0]

            i_desc = cl.Tcgen05InstructionDescriptor(
                d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
                a_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
                b_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
                n=block_n,
                m=BLOCK_M * cta_group,
            ).encode()

            if cta_rank == 0 and cl.elect_sync():
                tma_stage = 0
                tma_phase = 0
                mainloop_stage = 0
                epilogue_phase = 1
                cta_mask = (1 << cta_group) - 1

                this_bid = bid
                while this_bid < num_tiles:
                    wait_mbarrier(
                        epilogue_mbars.get_element_pointer(mainloop_stage),
                        epilogue_phase,
                    )

                    for iter_k in range(num_iters):
                        a_stage_ptr = a_smem.get_element_pointer((tma_stage, 0))
                        b_stage_ptr = b_smem.get_element_pointer((tma_stage, 0))
                        tensor_memory_address = mainloop_stage * block_n

                        a_smem_addr = p3_to_u64(a_stage_ptr)
                        b_smem_addr = p3_to_u64(b_stage_ptr)
                        a_desc = cl.Tcgen05SharedMemoryDescriptor(
                            matrix_start_address=a_smem_addr,
                            leading_dimension_byte_offset=16,
                            stride_dimension_byte_offset=8 * 128,
                            swizzle_mode=(cl.SwizzleMode.SWIZZLE_128B),
                        ).encode()
                        b_desc = cl.Tcgen05SharedMemoryDescriptor(
                            matrix_start_address=b_smem_addr,
                            leading_dimension_byte_offset=16,
                            stride_dimension_byte_offset=8 * 128,
                            swizzle_mode=(cl.SwizzleMode.SWIZZLE_128B),
                        ).encode()

                        wait_mbarrier(
                            tma_mbars.get_element_pointer(tma_stage), tma_phase
                        )
                        cl.tcgen05_fence_after_thread_sync()

                        cl.tcgen05_mma(
                            cl.Tcgen05MMAKind.F16,
                            tmem_ptr + tensor_memory_address,
                            a_desc,
                            b_desc,
                            i_desc,
                            accumulate=iter_k != 0,
                            cta_group=cta_group_kind,
                        )
                        for kk in cl.static_iter(range(1, BLOCK_K // MMA_K)):
                            cl.tcgen05_mma(
                                cl.Tcgen05MMAKind.F16,
                                tmem_ptr + tensor_memory_address,
                                a_desc + (32 >> 4) * kk,
                                b_desc + (32 >> 4) * kk,
                                i_desc,
                                accumulate=True,
                                cta_group=cta_group_kind,
                            )

                        if cta_group > 1:
                            cl.tcgen05_commit(
                                mma_mbars.get_element_pointer(tma_stage),
                                multicast_mask=cta_mask,
                                cta_group=cta_group_kind,
                            )
                        else:
                            cl.tcgen05_commit(
                                mma_mbars.get_element_pointer(tma_stage),
                                cta_group=cta_group_kind,
                            )

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            tma_phase = tma_phase ^ 1

                    if cta_group > 1:
                        cl.tcgen05_commit(
                            mainloop_mbars.get_element_pointer(mainloop_stage),
                            multicast_mask=cta_mask,
                            cta_group=cta_group_kind,
                        )
                    else:
                        cl.tcgen05_commit(
                            mainloop_mbars.get_element_pointer(mainloop_stage),
                            cta_group=cta_group_kind,
                        )

                    mainloop_stage = (mainloop_stage + 1) % 2
                    if mainloop_stage == 0:
                        epilogue_phase = epilogue_phase ^ 1

                    this_bid = this_bid + num_bids

        else:
            mainloop_stage = 0
            mainloop_phase = 0

            this_bid = bid
            while this_bid < num_tiles:
                bid_m = this_bid // (grid_n * 2) * 2 + (this_bid % 2)
                bid_n = (this_bid // 2) % grid_n

                if warp_id == 0:
                    wait_mbarrier(
                        mainloop_mbars.get_element_pointer(mainloop_stage),
                        mainloop_phase,
                    )

                cl.barrier_sync_block(barrier_id=1, number_of_threads=4 * WARP_SIZE)
                cl.tcgen05_fence_after_thread_sync()

                for tile_n in cl.static_iter(range(block_n // 16)):
                    t_row = cta_rank * 128 + warp_id * 32
                    t_col = mainloop_stage * block_n + tile_n * 16
                    g_row = bid_m * BLOCK_M + tid
                    g_col = bid_n * block_n + tile_n * 16
                    epilogue_store_tile(
                        c,
                        tmem_storage[0],
                        t_row // 32,
                        t_col,
                        g_row,
                        g_col,
                        n,
                    )

                epilogue_mbar = epilogue_mbars.get_element_pointer(mainloop_stage)
                if cta_group > 1:
                    epilogue_mbar = cl.map_shared_to_cluster(epilogue_mbar, 0)
                cl.mbarrier_arrive(epilogue_mbar, scope=cl.MbarrierScope.BLOCK)

                mainloop_stage = (mainloop_stage + 1) % 2
                if mainloop_stage == 0:
                    mainloop_phase = mainloop_phase ^ 1

                this_bid = this_bid + num_bids

            if cta_group > 1:
                cl.barrier_sync_cluster(aligned=True)
            else:
                cl.barrier_sync_block(barrier_id=1, number_of_threads=4 * WARP_SIZE)

            if warp_id == 0:
                cl.tcgen05_deallocate(
                    tmem_storage[0],
                    block_n * 2,
                    cta_group=cta_group_kind,
                )

    return mma_kernel


def make_3d_view(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    assert cols % 64 == 0
    return torch.as_strided(
        x,
        size=(64, rows, cols // 64),
        stride=(1, cols, 64),
    )


@pytest.mark.parametrize(
    "block_n, cta_group, num_stages, m, n, k",
    (
        pytest.param(128, 1, 3, 128, 128, 64),
        pytest.param(256, 1, 3, 128, 256, 64),
        pytest.param(128, 2, 3, 256, 128, 64),
        pytest.param(256, 2, 3, 256, 256, 64),
        pytest.param(256, 2, 2, 256, 256, 64),
    ),
)
def test_tcgen05_mma(block_n, cta_group, num_stages, m, n, k):
    torch.manual_seed(0)
    cta_group_kind = cl.CTAGroup.CTA_2 if cta_group == 2 else cl.CTAGroup.CTA_1
    kernel = make_mma_kernel(
        block_n=block_n,
        cta_group_kind=cta_group_kind,
        num_stages=num_stages,
    )

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    c = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)

    a_tma_view = make_3d_view(a)
    b_tma_view = make_3d_view(b)

    args = (a_tma_view, b_tma_view, c.reshape(m * n), m, n, k)
    grid = ((m // BLOCK_M) * (n // block_n), 1, 1)
    if cta_group > 1:
        cl.launch(
            torch.cuda.current_stream(),
            grid,
            (TB_SIZE, 1, 1),
            kernel,
            args,
            block_in_cluster_count=(cta_group, 1, 1),
        )
    else:
        cl.launch(
            torch.cuda.current_stream(),
            grid,
            (TB_SIZE, 1, 1),
            kernel,
            args,
        )
    torch.cuda.synchronize()

    expected = a.float() @ b.float().T
    torch.testing.assert_close(c.float(), expected, rtol=1e-2, atol=1e-1)


if __name__ == "__main__":
    configs = (
        (128, 1, 3, 128, 128, 64),
        (256, 1, 3, 128, 256, 64),
        (128, 2, 3, 256, 128, 64),
        (256, 2, 3, 256, 256, 64),
        (256, 2, 2, 256, 256, 64),
    )
    for config in configs:
        test_tcgen05_mma(*config)
