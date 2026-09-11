# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import pytest
import torch

from .util import (
    require_blackwell_cc100,
    require_blackwell_or_newer,
    require_hopper_or_newer,
)


def make_inputs(shape, dtype, tile_shape, ctas_per_tile=1):
    random_dtype = torch.float32 if dtype == torch.float8_e4m3fn else dtype
    x = torch.randn(shape, dtype=random_dtype, device='cuda:0').to(dtype)
    y = torch.zeros_like(x)
    M, N = shape
    tm, tn = tile_shape
    grid = (cl.cdiv(M, tm) * ctas_per_tile, cl.cdiv(N, tn))
    block = (tm, tn)
    return (x, y), (grid, block)


# Global to shared

def build_copy_kernel(
    init_mbar, do_copy, tile_shape, element_bytes, ctas_per_tile=1,
    output_tile_rows=None,
):
    @cl.kernel
    def kernel(input_tmap, output):
        by, bx = cl.block_index(0), cl.block_index(1)
        ty, tx = cl.thread_index(0), cl.thread_index(1)
        w, h = tile_shape
        # Gather4 produces four output rows from a tensor-map tile one row high.
        if output_tile_rows is not None:
            h = output_tile_rows
        row, col = (by // ctas_per_tile) * h, bx * w
        smem = cl.shared_array(shape=(h * w,), dtype=output.dtype, alignment=512).pointer()
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        init_mbar(mbar)
        do_copy(
            tx == 0 and ty == 0, input_tmap, smem, (col, row), mbar,
            h * w * element_bytes, w,
        )
        oy, ox = row + ty, col + tx
        # When CTAs share a tile, split the stores to check every destination.
        if (
            ty % ctas_per_tile == by % ctas_per_tile
            and oy < output.shape[0]
            and ox < output.shape[1]
        ):
            output[oy, ox] = smem[ty * w + tx]

    return kernel


def init_mbar(mbar):
    if cl.thread_index(0) == 0 and cl.thread_index(1) == 0:
        cl.mbarrier_initialize(mbar, 1)
        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            restriction=cl.FenceRestriction.mbarrier_initialize(),
        )
    cl.barrier_sync_block_aligned()


def do_copy(is_cta_leader, input_tmap, smem, coords, mbar,
            transaction_bytes, tile_width):
    if is_cta_leader:
        cl.mbarrier_arrive_expect_transaction(mbar, transaction_bytes)
        cl.copy_async_bulk_tensor_global_to_shared(input_tmap, coords, smem, mbar)
    cl.mbarrier_wait_parity(mbar, 0)


def do_copy_multicast(is_cta_leader, input_tmap, smem, coords, mbar,
                      transaction_bytes, tile_width):
    cl.barrier_sync_cluster_aligned()
    if is_cta_leader:
        cl.mbarrier_arrive_expect_transaction(mbar, transaction_bytes)
    cl.barrier_sync_cluster_aligned()

    if cl.block_in_cluster_index(0) == 0 and is_cta_leader:
        cl.copy_async_bulk_tensor_global_to_shared(
            input_tmap,
            coords,
            cl.map_shared_to_cluster(smem, 0),
            mbar,
            multicast_mask=0b11,
        )
    cl.mbarrier_wait_parity(mbar, 0)


def do_copy_cta2(is_cta_leader, input_tmap, smem, coords, mbar,
                 transaction_bytes, tile_width):
    rank = cl.block_in_cluster_index(0)
    cl.barrier_sync_cluster_aligned()
    if rank == 0 and is_cta_leader:
        cl.mbarrier_arrive_expect_transaction(
            mbar, 2 * transaction_bytes
        )
    cl.barrier_sync_cluster_aligned()

    if is_cta_leader:
        cl.copy_async_bulk_tensor_global_to_shared(
            input_tmap,
            coords,
            cl.map_shared_to_cluster(smem, rank),
            cl.map_shared_to_leader_block(mbar),
            multicast_mask=cl.int16(1 << rank),
            cta_group=cl.CTAGroup.CTA_2,
        )
    if rank == 0:
        cl.mbarrier_wait_parity(mbar, 0)
    cl.barrier_sync_cluster_aligned()


def do_copy_gather4(is_cta_leader, input_tmap, smem, coords, mbar,
                    transaction_bytes, tile_width):
    col, row = coords
    if is_cta_leader:
        cl.mbarrier_arrive_expect_transaction(
            mbar, transaction_bytes
        )
        w = tile_width
        for group in cl.static_iter(range(4)):
            group_row = row + 4 * group
            cl.copy_async_bulk_tensor_global_to_shared(
                input_tmap,
                (col, group_row + 3, group_row + 1, group_row, group_row + 2),
                smem + group * 4 * w,
                mbar,
                mode=cl.TMALoadMode.TILE_GATHER4,
            )
    cl.mbarrier_wait_parity(mbar, 0)


@pytest.mark.parametrize(
    "tile_mode",
    (
        pytest.param(cl.TMALoadMode.TILE, id="tiled"),
        pytest.param(
            cl.TMALoadMode.TILE_GATHER4,
            marks=require_blackwell_or_newer(),
            id="gather4",
        ),
    ),
)
@require_hopper_or_newer()
def test_g2s_tiled_cta_no_cluster(tile_mode):
    tm, tn = 16, 32
    gather4 = tile_mode == cl.TMALoadMode.TILE_GATHER4
    (x, y), (grid, block) = make_inputs((96, 160), torch.float32, (tm, tn))
    x_map = cl.tensor_map_tiled(x, (tn, 1 if gather4 else tm), order=(1, 0))
    copy_fn = do_copy_gather4 if gather4 else do_copy
    kernel = build_copy_kernel(
        init_mbar, copy_fn, (tn, 1 if gather4 else tm),
        x.element_size(), output_tile_rows=tm if gather4 else None,
    )
    cl.launch(torch.cuda.current_stream(), grid, block, kernel, (x_map, y))
    if gather4:
        expected = x.reshape(-1, 4, x.shape[1])[:, (3, 1, 0, 2), :]
        expected = expected.reshape(y.shape)
    else:
        expected = x
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


@require_hopper_or_newer()
def test_g2s_tiled_shared_cluster_multicast():
    cluster_size = 2
    tm, tn = 16, 32
    (x, y), (grid, block) = make_inputs(
        (96, 160), torch.float16, (tm, tn), ctas_per_tile=cluster_size
    )
    x_map = cl.tensor_map_tiled(x, (tn, tm), order=(1, 0))
    kernel = build_copy_kernel(
        init_mbar, do_copy_multicast, (tn, tm), x.element_size(),
        ctas_per_tile=cluster_size,
    )
    cl.launch(
        torch.cuda.current_stream(), grid, block, kernel, (x_map, y),
        block_in_cluster_count=(cluster_size, 1, 1),
    )
    torch.testing.assert_close(y, x, rtol=0, atol=0)


@require_blackwell_cc100()
def test_g2s_tiled_shared_cluster_cta2():
    cluster_size = 2
    tm, tn = 16, 32
    (x, y), (grid, block) = make_inputs(
        (96, 160), torch.float32, (tm, tn)
    )
    x_map = cl.tensor_map_tiled(x, (tn, tm), order=(1, 0))
    kernel = build_copy_kernel(init_mbar, do_copy_cta2, (tn, tm), x.element_size())
    cl.launch(
        torch.cuda.current_stream(), grid, block, kernel, (x_map, y),
        block_in_cluster_count=(cluster_size, 1, 1),
    )
    torch.testing.assert_close(y, x, rtol=0, atol=0)


# Shared to global


def build_store_kernel(do_store, tile_width, output_tile_rows):
    @cl.kernel
    def kernel(input, output_tmap):
        by, bx = cl.block_index(0), cl.block_index(1)
        ty, tx = cl.thread_index(0), cl.thread_index(1)
        w = tile_width
        h = output_tile_rows
        row, col = by * h, bx * w
        smem = cl.shared_array(shape=(h * w,), dtype=input.dtype, alignment=512).pointer()

        smem[ty * w + tx] = input[row + ty, col + tx]
        cl.fence_proxy_bidirectional(
            cl.FenceProxy.ASYNC,
            restriction=cl.FenceRestriction.shared_block(),
        )
        cl.barrier_sync_block_aligned()

        if ty == 0 and tx == 0:
            do_store(smem, output_tmap, (col, row), w)
            cl.copy_async_bulk_commit_group()
            cl.copy_async_bulk_wait_group(0)

    return kernel


def do_store_tiled(smem, output_tmap, coords, tile_width):
    cl.copy_async_bulk_tensor_shared_to_global(smem, output_tmap, coords)


def do_store_scatter4(smem, output_tmap, coords, tile_width):
    col, row = coords
    w = tile_width
    for group in cl.static_iter(range(4)):
        group_row = row + 4 * group
        cl.copy_async_bulk_tensor_shared_to_global(
            smem + group * 4 * w,
            output_tmap,
            (col, group_row + 1, group_row, group_row + 3, group_row + 2),
            mode=cl.TMAStoreMode.TILE_SCATTER4,
        )


@pytest.mark.parametrize(
    "tile_store_mode",
    (
        pytest.param(cl.TMAStoreMode.TILE, id="tiled"),
        pytest.param(
            cl.TMAStoreMode.TILE_SCATTER4,
            marks=require_blackwell_cc100(),
            id="scatter4",
        ),
    ),
)
@require_hopper_or_newer()
def test_s2g_tiled_cta_no_cluster(tile_store_mode):
    tm, tn = 16, 32
    (x, y), (grid, block) = make_inputs((96, 160), torch.float32, (tm, tn))
    scatter4 = tile_store_mode == cl.TMAStoreMode.TILE_SCATTER4
    y_map = cl.tensor_map_tiled(y, (tn, 1 if scatter4 else tm), order=(1, 0))
    store_fn = do_store_scatter4 if scatter4 else do_store_tiled
    kernel = build_store_kernel(store_fn, tn, output_tile_rows=tm)
    cl.launch(
        torch.cuda.current_stream(), grid, block, kernel, (x, y_map)
    )
    if scatter4:
        expected = x.reshape(-1, 4, x.shape[1])[:, (1, 0, 3, 2), :]
        expected = expected.reshape(y.shape)
    else:
        expected = x
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


# Roundtrip with swizzle

@cl.static_def
def swizzled_shared_elements(tile_shape, mode, element_bytes):
    w, h = tile_shape
    if mode == cl.SwizzleMode.SWIZZLE_NONE:
        swizzle_bytes = 0
    elif mode == cl.SwizzleMode.SWIZZLE_32B:
        swizzle_bytes = 32
    elif mode == cl.SwizzleMode.SWIZZLE_64B:
        swizzle_bytes = 64
    else:
        swizzle_bytes = 128
    row_bytes = max(w * element_bytes, swizzle_bytes)
    return h * row_bytes // element_bytes


def build_copy_tiled_roundtrip_kernel(tile_shape, swizzle_mode, smem_dtype):
    @cl.kernel
    def kernel(input_tmap, output_tmap):
        by, bx = cl.block_index(0), cl.block_index(1)
        ty, tx = cl.thread_index(0), cl.thread_index(1)
        w, h = tile_shape
        coords = (bx * w, by * h)
        element_bytes = smem_dtype.bitwidth // 8
        smem = cl.shared_array(
            swizzled_shared_elements(tile_shape, swizzle_mode, element_bytes),
            smem_dtype,
            alignment=1024,
        ).pointer()
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        is_leader = ty == 0 and tx == 0

        init_mbar(mbar)
        do_copy(is_leader, input_tmap, smem, coords, mbar,
                w * h * element_bytes, w)

        if is_leader:
            do_store_tiled(smem, output_tmap, coords, w)
            cl.copy_async_bulk_commit_group()
            cl.copy_async_bulk_wait_group(0)

    return kernel


# The flip-8B mode cannot store from shared to global; atom-64B loads fault on B200.
@pytest.mark.parametrize(
    "swizzle_mode",
    (
        pytest.param(cl.SwizzleMode.SWIZZLE_NONE, id="none"),
        pytest.param(cl.SwizzleMode.SWIZZLE_32B, id="32b"),
        pytest.param(cl.SwizzleMode.SWIZZLE_64B, id="64b"),
        pytest.param(cl.SwizzleMode.SWIZZLE_128B, id="128b"),
        pytest.param(
            cl.SwizzleMode.SWIZZLE_128B_ATOM_32B,
            marks=require_blackwell_or_newer(),
            id="128b-atom-32b",
        ),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(torch.float8_e4m3fn, id="float8"),
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.float32, id="float32"),
    ),
)
@require_hopper_or_newer()
def test_tiled_roundtrip_swizzle(swizzle_mode, dtype):
    element_bytes = torch.empty((), dtype=dtype).element_size()
    tm, tn = 16, 16 // element_bytes
    (x, y), (grid, block) = make_inputs((96, 160), dtype, (tm, tn))
    x_map = cl.tensor_map_tiled(x, (tn, tm), order=(1, 0), swizzle=swizzle_mode)
    y_map = cl.tensor_map_tiled(y, (tn, tm), order=(1, 0), swizzle=swizzle_mode)
    smem_dtype = {
        torch.float8_e4m3fn: cl.uint8,
        torch.float16: cl.float16,
        torch.float32: cl.float32,
    }[dtype]
    kernel = build_copy_tiled_roundtrip_kernel((tn, tm), swizzle_mode, smem_dtype)
    cl.launch(
        torch.cuda.current_stream(), grid, block, kernel,
        (x_map, y_map),
    )
    torch.testing.assert_close(y.view(torch.uint8), x.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("direction", "swizzle_mode"),
    (
        pytest.param(
            "global-to-shared", cl.SwizzleMode.SWIZZLE_128B_ATOM_32B_FLIP_8B,
            id="flip8-load",
        ),
        pytest.param(
            "shared-to-global", cl.SwizzleMode.SWIZZLE_128B_ATOM_64B,
            id="atom64-store",
        ),
    ),
)
@require_blackwell_cc100()
def test_tensor_copy_swizzle_direction(direction, swizzle_mode):
    @cl.kernel
    def load_kernel(input_tmap, output):
        tx = cl.thread_index(1)
        smem = cl.shared_array(8 * 32, cl.float32, alignment=1024).pointer()
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        init_mbar(mbar)
        do_copy(tx == 0, input_tmap, smem, (0, 0), mbar, 8 * 32 * 4, 32)
        for row in cl.static_iter(range(8)):
            output[row, tx] = smem[row * 32 + tx]

    @cl.kernel
    def store_kernel(output_tmap):
        tx = cl.thread_index(0)
        smem = cl.shared_array(8 * 32, cl.float32, alignment=1024).pointer()
        for row in cl.static_iter(range(8)):
            smem[row * 32 + tx] = 123.0
        cl.barrier_sync_block_aligned()
        cl.fence_proxy_bidirectional(
            cl.FenceProxy.ASYNC,
            restriction=cl.FenceRestriction.shared_block(),
        )
        if tx == 0:
            cl.copy_async_bulk_tensor_shared_to_global(smem, output_tmap, (0, 0))
            cl.copy_async_bulk_commit_group()
            cl.copy_async_bulk_wait_group(0)

    tensor = torch.full((8, 32), 123, dtype=torch.float32, device="cuda")
    if direction == "shared-to-global":
        tensor.zero_()
    tensor_map = cl.tensor_map_tiled(tensor, (32, 8), order=(1, 0), swizzle=swizzle_mode)
    output = torch.zeros_like(tensor) if direction == "global-to-shared" else tensor

    def launch_case():
        if direction == "global-to-shared":
            cl.launch(
                torch.cuda.current_stream(), (1,), (1, 32),
                load_kernel, (tensor_map, output),
            )
        else:
            cl.launch(
                torch.cuda.current_stream(), (1,), (32,),
                store_kernel, (tensor_map,),
            )

    launch_case()
    torch.testing.assert_close(output, torch.full_like(output, 123), rtol=0, atol=0)
