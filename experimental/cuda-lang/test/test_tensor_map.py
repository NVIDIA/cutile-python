# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang.compilation import KernelSignature
import pytest
import torch

import cuda.lang as cl
from cuda.lang._datatype import float4_e2m1fn, to_torch_dtype
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.ops import CreateTensorMap
from cuda.tile import _cext

from .util import get_ir, make_symbolic_tensor, require_hopper_or_newer


def _build_ir(kernel, dtype):
    return get_ir(kernel, [make_symbolic_tensor((1, 1), dtype)])


def test_float4_tensor_map_requires_explicit_encoding():
    def kernel(x):
        cl.tensor_map_tiled(x, 1)

    with pytest.raises(
        TypeCheckingError,
        match=r"Data type float4_e2m1fn is not supported by tensor map",
    ):
        _build_ir(kernel, float4_e2m1fn)


def _make_expected_tile(x, row, column, tile_height, tile_width):
    expected = torch.zeros((tile_height, tile_width), dtype=x.dtype, device=x.device)
    source = x[row:row + tile_height, column:column + tile_width]
    expected[:source.shape[0], :source.shape[1]] = source
    return expected


@require_hopper_or_newer()
@pytest.mark.parametrize(
    "dtype",
    (
        cl.uint8,
        cl.int8,
        cl.float8_e4m3fn,
        cl.float8_e5m2,
        cl.float8_e8m0fnu,
    ),
)
def test_tmadesc_byte_types(dtype):
    def kernel(x):
        tmap = cl.tensor_map_tiled(x, (16, 16), order="F")
        cl.prefetch_tensor_map(tmap)

    ir = _build_ir(kernel, dtype)
    [create] = [op for op in ir.traverse() if isinstance(op, CreateTensorMap)]
    assert create.result_var.get_type().data_type == "CU_TENSOR_MAP_DATA_TYPE_UINT8"

    kernel = cl.kernel(kernel)
    sig = KernelSignature([make_symbolic_tensor(1, dtype)])
    cres = cl.compile_simt(kernel, [sig], gpu_name="sm_100a", arch="compute_100a")
    assert len(cres.hoisted_tensor_maps) == 1
    assert cres.hoisted_tensor_maps[0].data_type == _cext.CU_TENSOR_MAP_DATA_TYPE_UINT8


@require_hopper_or_newer()
@pytest.mark.parametrize("l2_promotion", tuple(cl.TensorMapL2Promotion))
def test_tensor_map_l2_promotion_metadata(l2_promotion):
    def kernel(x):
        tmap = cl.tensor_map_tiled(
            x,
            (16, 16),
            order="F",
            l2_promotion=l2_promotion,
        )
        cl.prefetch_tensor_map(tmap)

    sig = KernelSignature([make_symbolic_tensor((1, 1), cl.float16)])
    result = cl.compile_simt(
        kernel,
        [sig],
        gpu_name="sm_100a",
        arch="compute_100a",
    )
    assert len(result.hoisted_tensor_maps) == 1
    assert result.hoisted_tensor_maps[0].l2_promotion is l2_promotion


@require_hopper_or_newer()
@pytest.mark.parametrize(
    "cl_dtype",
    (
        cl.uint8,
        cl.int8,
        cl.float8_e4m3fn,
        cl.float8_e5m2,
        cl.float8_e8m0fnu,
    ),
)
def test_tensor_map_byte_types_l2_promotion_launch(cl_dtype):
    torch_dtype = to_torch_dtype(cl_dtype)
    tile_height, tile_width = 8, 32

    @cl.kernel
    def kernel(
        x,
        y,
        row,
        column,
        tile_height: cl.Constant[int],
        tile_width: cl.Constant[int],
    ):
        tmap = cl.tensor_map_tiled(
            x,
            (tile_width, tile_height),
            order="F",
            l2_promotion=cl.TensorMapL2Promotion.L2_128B,
        )
        smem = cl.shared_array(
            tile_height * tile_width,
            cl.uint8,
            alignment=128,
        )
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

        if cl.thread_index(0) == 0:
            cl.mbarrier_initialize(mbar, cl.thread_count(0))
            cl.fence_mbarrier_initialize()

        cl.barrier_sync_block()
        if cl.elect_sync():
            cl.copy_async_bulk_tensor_global_to_shared(
                tmap,
                (column, row),
                smem.get_base_pointer(),
                mbar,
            )
            token = cl.mbarrier_arrive_expect_transaction(
                mbar,
                tmap.get_transaction_bytes(),
            )
        else:
            token = cl.mbarrier_arrive(mbar)

        cl.mbarrier_wait(mbar, token, time_hint=10_000)
        index = cl.thread_index(0)
        y[index] = smem[index]

    byte_values = (
        torch.arange(37 * 48, dtype=torch.int32, device="cuda")
        .remainder(256)
        .to(torch.uint8)
        .reshape((37, 48))
        .contiguous()
    )
    x = byte_values if torch_dtype is torch.uint8 else byte_values.view(torch_dtype)

    for row, column in ((0, 0), (1, 16)):
        y = torch.zeros(tile_height * tile_width, dtype=torch.uint8, device="cuda")
        cl.launch(
            torch.cuda.current_stream(),
            (1,),
            (tile_height * tile_width,),
            kernel,
            (x, y, row, column, tile_height, tile_width),
        )
        expected = byte_values[
            row:row + tile_height,
            column:column + tile_width,
        ]
        torch.testing.assert_close(y.reshape((tile_height, tile_width)), expected)


@require_hopper_or_newer()
@pytest.mark.parametrize(
    "row,column",
    (
        pytest.param(0, 0, id="in_bounds"),
        pytest.param(1, 12, id="offset_in_bounds"),
        pytest.param(20, 44, id="partial_oob"),
    ),
)
def test_transaction_bytes_with_oob_fill(row, column):
    tma_alignment = 128
    mbarrier_alignment = 8
    poll_delay_ns = 10_000

    @cl.kernel
    def kernel(
        x, y, row, column, tile_height: cl.Constant[int], tile_width: cl.Constant[int]
    ):
        tensor_map = cl.tensor_map_tiled(x, (tile_width, tile_height), order="F")
        smem = cl.shared_array(
            tile_width * tile_height, cl.int32, alignment=tma_alignment
        )
        mbar = cl.shared_array(
            1, cl.mbarrier, alignment=mbarrier_alignment
        ).get_base_pointer()

        if cl.thread_index(0) == 0:
            cl.mbarrier_initialize(mbar, cl.thread_count(0))
            cl.fence_mbarrier_initialize()

        cl.barrier_sync_block()
        if cl.elect_sync():
            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map, (column, row), smem.get_base_pointer(), mbar
            )
            token = cl.mbarrier_arrive_expect_transaction(
                mbar, tensor_map.get_transaction_bytes()
            )
        else:
            token = cl.mbarrier_arrive(mbar)

        cl.mbarrier_wait(mbar, token, time_hint=poll_delay_ns)

        index = cl.thread_index(0)
        y[index] = smem[index]

    x = torch.arange(37 * 48, dtype=torch.int32, device="cuda").reshape(37, 48)
    tile_height, tile_width = 32, 8
    y = torch.empty(tile_height * tile_width, dtype=x.dtype, device=x.device)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (tile_height * tile_width,),
        kernel,
        (x, y, row, column, tile_height, tile_width),
    )

    expected = _make_expected_tile(x, row, column, tile_height, tile_width)
    torch.testing.assert_close(y.reshape(tile_height, tile_width), expected)


@require_hopper_or_newer()
def test_transaction_bytes_with_multicast():
    tma_alignment = 128
    mbarrier_alignment = 8
    poll_delay_ns = 10_000
    multicast_cta_count = 2
    multicast_mask = (1 << multicast_cta_count) - 1

    @cl.kernel
    def kernel(x, y, tile_height: cl.Constant[int], tile_width: cl.Constant[int]):
        rank = cl.block_in_cluster_index(0)
        tensor_map = cl.tensor_map_tiled(x, (tile_width, tile_height), order="F")
        smem = cl.shared_array(
            tile_width * tile_height, cl.int32, alignment=tma_alignment
        )
        mbar = cl.shared_array(
            1, cl.mbarrier, alignment=mbarrier_alignment
        ).get_base_pointer()

        if cl.thread_index(0) == 0:
            cl.mbarrier_initialize(mbar, cl.thread_count(0))
            cl.fence_mbarrier_initialize()

        cl.barrier_sync_block()
        cl.barrier_sync_cluster()

        # Each destination CTA establishes its expected transaction count
        # before rank 0 initiates the multicast load.
        if cl.elect_sync():
            token = cl.mbarrier_arrive_expect_transaction(
                mbar, tensor_map.get_transaction_bytes()
            )
        else:
            token = cl.mbarrier_arrive(mbar)

        cl.barrier_sync_cluster()
        if rank == 0 and cl.elect_sync():
            destination = cl.map_shared_to_cluster(smem.get_base_pointer(), 0)
            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (0, 0),
                destination,
                mbar,
                multicast_mask=multicast_mask,
            )

        cl.mbarrier_wait(mbar, token, time_hint=poll_delay_ns)

        index = cl.thread_index(0)
        y[rank, index] = smem[index]

    tile_height, tile_width = 32, 8
    x = torch.arange(
        tile_height * tile_width, dtype=torch.int32, device="cuda"
    ).reshape(tile_height, tile_width)
    y = torch.empty(
        (multicast_cta_count, tile_height * tile_width),
        dtype=x.dtype,
        device=x.device,
    )
    cl.launch(
        torch.cuda.current_stream(),
        (multicast_cta_count,),
        (tile_height * tile_width,),
        kernel,
        (x, y, tile_height, tile_width),
        block_in_cluster_count=(multicast_cta_count, 1, 1),
    )

    expected = x.reshape(1, -1).broadcast_to(multicast_cta_count, -1)
    torch.testing.assert_close(y, expected)


@require_hopper_or_newer()
def test_transaction_bytes_with_128b_swizzle():
    tma_alignment = 128
    mbarrier_alignment = 8
    poll_delay_ns = 10_000
    block_size = 32

    @cl.kernel
    def kernel(x, y, tile_height: cl.Constant[int], tile_width: cl.Constant[int]):
        src_map = cl.tensor_map_tiled(
            x, (tile_width, tile_height), order="F", swizzle=cl.SwizzleMode.SWIZZLE_128B
        )
        dst_map = cl.tensor_map_tiled(
            y, (tile_width, tile_height), order="F", swizzle=cl.SwizzleMode.SWIZZLE_128B
        )
        smem = cl.shared_array(
            tile_width * tile_height, cl.int32, alignment=tma_alignment
        )
        mbar = cl.shared_array(
            1, cl.mbarrier, alignment=mbarrier_alignment
        ).get_base_pointer()

        if cl.thread_index(0) == 0:
            cl.mbarrier_initialize(mbar, cl.thread_count(0))
            cl.fence_mbarrier_initialize()

        cl.barrier_sync_block()
        if cl.elect_sync():
            cl.copy_async_bulk_tensor_global_to_shared(
                src_map, (0, 0), smem.get_base_pointer(), mbar
            )

            token = cl.mbarrier_arrive_expect_transaction(
                mbar, src_map.get_transaction_bytes()
            )
        else:
            token = cl.mbarrier_arrive(mbar)

        cl.mbarrier_wait(mbar, token, time_hint=poll_delay_ns)

        # A matching TMA store consumes the swizzled shared-memory layout
        # without assuming that it is linearly addressable by threads.
        if cl.elect_sync():
            cl.copy_async_bulk_tensor_shared_to_global(
                smem.get_base_pointer(), dst_map, (0, 0)
            )
            cl.copy_async_bulk_commit_group()
            cl.copy_async_bulk_wait_group(0)

    tile_height, tile_width = 8, 32
    x = torch.arange(
        tile_height * tile_width, dtype=torch.int32, device="cuda"
    ).reshape(tile_height, tile_width)
    y = torch.empty_like(x)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (block_size,),
        kernel,
        (x, y, tile_height, tile_width),
    )

    torch.testing.assert_close(y, x)


@pytest.mark.parametrize(
    "mode",
    (
        cl.TMALoadMode.IM2COL,
        cl.TMALoadMode.IM2COL_W,
        cl.TMALoadMode.IM2COL_W_128,
    ),
)
def test_tiled_map_rejects_im2col_transaction_byte_computation(mode):
    def kernel(x):
        tensor_map = cl.tensor_map_tiled(x, (7, 3), order="F")
        x[0, 0] = tensor_map.get_transaction_bytes(mode=mode)

    with pytest.raises(
        TypeCheckingError,
        match=rf"^Cannot compute {mode.name} transaction bytes from a tiled tensor map",
    ):
        _build_ir(kernel, cl.int32)


def test_invalid_gather4_map_is_rejected():
    def kernel(x):
        tensor_map = cl.tensor_map_tiled(x, (7, 2), order="F")
        x[0, 0] = tensor_map.get_transaction_bytes(mode=cl.TMALoadMode.TILE_GATHER4)

    with pytest.raises(
        TypeCheckingError,
        match=r"^TILE_GATHER4 requires a rank-2 tensor map with tile_shape\[1\] == 1",
    ):
        _build_ir(kernel, cl.int32)
