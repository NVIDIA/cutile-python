# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability

__doc__ = """
Simple ptx verification of tcgen05 mma instructions is tested elsewhere, but
it's not straightforward to functionally test them in isolation. These tests
attempt to do the absolute minimum outside verifying the functionality of mma
instructions.
"""

cc = get_compute_capability()
if tuple(cc) != (10, 0):
    pytest.skip("requires tcgen05", allow_module_level=True)


WARP_SIZE = 32
OUTPUT_WARPS = 4
MMA_WARP = OUTPUT_WARPS
THREADS = (OUTPUT_WARPS + 1) * WARP_SIZE

M = 128
N = 128
OUTPUT_COLUMNS = 16
INPUT_WORDS = 4096
SCALE_WORDS = 128
TMEM_COLUMNS = 256
SCALE_A_COLUMN = 128
SCALE_B_COLUMN = 136
SPARSE_METADATA_COLUMN = 144
AUXILIARY_END_COLUMN = 160


def make_tcgen05_mma_kernel(
    entrypoint,
    is_sparse,
    *,
    mma_kind=cl.Tcgen05MMAKind.F16,
    block_scale_kind=cl.Tcgen05MMABlockScaleKind.MXF8F6F4,
    scale_vector_size=cl.Tcgen05MMAScaleVectorSize.DEFAULT,
    copy_scales=False,
):
    is_block_scale = entrypoint is cl.tcgen05_mma_block_scale
    is_weight_stationary = entrypoint is cl.tcgen05_mma_weight_stationary
    is_fp8 = mma_kind is cl.Tcgen05MMAKind.F8F6F4
    is_mxfp8 = block_scale_kind is cl.Tcgen05MMABlockScaleKind.MXF8F6F4
    is_nvfp4 = block_scale_kind is cl.Tcgen05MMABlockScaleKind.MXF4NVF4

    @cl.kernel
    def kernel(output):
        matrix_a = cl.shared_array(INPUT_WORDS, cl.uint32, alignment=512)
        matrix_b = cl.shared_array(INPUT_WORDS, cl.uint32, alignment=512)
        scale_a_smem = cl.shared_array(SCALE_WORDS, cl.uint32, alignment=128)
        scale_b_smem = cl.shared_array(SCALE_WORDS, cl.uint32, alignment=128)
        mma_done = cl.shared_array(1, cl.mbarrier, alignment=8)
        tmem_storage = cl.shared_array(
            1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
        )

        tid = cl.thread_index(0)
        warp = tid // WARP_SIZE

        # Eight copies of E2M1(1.0), four copies of E4M3(1.0), or two copies
        # of BF16(1.0).
        if is_block_scale and is_nvfp4:
            input_word = cl.uint32(0x22222222)
        elif is_block_scale or is_fp8:
            input_word = cl.uint32(0x38383838)
        else:
            input_word = cl.uint32(0x3F803F80)
        for i in cl.static_iter(range((INPUT_WORDS + THREADS - 1) // THREADS)):
            index = tid + i * THREADS
            if index < INPUT_WORDS:
                matrix_a[index] = input_word
                matrix_b[index] = input_word

        if copy_scales and tid < SCALE_WORDS:
            if not is_nvfp4:
                # UE8M0 encodes 2.0 and 4.0 as exponents 0x80 and 0x81.
                scale_a_smem[tid] = cl.uint32(0x80808080)
                scale_b_smem[tid] = cl.uint32(0x81818181)
            else:
                # UE4M3 encodes 2.0 and 4.0 as 0x40 and 0x48.
                scale_a_smem[tid] = cl.uint32(0x40404040)
                scale_b_smem[tid] = cl.uint32(0x48484848)

        if warp == 0 and cl.elect_sync():
            cl.mbarrier_initialize(mma_done.get_base_pointer(), 1)
            cl.fence_mbarrier_initialize()

        cl.barrier_sync_block()

        if warp == MMA_WARP:
            cl.tcgen05_allocate(tmem_storage.get_base_pointer(), TMEM_COLUMNS)

        cl.barrier_sync_block()

        tmem = tmem_storage[0]
        initializes_tmem = (is_block_scale and not copy_scales) or is_sparse
        if warp < OUTPUT_WARPS and initializes_tmem:
            if is_block_scale and not copy_scales:
                # UE8M0(1.0) is 0x7f; UE4M3(1.0) is 0x38.
                scale_word = cl.int32(0x38383838 if is_nvfp4 else 0x7F7F7F7F)
                for column in cl.static_iter(
                    range(SCALE_A_COLUMN, SPARSE_METADATA_COLUMN)
                ):
                    scale_ptr = cl.tcgen05_tmem_offset(
                        tmem,
                        lane_offset=warp * WARP_SIZE,
                        column_offset=column,
                    )
                    cl.tcgen05_store(
                        cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                        scale_ptr,
                        scale_word,
                    )

            # Metadata value zero selects a valid 2:4 pattern. B is all ones, so
            # every valid metadata selection has the same numerical result.
            if is_sparse:
                for column in cl.static_iter(
                    range(SPARSE_METADATA_COLUMN, AUXILIARY_END_COLUMN)
                ):
                    metadata_ptr = cl.tcgen05_tmem_offset(
                        tmem,
                        lane_offset=warp * WARP_SIZE,
                        column_offset=column,
                    )
                    cl.tcgen05_store(
                        cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                        metadata_ptr,
                        cl.int32(0),
                    )

            cl.tcgen05_wait_store()
            cl.tcgen05_fence_before_thread_sync()

        cl.barrier_sync_block()
        cl.tcgen05_fence_after_thread_sync()

        if warp == MMA_WARP and cl.elect_sync():
            if copy_scales:
                cl.fence_proxy(
                    cl.FenceProxyKind.ASYNC_SHARED,
                    space=cl.MemorySpace.SHARED,
                )
                scale_a_descriptor = cl.Tcgen05SharedMemoryDescriptor(
                    matrix_start_address=scale_a_smem,
                    leading_dimension_byte_offset=128,
                    stride_dimension_byte_offset=128,
                ).encode()
                scale_b_descriptor = cl.Tcgen05SharedMemoryDescriptor(
                    matrix_start_address=scale_b_smem,
                    leading_dimension_byte_offset=128,
                    stride_dimension_byte_offset=128,
                ).encode()
                cl.tcgen05_copy(
                    tmem + SCALE_A_COLUMN,
                    scale_a_descriptor,
                    shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                    multicast=cl.Tcgen05CopyMulticast.WARPX4,
                )
                cl.tcgen05_copy(
                    tmem + SCALE_B_COLUMN,
                    scale_b_descriptor,
                    shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                    multicast=cl.Tcgen05CopyMulticast.WARPX4,
                )

            a_descriptor = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=matrix_a,
                leading_dimension_byte_offset=0,
                stride_dimension_byte_offset=8 * 128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
            ).encode()
            b_descriptor = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=matrix_b,
                leading_dimension_byte_offset=0,
                stride_dimension_byte_offset=8 * 128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
            ).encode()
            sparse_metadata = tmem + SPARSE_METADATA_COLUMN if is_sparse else None

            if is_block_scale:
                if is_mxfp8:
                    instruction_descriptor = cl.Tcgen05Mxf8f6f4InstructionDescriptor(
                        sparse=is_sparse,
                        a_type=cl.Tcgen05Mxf8f6f4InstructionDescriptor.Type.E4M3,
                        b_type=cl.Tcgen05Mxf8f6f4InstructionDescriptor.Type.E4M3,
                        n=N,
                        m=M,
                    ).encode()
                else:
                    instruction_descriptor = cl.Tcgen05Mxf4InstructionDescriptor(
                        sparse=is_sparse,
                        a_type=cl.Tcgen05Mxf4InstructionDescriptor.Type.E2M1,
                        b_type=cl.Tcgen05Mxf4InstructionDescriptor.Type.E2M1,
                        n=N,
                        scale_format=(
                            cl.Tcgen05Mxf4InstructionDescriptor.ScaleFormat.UE4M3
                            if is_nvfp4
                            else cl.Tcgen05Mxf4InstructionDescriptor.ScaleFormat.UE8M0
                        ),
                        m=M,
                    ).encode()
                cl.tcgen05_mma_block_scale(
                    block_scale_kind,
                    tmem,
                    a_descriptor,
                    b_descriptor,
                    instruction_descriptor,
                    tmem + SCALE_A_COLUMN,
                    tmem + SCALE_B_COLUMN,
                    accumulate=False,
                    sparse_metadata=sparse_metadata,
                    scale_vector_size=scale_vector_size,
                )
            else:
                a_type = (
                    cl.Tcgen05InstructionDescriptor.F8F6F4Type.E4M3
                    if is_fp8
                    else cl.Tcgen05InstructionDescriptor.F16Type.BF16
                )
                instruction_descriptor = cl.Tcgen05InstructionDescriptor(
                    sparse=is_sparse,
                    d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
                    a_type=a_type,
                    b_type=a_type,
                    n=N,
                    m=M,
                ).encode()
                operation = (
                    cl.tcgen05_mma_weight_stationary
                    if is_weight_stationary
                    else cl.tcgen05_mma
                )
                operation(
                    mma_kind,
                    tmem,
                    a_descriptor,
                    b_descriptor,
                    instruction_descriptor,
                    accumulate=False,
                    sparse_metadata=sparse_metadata,
                )

            cl.tcgen05_commit(mma_done.get_base_pointer())

        if warp < OUTPUT_WARPS:
            if warp == 0:
                cl.mbarrier_wait_parity(mma_done.get_base_pointer(), 0)

            cl.barrier_sync_block(
                number_of_threads=OUTPUT_WARPS * WARP_SIZE,
                barrier_id=1,
            )
            cl.tcgen05_fence_after_thread_sync()

            output_tmem = cl.tcgen05_tmem_offset(
                tmem,
                lane_offset=warp * WARP_SIZE,
            )
            registers = cl.tcgen05_load(
                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                output_tmem,
                count=OUTPUT_COLUMNS,
            )
            cl.tcgen05_wait_load()
            for column in cl.static_iter(range(OUTPUT_COLUMNS)):
                output[tid * OUTPUT_COLUMNS + column] = cl.bitcast(
                    registers[column], cl.float32
                )

        cl.barrier_sync_block()
        if warp == 0:
            cl.tcgen05_deallocate(tmem, TMEM_COLUMNS)

    return kernel


def run_tcgen05_mma_kernel(kernel, expected):
    output = torch.empty(M * OUTPUT_COLUMNS, dtype=torch.float32, device="cuda")

    cl.launch(
        torch.cuda.current_stream(),
        (1, 1, 1),
        (THREADS, 1, 1),
        kernel,
        (output,),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, torch.full_like(output, expected))


@pytest.mark.parametrize(
    "entrypoint,is_sparse,expected",
    (
        (cl.tcgen05_mma, False, 16.0),
        (cl.tcgen05_mma, True, 16.0),
        (cl.tcgen05_mma_block_scale, False, 32.0),
        (cl.tcgen05_mma_block_scale, True, 32.0),
        (cl.tcgen05_mma_weight_stationary, False, 16.0),
        (cl.tcgen05_mma_weight_stationary, True, 16.0),
    ),
)
def test_tcgen05_mma_instruction(entrypoint, is_sparse, expected):
    """Exercise every mma entrypoint end to end with all-1 input matrices.
    The expected result is the instruction's accumulation count.
    """
    kernel = make_tcgen05_mma_kernel(entrypoint, is_sparse)
    run_tcgen05_mma_kernel(kernel, expected)


def test_tcgen05_f8_mma_instruction():
    """Exercise the unscaled E4M3 MMA used by the FP8 GEMM."""
    kernel = make_tcgen05_mma_kernel(
        cl.tcgen05_mma,
        False,
        mma_kind=cl.Tcgen05MMAKind.F8F6F4,
    )
    run_tcgen05_mma_kernel(kernel, 32.0)


@pytest.mark.parametrize(
    "block_scale_kind,scale_vector_size,expected",
    (
        pytest.param(
            cl.Tcgen05MMABlockScaleKind.MXF8F6F4,
            cl.Tcgen05MMAScaleVectorSize.DEFAULT,
            256.0,
            id="mxfp8",
        ),
        pytest.param(
            cl.Tcgen05MMABlockScaleKind.MXF4NVF4,
            cl.Tcgen05MMAScaleVectorSize.BLOCK_16,
            512.0,
            id="nvfp4",
        ),
    ),
)
def test_tcgen05_block_scale_mma_with_copied_scales(
    block_scale_kind, scale_vector_size, expected
):
    """Exercise the scale-copy and block-scaled MMA paths needed by TK ports."""
    kernel = make_tcgen05_mma_kernel(
        cl.tcgen05_mma_block_scale,
        False,
        block_scale_kind=block_scale_kind,
        scale_vector_size=scale_vector_size,
        copy_scales=True,
    )
    run_tcgen05_mma_kernel(kernel, expected)
