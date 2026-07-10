# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability


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
TMEM_COLUMNS = 256
SCALE_A_COLUMN = 128
SCALE_B_COLUMN = 136
SPARSE_METADATA_COLUMN = 144
AUXILIARY_END_COLUMN = 160


def wait_mbarrier(mbar, phase):
    while not cl.mbarrier_try_wait_parity(mbar, phase, time_hint=10_000):
        pass


def p3_to_u64(pointer):
    return cl.uint64(cl.bitcast(pointer, cl.uint32))


def make_tcgen05_mma_kernel(entrypoint, is_sparse):
    is_block_scale = entrypoint is cl.tcgen05_mma_block_scale
    is_weight_stationary = entrypoint is cl.tcgen05_mma_weight_stationary

    @cl.kernel
    def kernel(output):
        matrix_a = cl.shared_array(INPUT_WORDS, cl.uint32, alignment=512)
        matrix_b = cl.shared_array(INPUT_WORDS, cl.uint32, alignment=512)
        mma_done = cl.shared_array(1, cl.mbarrier, alignment=8)
        tmem_storage = cl.shared_array(
            1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
        )

        tid = cl.thread_index(0)
        warp = tid // WARP_SIZE

        # Four copies of E4M3(1.0), or two copies of BF16(1.0).
        input_word = cl.uint32(0x38383838 if is_block_scale else 0x3F803F80)
        for i in cl.static_iter(range((INPUT_WORDS + THREADS - 1) // THREADS)):
            index = tid + i * THREADS
            if index < INPUT_WORDS:
                matrix_a[index] = input_word
                matrix_b[index] = input_word

        if warp == 0 and cl.elect_sync():
            cl.mbarrier_initialize(mma_done.get_base_pointer(), 1)
            cl.fence_mbarrier_initialize()

        cl.barrier_sync_block()

        if warp == MMA_WARP:
            cl.tcgen05_allocate(tmem_storage.get_base_pointer(), TMEM_COLUMNS)

        cl.barrier_sync_block()

        tmem = tmem_storage[0]
        if warp < OUTPUT_WARPS and (is_block_scale or is_sparse):
            # UE8M0(1.0) is 0x7f. Fill every byte used by either scale tensor.
            for column in cl.static_iter(range(SCALE_A_COLUMN, SPARSE_METADATA_COLUMN)):
                scale_ptr = cl.tcgen05_tmem_offset(
                    tmem,
                    lane_offset=warp * WARP_SIZE,
                    column_offset=column,
                )
                cl.tcgen05_store(
                    cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                    scale_ptr,
                    cl.int32(0x7F7F7F7F),
                )

            # Metadata value zero selects a valid 2:4 pattern. B is all ones, so
            # every valid metadata selection has the same numerical result.
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
            a_descriptor = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=p3_to_u64(matrix_a.get_base_pointer()),
                leading_dimension_offset=0,
                stride_dimension_offset=8 * 128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
            ).encode()
            b_descriptor = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=p3_to_u64(matrix_b.get_base_pointer()),
                leading_dimension_offset=0,
                stride_dimension_offset=8 * 128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
            ).encode()
            sparse_metadata = tmem + SPARSE_METADATA_COLUMN if is_sparse else None

            if is_block_scale:
                instruction_descriptor = cl.Tcgen05Mxf8f6f4InstructionDescriptor(
                    sparse=is_sparse,
                    a_type=cl.Tcgen05Mxf8f6f4InstructionDescriptor.Type.E4M3,
                    b_type=cl.Tcgen05Mxf8f6f4InstructionDescriptor.Type.E4M3,
                    n=N,
                    m=M,
                ).encode()
                cl.tcgen05_mma_block_scale(
                    cl.Tcgen05MMABlockScaleKind.MXF8F6F4,
                    tmem,
                    a_descriptor,
                    b_descriptor,
                    instruction_descriptor,
                    tmem + SCALE_A_COLUMN,
                    tmem + SCALE_B_COLUMN,
                    accumulate=False,
                    sparse_metadata=sparse_metadata,
                )
            else:
                instruction_descriptor = cl.Tcgen05InstructionDescriptor(
                    sparse=is_sparse,
                    d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
                    a_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
                    b_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
                    n=N,
                    m=M,
                ).encode()
                operation = (
                    cl.tcgen05_mma_weight_stationary
                    if is_weight_stationary
                    else cl.tcgen05_mma
                )
                operation(
                    cl.Tcgen05MMAKind.F16,
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
                wait_mbarrier(mma_done.get_base_pointer(), 0)

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


@pytest.mark.xfail(strict=False)
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
    output = torch.empty(M * OUTPUT_COLUMNS, dtype=torch.float32, device="cuda")
    kernel = make_tcgen05_mma_kernel(entrypoint, is_sparse)

    cl.launch(
        torch.cuda.current_stream(),
        (1, 1, 1),
        (THREADS, 1, 1),
        kernel,
        (output,),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, torch.full_like(output, expected))
