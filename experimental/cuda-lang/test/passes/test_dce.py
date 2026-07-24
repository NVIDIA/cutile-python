# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Ensure cudalang ops with memory effects are not removed by DCE
"""

import cuda.lang as cl
from cuda.lang._compile import _transform_ir
from cuda.lang._ir.ir import Operation
from cuda.lang._ir.ops import (
    AtomicRMW, InlinePTX, AllocStaticSharedMemory, RawNVVMIntrinsic, Return, LoadPointer
)

from ..util import get_ir, make_symbolic_scalar, make_symbolic_tensor


def _get_transformed_ir(func, constraints):
    body = get_ir(func, constraints)
    dyn_smem_size_program, _ = _transform_ir(body, body.ctx)
    return body, dyn_smem_size_program


def ir_wrapper(func):
    """Tests in this file mostly need to get the post-dce IR and check if
    an op is present."""
    body, dyn_smem_size_program = _get_transformed_ir(
        func,
        [
            make_symbolic_tensor(shape=(1,), dtype=cl.int32),
            make_symbolic_scalar(dtype=cl.int32),
        ],
    )

    def has_op(cls: type[Operation], body=body):
        return any(isinstance(op, cls) for op in body.traverse())

    setattr(func, "body", body)
    setattr(func, "dyn_smem_size_program", dyn_smem_size_program)
    setattr(func, "has_op", has_op)
    return func


class TestOpsSurviveDCE:
    def test_dead_load_is_removed(self):
        @ir_wrapper
        def kernel(A, n):
            A[0]

        assert not kernel.has_op(LoadPointer)

    def test_unused_atomic_result_is_kept(self):
        @ir_wrapper
        def kernel(A, n):
            cl.atomic_add(A.get_element_pointer(0), cl.int32(1))

        assert kernel.has_op(AtomicRMW)

    def test_syncthreads_intrinsic_is_kept(self):
        @ir_wrapper
        def kernel(A, n):
            cl.barrier_sync_block()

        assert kernel.has_op(RawNVVMIntrinsic)

    def test_inline_ptx_without_used_results_is_kept(self):
        @ir_wrapper
        def kernel(A, n):
            cl._inline_ptx("bar.sync 0;")

        assert kernel.has_op(InlinePTX)

    def test_unused_static_shared_memory_allocation_is_kept(self):
        @ir_wrapper
        def kernel(A, n):
            cl.shared_array(shape=(32,), dtype=cl.int32)

        assert kernel.has_op(AllocStaticSharedMemory)

    def test_unused_dynamic_shared_memory_is_preserved_in_size_program(self):
        @ir_wrapper
        def kernel(A, n):
            cl.shared_array(shape=(n,), dtype=cl.int32, dynamic=True)

        assert kernel.dyn_smem_size_program is not None
        assert kernel.dyn_smem_size_program.opcodes == ["Const", "KernelArgI32", "Mul"]
        ops = kernel.body.traverse()
        assert isinstance(next(ops), Return)
        assert next(ops, None) is None
