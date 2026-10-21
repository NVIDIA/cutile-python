# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.lang as cl
from cuda.lang.compilation import KernelSignature
from cuda.lang._datatype import to_torch_dtype
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.ops import AtomicCAS, AtomicExchange, AtomicRMW

from .util import (
    compile_kernel,
    get_ir,
    make_symbolic_scalar,
    make_symbolic_tensor,
)


ALL_INT_DTYPES = [cl.int32, cl.int64]
ALL_UINT_DTYPES = [cl.uint32, cl.uint64]
ALL_FLOAT_DTYPES = [cl.float32, cl.float64]
ALL_REAL_DTYPES = ALL_INT_DTYPES + ALL_UINT_DTYPES + ALL_FLOAT_DTYPES
ALL_INTEGER_DTYPES = ALL_INT_DTYPES + ALL_UINT_DTYPES


RMW_CASES = [
    (cl.AtomicOp.ADD, ALL_REAL_DTYPES, 7, 3, 10),
    (cl.AtomicOp.SUB, ALL_REAL_DTYPES, 7, 3, 4),
    (cl.AtomicOp.AND, ALL_INTEGER_DTYPES, 0b1110, 0b1011, 0b1010),
    (cl.AtomicOp.OR, ALL_INTEGER_DTYPES, 0b1100, 0b0011, 0b1111),
    (cl.AtomicOp.XOR, ALL_INTEGER_DTYPES, 0b1100, 0b1010, 0b0110),
    (cl.AtomicOp.MIN, ALL_REAL_DTYPES, 7, 3, 3),
    (cl.AtomicOp.MAX, ALL_REAL_DTYPES, 7, 11, 11),
    (cl.AtomicOp.INC, [cl.uint32], 7, 11, 8),
    (cl.AtomicOp.DEC, [cl.uint32], 7, 11, 6),
    (cl.AtomicOp.EXCH, ALL_REAL_DTYPES, 7, 11, 11),
]

RMW_VARIANTS = [
    (op, dtype, initial, update, expected_new)
    for op, dtypes, initial, update, expected_new in RMW_CASES
    for dtype in dtypes
]

UNSUPPORTED_DTYPE_CASES = [
    (cl.AtomicOp.ADD, cl.int16),
    (cl.AtomicOp.SUB, cl.float16),
    (cl.AtomicOp.AND, cl.float32),
    (cl.AtomicOp.OR, cl.float32),
    (cl.AtomicOp.XOR, cl.float32),
    (cl.AtomicOp.MIN, cl.float16),
    (cl.AtomicOp.MAX, cl.float16),
    (cl.AtomicOp.INC, cl.uint64),
    (cl.AtomicOp.DEC, cl.uint64),
    (cl.AtomicOp.EXCH, cl.int16),
    (cl.AtomicOp.CAS, cl.float32),
]

ATOMIC_MEMORY_ARGUMENT_CASES = [
    (cl.AtomicOp.ADD, cl.int32, AtomicRMW),
    (cl.AtomicOp.SUB, cl.int32, AtomicRMW),
    (cl.AtomicOp.AND, cl.int32, AtomicRMW),
    (cl.AtomicOp.OR, cl.int32, AtomicRMW),
    (cl.AtomicOp.XOR, cl.int32, AtomicRMW),
    (cl.AtomicOp.MIN, cl.int32, AtomicRMW),
    (cl.AtomicOp.MAX, cl.int32, AtomicRMW),
    (cl.AtomicOp.INC, cl.uint32, AtomicRMW),
    (cl.AtomicOp.DEC, cl.uint32, AtomicRMW),
    (cl.AtomicOp.EXCH, cl.int32, AtomicExchange),
    (cl.AtomicOp.CAS, cl.int32, AtomicCAS),
]

ATOMIC_ALIGNMENT_CASES = [
    (op, cl.uint32 if op in (cl.AtomicOp.INC, cl.AtomicOp.DEC) else cl.int32)
    for op in cl.AtomicOp
]


def _get_single_op(body, op_type):
    return next(op for op in body.traverse() if isinstance(op, op_type))


@pytest.mark.parametrize("op,dtype,initial,update,expected_new", RMW_VARIANTS)
def test_atomic_rmw_supported_types(op, dtype, initial, update, expected_new):
    torch_dtype = to_torch_dtype(dtype)

    @cl.kernel
    def kernel(A, out):
        ptr = A.pointer(0)
        out[0] = cl.atomic_rmw(op, ptr, dtype(update))

    A = torch.tensor([initial], dtype=torch_dtype, device="cuda:0")
    out = torch.zeros(1, dtype=torch_dtype, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert torch.allclose(out.cpu(), torch.tensor([initial], dtype=torch_dtype))
    assert torch.allclose(A.cpu(), torch.tensor([expected_new], dtype=torch_dtype))


@pytest.mark.parametrize("dtype", ALL_INTEGER_DTYPES)
def test_atomic_cas_supported_types(dtype):
    torch_dtype = to_torch_dtype(dtype)

    @cl.kernel
    def kernel(A, out):
        ptr = A.pointer(0)
        out[0] = cl.atomic_rmw(
            cl.AtomicOp.CAS, ptr, dtype(7), dtype(11)
        )

    A = torch.tensor([7], dtype=torch_dtype, device="cuda:0")
    out = torch.zeros(1, dtype=torch_dtype, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert torch.allclose(out.cpu(), torch.tensor([7], dtype=torch_dtype))
    assert torch.allclose(A.cpu(), torch.tensor([11], dtype=torch_dtype))


def test_atomic_cas_failure():
    @cl.kernel
    def kernel(A, out):
        ptr = A.pointer(0)
        out[0] = cl.atomic_rmw(
            cl.AtomicOp.CAS, ptr, cl.int32(8), cl.int32(11)
        )

    A = torch.tensor([7], dtype=torch.int32, device="cuda:0")
    out = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 7
    assert A.cpu()[0].item() == 7


@pytest.mark.parametrize(
    "op,initial,expected_new",
    ((cl.AtomicOp.INC, 7, 0), (cl.AtomicOp.DEC, 0, 7)),
)
def test_atomic_wrap(op, initial, expected_new):
    @cl.kernel
    def kernel(A, out):
        ptr = A.pointer(0)
        out[0] = cl.atomic_rmw(op, ptr, cl.uint32(7))

    A = torch.tensor([initial], dtype=torch.uint32, device="cuda:0")
    out = torch.zeros(1, dtype=torch.uint32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == initial
    assert A.cpu()[0].item() == expected_new


def test_atomic_tuple_index():
    @cl.kernel
    def kernel(A, out):
        ptr = A.pointer((0, 1))
        out[0] = cl.atomic_rmw(cl.AtomicOp.ADD, ptr, cl.int32(5))

    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cuda:0")
    out = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 2
    assert A.cpu()[0, 1].item() == 7


@pytest.mark.parametrize("op,dtype", UNSUPPORTED_DTYPE_CASES)
def test_atomic_unsupported_dtypes(op, dtype):
    def kernel(A):
        ptr = A.pointer(0)
        cl.atomic_rmw(op, ptr, A[0], A[0] if op == cl.AtomicOp.CAS else None)

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, dtype),)),
        raises=pytest.raises(
            TypeCheckingError,
            match=f"{op.value} does not support dtype {dtype}",
        ),
    )


@pytest.mark.parametrize(
    "op,dtype,operation_type", ATOMIC_MEMORY_ARGUMENT_CASES
)
def test_atomic_memory_arguments(op, dtype, operation_type):
    def kernel(A):
        ptr = A.pointer(0)
        cl.atomic_rmw(
            op,
            ptr,
            A[0],
            A[0] if op == cl.AtomicOp.CAS else None,
            memory_order=cl.MemoryOrder.RELAXED,
            memory_scope=cl.MemoryScope.BLOCK,
            alignment=16,
        )

    body = get_ir(kernel, (make_symbolic_tensor(1, dtype),))
    operation = _get_single_op(body, operation_type)
    assert operation.memory_order is cl.MemoryOrder.RELAXED
    assert operation.memory_scope is cl.MemoryScope.BLOCK
    assert operation.alignment == 16


@pytest.mark.parametrize("op,dtype", ATOMIC_ALIGNMENT_CASES)
def test_atomic_natural_alignment(op, dtype):
    def kernel(A):
        ptr = A.pointer(0)
        cl.atomic_rmw(op, ptr, A[0], (A[0] if op == cl.AtomicOp.CAS else None))

    body = get_ir(kernel, (make_symbolic_tensor(1, dtype),))
    if op is cl.AtomicOp.CAS:
        operation_type = AtomicCAS
    elif op is cl.AtomicOp.EXCH:
        operation_type = AtomicExchange
    else:
        operation_type = AtomicRMW
    assert _get_single_op(body, operation_type).alignment == 4


@pytest.mark.parametrize("op,dtype", ATOMIC_ALIGNMENT_CASES)
def test_atomic_alignment_lowering(op, dtype):
    def kernel(A):
        ptr = A.pointer(0)
        cl.atomic_rmw(
            op,
            ptr,
            A[0],
            (A[0] if op == cl.AtomicOp.CAS else None),
            alignment=16,
        )

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, dtype),)),
        assert_in_nvvm="align 16",
    )


@pytest.mark.parametrize("alignment", (0, -1, 3, True, 4.0))
def test_atomic_invalid_alignment(alignment):
    def kernel(A):
        cl.atomic_rmw(cl.AtomicOp.ADD, A.pointer(0), A[0], alignment=alignment)

    match = (
        "alignment must be a positive power of two"
        if isinstance(alignment, int) and not isinstance(alignment, bool)
        else "Expected an integer constant"
    )
    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        raises=pytest.raises(TypeCheckingError, match=match),
    )


def test_atomic_alignment_must_be_constant():
    def kernel(A, alignment):
        cl.atomic_rmw(cl.AtomicOp.ADD, A.pointer(0), A[0], alignment=alignment)

    compile_kernel(
        kernel,
        signature=KernelSignature(
            (
                make_symbolic_tensor(1, cl.int32),
                make_symbolic_scalar(cl.int32),
            )
        ),
        raises=pytest.raises(TypeCheckingError, match="Expected an integer constant"),
    )


@pytest.mark.parametrize(
    "order,scope,msg",
    (
        (cl.MemoryOrder.WEAK, cl.MemoryScope.DEVICE, "Invalid memory order"),
        (cl.MemoryOrder.RELEASE, cl.MemoryScope.NONE, "Invalid memory scope"),
    ),
)
def test_atomic_unsupported_memory_order_scope(order, scope, msg):
    def kernel(A):
        cl.atomic_rmw(
            cl.AtomicOp.ADD,
            A.pointer(0),
            A[0],
            memory_order=order,
            memory_scope=scope,
        )

    with pytest.raises(TypeCheckingError, match=msg):
        get_ir(kernel, (make_symbolic_tensor(1, cl.int32),))


@pytest.mark.parametrize(
    "op,operand2,msg",
    (
        (cl.AtomicOp.CAS, None, "AtomicOp.CAS requires a second operand"),
        (cl.AtomicOp.ADD, 1, "AtomicOp.ADD does not use a second operand"),
    ),
)
def test_atomic_second_operand(op, operand2, msg):
    def kernel(A):
        cl.atomic_rmw(op, A.pointer(0), A[0], operand2)

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        raises=pytest.raises(TypeCheckingError, match=msg),
    )


def test_atomic_cas_compare_type():
    def kernel(A):
        cl.atomic_rmw(cl.AtomicOp.CAS, A.pointer(0), cl.int64(0), A[0])

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        raises=pytest.raises(
            TypeCheckingError,
            match="Expected atomic compare value of type int32, got int64",
        ),
    )


def test_atomic_operation_must_be_constant():
    def kernel(A, op):
        cl.atomic_rmw(op, A.pointer(0), A[0])

    compile_kernel(
        kernel,
        signature=KernelSignature(
            (make_symbolic_tensor(1, cl.int32), make_symbolic_scalar(cl.int32))
        ),
        raises=pytest.raises(TypeCheckingError, match="Expected AtomicOp constant"),
    )


def test_atomic_operation_type():
    def kernel(A):
        cl.atomic_rmw(cl.MemoryOrder.RELAXED, A.pointer(0), A[0])

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        raises=pytest.raises(TypeCheckingError, match="Expected AtomicOp"),
    )
