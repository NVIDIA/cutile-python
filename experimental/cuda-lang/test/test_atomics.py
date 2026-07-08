# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.lang as cl
from cuda.lang.compilation import KernelSignature
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.ops import AtomicCAS, AtomicExchange, AtomicRMW

from .util import compile_kernel, get_ir, make_symbolic_tensor


ALL_INT_DTYPES = ["int32", "int64"]
ALL_UINT_DTYPES = ["uint32", "uint64"]
ALL_FLOAT_DTYPES = ["float32", "float64"]
ALL_REAL_DTYPES = ALL_INT_DTYPES + ALL_UINT_DTYPES + ALL_FLOAT_DTYPES
ALL_INTEGER_DTYPES = ALL_INT_DTYPES + ALL_UINT_DTYPES


def _torch_dtype(dtype):
    return getattr(torch, dtype)


def _cl_dtype(dtype):
    return getattr(cl, dtype)


def _scalar(dtype, value):
    return _cl_dtype(dtype)(value)


RMW_CASES = [
    ("atomic_add", ALL_REAL_DTYPES, 7, 3, 10),
    ("atomic_sub", ALL_REAL_DTYPES, 7, 3, 4),
    ("atomic_and", ALL_INTEGER_DTYPES, 0b1110, 0b1011, 0b1010),
    ("atomic_or", ALL_INTEGER_DTYPES, 0b1100, 0b0011, 0b1111),
    ("atomic_xor", ALL_INTEGER_DTYPES, 0b1100, 0b1010, 0b0110),
    ("atomic_min", ALL_REAL_DTYPES, 7, 3, 3),
    ("atomic_max", ALL_REAL_DTYPES, 7, 11, 11),
    ("atomic_inc", ["uint32"], 7, 11, 8),
    ("atomic_dec", ["uint32"], 7, 11, 6),
]

RMW_VARIANTS = [
    (op, dtype, initial, update, expected_new)
    for op, dtypes, initial, update, expected_new in RMW_CASES
    for dtype in dtypes
]

UNSUPPORTED_DTYPE_CASES = [
    ("atomic_add", "int16"),
    ("atomic_sub", "float16"),
    ("atomic_and", "float32"),
    ("atomic_or", "float32"),
    ("atomic_xor", "float32"),
    ("atomic_min", "float16"),
    ("atomic_max", "float16"),
    ("atomic_inc", "uint64"),
    ("atomic_dec", "uint64"),
    ("atomic_xchg", "int16"),
    ("atomic_cas", "float32"),
]

ATOMIC_MEMORY_ARGUMENT_CASES = [
    ("atomic_add", "int32", AtomicRMW),
    ("atomic_sub", "int32", AtomicRMW),
    ("atomic_and", "int32", AtomicRMW),
    ("atomic_or", "int32", AtomicRMW),
    ("atomic_xor", "int32", AtomicRMW),
    ("atomic_min", "int32", AtomicRMW),
    ("atomic_max", "int32", AtomicRMW),
    ("atomic_inc", "uint32", AtomicRMW),
    ("atomic_dec", "uint32", AtomicRMW),
    ("atomic_xchg", "int32", AtomicExchange),
    ("atomic_cas", "int32", AtomicCAS),
]


@pytest.mark.parametrize("op,dtype,initial,update,expected_new", RMW_VARIANTS)
def test_atomic_rmw_supported_types(op, dtype, initial, update, expected_new):
    atomic = getattr(cl, op)
    torch_dtype = _torch_dtype(dtype)

    @cl.kernel
    def kernel(A, out):
        ptr = A.get_element_pointer(0)
        out[0] = atomic(ptr, _scalar(dtype, update))

    A = torch.tensor([initial], dtype=torch_dtype, device="cuda")
    out = torch.zeros(1, dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert torch.allclose(out.cpu(), torch.tensor([initial], dtype=torch_dtype))
    assert torch.allclose(A.cpu(), torch.tensor([expected_new], dtype=torch_dtype))


@pytest.mark.parametrize("dtype", ALL_REAL_DTYPES)
def test_atomic_xchg_supported_types(dtype):
    torch_dtype = _torch_dtype(dtype)

    @cl.kernel
    def kernel(A, out):
        ptr = A.get_element_pointer(0)
        out[0] = cl.atomic_xchg(ptr, _scalar(dtype, 11))

    A = torch.tensor([7], dtype=torch_dtype, device="cuda")
    out = torch.zeros(1, dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert torch.allclose(out.cpu(), torch.tensor([7], dtype=torch_dtype))
    assert torch.allclose(A.cpu(), torch.tensor([11], dtype=torch_dtype))


@pytest.mark.parametrize("dtype", ALL_INTEGER_DTYPES)
def test_atomic_cas_supported_types(dtype):
    torch_dtype = _torch_dtype(dtype)

    @cl.kernel
    def kernel(A, out):
        ptr = A.get_element_pointer(0)
        out[0] = cl.atomic_cas(ptr, _scalar(dtype, 7), _scalar(dtype, 11))

    A = torch.tensor([7], dtype=torch_dtype, device="cuda")
    out = torch.zeros(1, dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert torch.allclose(out.cpu(), torch.tensor([7], dtype=torch_dtype))
    assert torch.allclose(A.cpu(), torch.tensor([11], dtype=torch_dtype))


def test_atomic_cas_failure():
    @cl.kernel
    def kernel(A, out):
        ptr = A.get_element_pointer(0)
        out[0] = cl.atomic_cas(ptr, cl.int32(8), cl.int32(11))

    A = torch.tensor([7], dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 7
    assert A.cpu()[0].item() == 7


def test_atomic_inc_wrap():
    @cl.kernel
    def kernel(A, out):
        ptr = A.get_element_pointer(0)
        out[0] = cl.atomic_inc(ptr, cl.uint32(7))

    A = torch.tensor([7], dtype=torch.uint32, device="cuda")
    out = torch.zeros(1, dtype=torch.uint32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 7
    assert A.cpu()[0].item() == 0


def test_atomic_dec_wrap():
    @cl.kernel
    def kernel(A, out):
        ptr = A.get_element_pointer(0)
        out[0] = cl.atomic_dec(ptr, cl.uint32(7))

    A = torch.tensor([0], dtype=torch.uint32, device="cuda")
    out = torch.zeros(1, dtype=torch.uint32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 0
    assert A.cpu()[0].item() == 7


def test_atomic_tuple_index():
    @cl.kernel
    def kernel(A, out):
        ptr = A.get_element_pointer((0, 1))
        out[0] = cl.atomic_add(ptr, cl.int32(5))

    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 2
    assert A.cpu()[0, 1].item() == 7


@pytest.mark.parametrize("op,dtype", UNSUPPORTED_DTYPE_CASES)
def test_atomic_unsupported_dtypes(op, dtype):
    atomic = getattr(cl, op)
    cl_dtype = _cl_dtype(dtype)

    def kernel(A):
        ptr = A.get_element_pointer(0)
        if op == "atomic_cas":
            atomic(ptr, A[0], A[0])
        else:
            atomic(ptr, A[0])

    compile_kernel(
        kernel,
        signature=KernelSignature(
            (make_symbolic_tensor(shape=(1,), dtype=cl_dtype),)
        ),
        raises=pytest.raises(
            TypeCheckingError, match=f"{op} does not support dtype {dtype}"
        ),
    )


@pytest.mark.parametrize("order,scope,msg",
                         [(cl.MemoryOrder.WEAK, cl.MemoryScope.DEVICE, "Invalid memory order"),
                          (cl.MemoryOrder.RELEASE, cl.MemoryScope.NONE, "Invalid memory scope")])
def test_atomic_unsupported_memory_order_scope(order, scope, msg):
    def kernel(A):
        ptr = A.get_element_pointer(0)
        cl.atomic_add(ptr, A[0], memory_order=order, memory_scope=scope)

    with pytest.raises(TypeCheckingError, match=msg):
        get_ir(kernel, [make_symbolic_tensor(shape=(1,), dtype=cl.int32)])
