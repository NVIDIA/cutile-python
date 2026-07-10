# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import operator

import pytest
import torch

import cuda.lang as cl
from cuda.lang._datatype import to_torch_dtype
from cuda.lang._exception import TypeCheckingError, InvalidValueError
from cuda.lang.compilation import KernelSignature


@pytest.mark.parametrize("volatile", [True, False])
@pytest.mark.parametrize("element_count", [2, 4, 8])
@pytest.mark.parametrize(
    "dtype",
    [
        cl.float16,
        cl.float32,
        cl.float64,
        cl.int8,
        cl.int16,
        cl.int32,
        cl.int64,
        cl.bool_,
    ],
)
def test_pointer_vector_ldst(volatile, element_count, dtype):
    assert (element_count & (element_count - 1)) == 0
    alignment = (dtype.bitwidth // 8) * element_count
    values = tuple(i % 2 if dtype is cl.bool_ else i for i in range(element_count))

    @cl.kernel
    def kernel(A):
        with cl.local_array(element_count, dtype, alignment=alignment) as larr:
            for i, value in cl.static_iter(enumerate(values)):
                larr[i] = dtype(value)
            v = larr.get_base_pointer().load(
                count=element_count,
                alignment=alignment,
                volatile=volatile,
            )
        A.get_base_pointer().store(
            v,
            alignment=alignment,
            volatile=volatile,
        )

    A = torch.zeros(element_count, dtype=to_torch_dtype(dtype)).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A,))
    got = A.cpu().tolist()
    expect = torch.tensor(values, dtype=to_torch_dtype(dtype)).tolist()
    assert got == expect, f"{expect=} {got=}"


def test_vector_apis():
    @cl.kernel
    def kernel(out):
        with cl.local_array(4, cl.int32, alignment=16) as larr:
            p = larr.get_base_pointer()
            vec = p.load(count=4, alignment=16)
            out[0] = cl.int32(vec.dtype == larr.dtype)
            out[1] = cl.int32(larr.dtype == cl.int32)
            out[2] = cl.int32(p.pointee_dtype == larr.dtype)
            out[3] = vec.element_count

    out = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [1, 1, 1, 4]


def test_vector_constructor():
    @cl.kernel
    def kernel(out):
        vec = cl.Vector(1, 2, 3, 4)
        out.get_base_pointer().store(vec, alignment=16)

    out = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [1, 2, 3, 4]


def test_vector_constructor_unsigned():
    @cl.kernel
    def kernel(out):
        vec = cl.Vector(cl.uint32(1), 2, 3, 4)
        out.get_base_pointer().store(vec, alignment=16)

    out = torch.zeros(4, dtype=torch.uint32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [1, 2, 3, 4]


def test_vector_constructor_uses_explicit_dtype():
    @cl.kernel
    def kernel(out):
        vec = cl.Vector(1, 2, 3, 4, dtype=cl.int8)
        out.get_base_pointer().store(vec, alignment=4)

    out = torch.zeros(4, dtype=torch.int8).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [1, 2, 3, 4]


def test_vector_constructor_rejects_empty():
    @cl.kernel
    def kernel():
        cl.Vector()

    with pytest.raises(TypeCheckingError, match=r"Vector\(\) expects at least one element"):
        cl.compile_simt(kernel, [KernelSignature([])])


def test_vector_constructor_rejects_non_scalar_element():
    @cl.kernel
    def kernel():
        cl.Vector((1, 2))

    with pytest.raises(TypeCheckingError, match=r"Vector\(\) element 0: Expected a scalar"):
        cl.compile_simt(kernel, [KernelSignature([])])


def test_vector_constructor_explicit_dtype_rejects_out_of_range_element():
    @cl.kernel
    def kernel():
        cl.Vector(1, 300, dtype=cl.int8)

    with pytest.raises(InvalidValueError, match="out of range of int8"):
        cl.compile_simt(kernel, [KernelSignature([])])


def test_vector_constructor_rejects_negative_for_unsigned():
    @cl.kernel
    def kernel():
        cl.Vector(cl.uint32(1), -1)

    with pytest.raises(
        InvalidValueError, match=r"out of range of uint32"
    ):
        cl.compile_simt(kernel, [KernelSignature([])])


def test_vector_constructor_rejects_widen_dtype():
    @cl.kernel
    def kernel():
        cl.Vector(cl.int8(1), 2, 3, 5_000_000_000)

    with pytest.raises(InvalidValueError, match="out of range of int8"):
        cl.compile_simt(kernel, [KernelSignature([])])


@pytest.mark.parametrize(
    "lhs_values,rhs_values",
    [((8, 9, 10, 11), (2, 3, 4, 5))],
)
@pytest.mark.parametrize(
    "dtype",
    [cl.int16, cl.int32, cl.int64, cl.float32, cl.float64],
)
@pytest.mark.parametrize(
    "operation",
    [operator.add, operator.sub, operator.mul, operator.truediv],
)
def test_pointer_vector_arithmetic(operation, dtype, lhs_values, rhs_values):
    expected = operation(
        torch.tensor(lhs_values, dtype=to_torch_dtype(dtype)),
        torch.tensor(rhs_values, dtype=to_torch_dtype(dtype)),
    )
    alignment = (dtype.bitwidth // 8) * 4
    out_alignment = expected.element_size() * 4

    @cl.kernel
    def kernel(out):
        with (
            cl.local_array(4, dtype, alignment=alignment) as lhs,
            cl.local_array(4, dtype, alignment=alignment) as rhs,
        ):
            for i, value in cl.static_iter(enumerate(lhs_values)):
                lhs[i] = dtype(value)
            for i, value in cl.static_iter(enumerate(rhs_values)):
                rhs[i] = dtype(value)
            lhs_vec = lhs.get_base_pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.get_base_pointer().load(count=4, alignment=alignment)
            new = operation(lhs_vec, rhs_vec)
            out.get_base_pointer().store(new, alignment=out_alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize(
    "lhs_values,rhs_values",
    [((9, 10, 11, 12), (2, 3, 4, 5))],
)
@pytest.mark.parametrize("dtype", [cl.int16, cl.int32, cl.int64])
def test_pointer_vector_arithmetic_floordiv(dtype, lhs_values, rhs_values):
    expected = operator.floordiv(
        torch.tensor(lhs_values, dtype=to_torch_dtype(dtype)),
        torch.tensor(rhs_values, dtype=to_torch_dtype(dtype)),
    )
    alignment = (dtype.bitwidth // 8) * 4

    @cl.kernel
    def kernel(out):
        with (
            cl.local_array(4, dtype, alignment=alignment) as lhs,
            cl.local_array(4, dtype, alignment=alignment) as rhs,
        ):
            for i, value in cl.static_iter(enumerate(lhs_values)):
                lhs[i] = dtype(value)
            for i, value in cl.static_iter(enumerate(rhs_values)):
                rhs[i] = dtype(value)
            lhs_vec = lhs.get_base_pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.get_base_pointer().load(count=4, alignment=alignment)
            new = operator.floordiv(lhs_vec, rhs_vec)
            out.get_base_pointer().store(new, alignment=alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize(
    "lhs_values,rhs_values",
    [((0b1100, 0b1010, 0b0110, 0b0011), (0b1010, 0b0101, 0b0011, 0b1111))],
)
@pytest.mark.parametrize(
    "dtype",
    [
        cl.int8,
        cl.int16,
        cl.int32,
        cl.int64,
        cl.uint8,
        cl.uint16,
        cl.uint32,
        cl.uint64,
    ],
)
@pytest.mark.parametrize(
    "operation",
    [operator.and_, operator.or_, operator.xor],
)
def test_pointer_vector_arithmetic_bitwise(operation, dtype, lhs_values, rhs_values):
    expected = torch.tensor(
        [operation(lhs, rhs) for lhs, rhs in zip(lhs_values, rhs_values)],
        dtype=to_torch_dtype(dtype),
    )
    alignment = (dtype.bitwidth // 8) * 4

    @cl.kernel
    def kernel(out):
        with (
            cl.local_array(4, dtype, alignment=alignment) as lhs,
            cl.local_array(4, dtype, alignment=alignment) as rhs,
        ):
            for i, value in cl.static_iter(enumerate(lhs_values)):
                lhs[i] = dtype(value)
            for i, value in cl.static_iter(enumerate(rhs_values)):
                rhs[i] = dtype(value)
            lhs_vec = lhs.get_base_pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.get_base_pointer().load(count=4, alignment=alignment)
            new = operation(lhs_vec, rhs_vec)
            out.get_base_pointer().store(new, alignment=alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize(
    "lhs_values,rhs_values",
    [((1, 2, 3, 4), (2, 2, 2, 2))],
)
@pytest.mark.parametrize(
    "dtype",
    [cl.int32, cl.int64, cl.float32, cl.float64],
)
@pytest.mark.parametrize(
    "operation",
    [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne],
)
def test_pointer_vector_arithmetic_comparison(operation, dtype, lhs_values, rhs_values):
    expected = operation(
        torch.tensor(lhs_values, dtype=to_torch_dtype(dtype)),
        torch.tensor(rhs_values, dtype=to_torch_dtype(dtype)),
    )
    alignment = (dtype.bitwidth // 8) * 4
    out_alignment = expected.element_size() * 4

    @cl.kernel
    def kernel(out):
        with (
            cl.local_array(4, dtype, alignment=alignment) as lhs,
            cl.local_array(4, dtype, alignment=alignment) as rhs,
        ):
            for i, value in cl.static_iter(enumerate(lhs_values)):
                lhs[i] = dtype(value)
            for i, value in cl.static_iter(enumerate(rhs_values)):
                rhs[i] = dtype(value)
            lhs_vec = lhs.get_base_pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.get_base_pointer().load(count=4, alignment=alignment)
            new = operation(lhs_vec, rhs_vec)
            out.get_base_pointer().store(new, alignment=out_alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize(
    "lhs_values,rhs_values",
    [((1, 2, 3, 4), (1, 2, 3, 4))],
)
@pytest.mark.parametrize("dtype", [cl.int32, cl.uint32])
@pytest.mark.parametrize(
    "operation",
    [operator.lshift, operator.rshift],
)
def test_pointer_vector_arithmetic_shift(operation, dtype, lhs_values, rhs_values):
    expected = torch.tensor(
        [operation(lhs, rhs) for lhs, rhs in zip(lhs_values, rhs_values)],
        dtype=to_torch_dtype(dtype),
    )
    alignment = (dtype.bitwidth // 8) * 4

    @cl.kernel
    def kernel(out):
        with (
            cl.local_array(4, dtype, alignment=alignment) as lhs,
            cl.local_array(4, dtype, alignment=alignment) as rhs,
        ):
            for i, value in cl.static_iter(enumerate(lhs_values)):
                lhs[i] = dtype(value)
            for i, value in cl.static_iter(enumerate(rhs_values)):
                rhs[i] = dtype(value)
            lhs_vec = lhs.get_base_pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.get_base_pointer().load(count=4, alignment=alignment)
            new = operation(lhs_vec, rhs_vec)
            out.get_base_pointer().store(new, alignment=alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize("values", [(1, -2, 3, -4)])
@pytest.mark.parametrize("dtype", [cl.int32, cl.float32, cl.float64])
@pytest.mark.parametrize(
    "operation",
    [operator.pos, operator.neg],
)
def test_pointer_vector_arithmetic_unary(operation, dtype, values):
    expected = torch.tensor(
        [operation(value) for value in values],
        dtype=to_torch_dtype(dtype),
    )
    alignment = (dtype.bitwidth // 8) * 4

    @cl.kernel
    def kernel(out):
        with cl.local_array(4, dtype, alignment=alignment) as value:
            for i, item in cl.static_iter(enumerate(values)):
                value[i] = dtype(item)
            vec = value.get_base_pointer().load(count=4, alignment=alignment)
            new = operation(vec)
            out.get_base_pointer().store(new, alignment=alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    torch.testing.assert_close(out.cpu(), expected)


def test_pointer_vector_count_can_be_non_power_of_two():
    @cl.kernel
    def kernel(out):
        out.get_base_pointer().load(count=3, alignment=4)

    out = torch.zeros(3, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))


def test_vector_getitem():
    @cl.kernel
    def kernel(tensor):
        v4 = tensor.get_base_pointer().load(count=4)
        tensor[0] = v4[3]
        tensor[1] = v4[2]
        tensor[2] = v4[1]
        tensor[3] = v4[0]

    tensor = torch.tensor(list(range(4)), dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor,))
    assert tensor.cpu().tolist() == [3, 2, 1, 0]


def test_vector_setitem():
    @cl.kernel
    def kernel():
        with cl.local_array(4, cl.int32) as arr:
            v = arr.get_base_pointer().load(count=4)
            v[0] = 1

    with pytest.raises(
        TypeCheckingError, match="Vectors are immutable: item assignment is not supported"
    ):
        cl.compile_simt(kernel, [KernelSignature([])])


def test_vector_from_tuple():
    @cl.kernel
    def kernel(tensor):
        v4 = cl.Vector(*tuple(i for i in cl.static_iter(range(4))))
        tensor.get_base_pointer().store(v4, alignment=16)

    tensor = torch.zeros(4, dtype=torch.int32, device='cuda')
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor,))
    assert tensor.cpu().tolist() == [0, 1, 2, 3]
