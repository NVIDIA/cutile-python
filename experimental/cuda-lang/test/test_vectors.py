# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from test.util import compile_kernel, make_symbolic_tensor
import operator

import pytest
import torch

import cuda.lang as cl
from cuda.lang._datatype import to_torch_dtype
from cuda.lang._exception import TypeCheckingError, InvalidValueError
from cuda.lang.compilation import KernelSignature, ScalarConstraint


def test_load_store_vector_free_functions():
    @cl.kernel
    def kernel(inp_1, inp_2, out_1, out_2):
        vector_1 = inp_1.pointer().load(count=4, alignment=16)
        out_1.pointer().store(vector_1, alignment=16)

        vector_2 = cl.load(inp_2.pointer(), count=4, alignment=16)
        cl.store(out_2.pointer(), vector_2, alignment=16)

    inp_1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).cuda()
    out_1 = torch.zeros((4,), dtype=torch.float32).cuda()
    inp_2 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).cuda()
    out_2 = torch.zeros((4,), dtype=torch.float32).cuda()

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp_1, inp_2, out_1, out_2))
    assert out_1.cpu().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out_2.cpu().tolist() == [1.0, 2.0, 3.0, 4.0]


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
def test_pointer_vector_ldst(element_count, dtype):
    assert (element_count & (element_count - 1)) == 0
    alignment = (dtype.bitwidth // 8) * element_count
    values = tuple(i % 2 if dtype is cl.bool_ else i for i in range(element_count))

    @cl.kernel
    def kernel(A):
        with cl.local_array(element_count, dtype, alignment=alignment) as larr:
            for i, value in cl.static_iter(enumerate(values)):
                larr[i] = dtype(value)
            v = larr.pointer().load(
                count=element_count,
                alignment=alignment,
            )
        A.pointer().store(
            v,
            alignment=alignment,
        )

    A = torch.zeros(element_count, dtype=to_torch_dtype(dtype)).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A,))
    got = A.cpu().tolist()
    expect = torch.tensor(values, dtype=to_torch_dtype(dtype)).tolist()
    assert got == expect, f"{expect=} {got=}"


def test_vector_apis():
    @cl.kernel
    def kernel(out):
        with cl.local_array(4, cl.int32, alignment=16) as larr:
            p = larr.pointer()
            vec = p.load(count=4, alignment=16)
            out[0] = cl.int32(vec.dtype == larr.dtype)
            out[1] = cl.int32(larr.dtype == cl.int32)
            out[2] = cl.int32(p.pointee_dtype == larr.dtype)
            out[3] = vec.element_count

    out = torch.zeros(4, dtype=torch.int32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [1, 1, 1, 4]


def test_astype_on_vector():
    @cl.kernel
    def kernel(inp, out):
        vector = inp.pointer().load(count=4, alignment=16)
        halved = vector.astype(cl.float16)
        out.pointer().store(halved, alignment=8)

    values = [1.0, 2.0, 3.0, 4.0]
    inp = torch.tensor(values, dtype=torch.float32).cuda(0)
    out = torch.zeros(4, dtype=torch.float16).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out.cpu().tolist() == values


@pytest.mark.parametrize('length', (2, 4))
def test_vector_tuple(length):
    expect = tuple(range(length))

    @cl.kernel
    def kernel(input, output):
        vector = input.pointer().load(count=length, alignment=16)
        elements = tuple(vector)
        output[0] = elements == expect

    input = torch.arange(4, dtype=torch.int32, device="cuda:0")
    output = torch.tensor([False], dtype=torch.bool, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (input, output))
    assert output.cpu().item()


@pytest.mark.parametrize('length', (2, 4))
def test_vector_tuple_len(length):
    @cl.kernel
    def kernel(input, output):
        vector = input.pointer().load(count=length, alignment=16)
        output[0] = len(vector)

    input = torch.arange(4, dtype=torch.int32, device="cuda:0")
    output = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (input, output))
    assert output.cpu().item() == length


@pytest.mark.parametrize("length", (2, 4))
def test_vector_len_in_static_iter(length):
    @cl.kernel
    def kernel(input, output):
        vector = input.pointer().load(count=length, alignment=16)
        for index in cl.static_iter(range(len(vector))):
            output[index] = vector[index]

    input = torch.arange(4, dtype=torch.int32, device="cuda:0")
    output = torch.zeros(length, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (input, output))
    assert output.cpu().tolist() == list(range(length))


def test_tuple_rejects_non_iterable():
    def kernel():
        tuple(1)

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TypeCheckingError,
            match="Object of type int32 cannot be converted to a tuple",
        ),
    )


def test_vector_constructor():
    @cl.kernel
    def kernel(out):
        vec = cl.Vector(1, 2, 3, 4)
        out.pointer().store(vec, alignment=16)

    out = torch.zeros(4, dtype=torch.int32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [1, 2, 3, 4]


def test_vector_constructor_unsigned():
    @cl.kernel
    def kernel(out):
        vec = cl.Vector(cl.uint32(1), 2, 3, 4)
        out.pointer().store(vec, alignment=16)

    out = torch.zeros(4, dtype=torch.uint32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [1, 2, 3, 4]


def test_vector_constructor_uses_explicit_dtype():
    @cl.kernel
    def kernel(out):
        vec = cl.Vector(1, 2, 3, 4, dtype=cl.int8)
        out.pointer().store(vec, alignment=4)

    out = torch.zeros(4, dtype=torch.int8).cuda(0)
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
            lhs_vec = lhs.pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.pointer().load(count=4, alignment=alignment)
            new = operation(lhs_vec, rhs_vec)
            out.pointer().store(new, alignment=out_alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda(0)
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
            lhs_vec = lhs.pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.pointer().load(count=4, alignment=alignment)
            new = operator.floordiv(lhs_vec, rhs_vec)
            out.pointer().store(new, alignment=alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda(0)
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
            lhs_vec = lhs.pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.pointer().load(count=4, alignment=alignment)
            new = operation(lhs_vec, rhs_vec)
            out.pointer().store(new, alignment=alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda(0)
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
            lhs_vec = lhs.pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.pointer().load(count=4, alignment=alignment)
            new = operation(lhs_vec, rhs_vec)
            out.pointer().store(new, alignment=out_alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda(0)
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
            lhs_vec = lhs.pointer().load(count=4, alignment=alignment)
            rhs_vec = rhs.pointer().load(count=4, alignment=alignment)
            new = operation(lhs_vec, rhs_vec)
            out.pointer().store(new, alignment=alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda(0)
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
            vec = value.pointer().load(count=4, alignment=alignment)
            new = operation(vec)
            out.pointer().store(new, alignment=alignment)

    out = torch.zeros(4, dtype=expected.dtype).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    torch.testing.assert_close(out.cpu(), expected)


def test_pointer_vector_count_can_be_non_power_of_two():
    @cl.kernel
    def kernel(out):
        out.pointer().load(count=3, alignment=4)

    out = torch.zeros(3, dtype=torch.int32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))


def test_vector_getitem():
    @cl.kernel
    def kernel(tensor):
        v4 = tensor.pointer().load(count=4)
        tensor[0] = v4[3]
        tensor[1] = v4[2]
        tensor[2] = v4[1]
        tensor[3] = v4[0]

    tensor = torch.tensor(list(range(4)), dtype=torch.int32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor,))
    assert tensor.cpu().tolist() == [3, 2, 1, 0]


def test_vector_setitem():
    def kernel():
        v = cl.shared_array(1, cl.int8).pointer().load(count=2)
        v[0] = 1

    compile_kernel(
        kernel,
        raises=pytest.raises(TypeCheckingError, match="Vectors are immutable"),
    )


def test_vector_with_item():
    @cl.kernel
    def kernel(original, updated):
        original_vector = original.pointer().load(count=4, alignment=16)
        updated_vector = original_vector.with_item(2, 42)
        updated.pointer().store(updated_vector, alignment=16)

    a = torch.arange(4, dtype=torch.int32, device="cuda:0")
    b = torch.arange(4, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b))
    assert a.cpu().tolist() == [0, 1, 2, 3]
    assert b.cpu().tolist() == [0, 1, 42, 3]


def test_vector_from_tuple():
    @cl.kernel
    def kernel(tensor):
        v4 = cl.Vector(*tuple(i for i in cl.static_iter(range(4))))
        tensor.pointer().store(v4, alignment=16)

    tensor = torch.zeros(4, dtype=torch.int32, device='cuda:0')
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor,))
    assert tensor.cpu().tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "dtype,op,propagate_nan,kind",
    (
        (cl.int32, cl.VectorReduction.add, False, "add"),
        (cl.int32, cl.VectorReduction.mul, False, "mul"),
        (cl.int32, cl.VectorReduction.bitwise_and, False, "and"),
        (cl.int32, cl.VectorReduction.bitwise_or, False, "or"),
        (cl.int32, cl.VectorReduction.bitwise_xor, False, "xor"),
        (cl.int32, cl.VectorReduction.max, False, "smax"),
        (cl.int32, cl.VectorReduction.min, False, "smin"),
        (cl.uint32, cl.VectorReduction.max, False, "umax"),
        (cl.uint32, cl.VectorReduction.min, False, "umin"),
        (cl.float32, cl.VectorReduction.add, False, "fadd"),
        (cl.float32, cl.VectorReduction.mul, False, "fmul"),
        (cl.float32, cl.VectorReduction.max, False, "fmax"),
        (cl.float32, cl.VectorReduction.min, False, "fmin"),
        (cl.float32, cl.VectorReduction.max, True, "fmaximum"),
        (cl.float32, cl.VectorReduction.min, True, "fminimum"),
    ),
)
def test_vector_reduce_mlir(dtype, op, propagate_nan, kind):
    @cl.kernel
    def kernel(output):
        vector = cl.Vector(1, 2, 3, 4, dtype=dtype)
        output[0] = vector.reduce(op, propagate_nan=propagate_nan)

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, dtype),)),
        assert_in_mlir=f'llvm.intr.vector.reduce.{kind}',
    )


@pytest.mark.parametrize(
    "op,mlir_op",
    (
        (cl.VectorReduction.add, "fadd"),
        (cl.VectorReduction.mul, "fmul"),
    ),
)
def test_vector_reduce_reassociate_mlir(op, mlir_op):
    mlir_op = 'llvm.intr.vector.reduce.' + mlir_op

    @cl.kernel
    def kernel(output):
        vector = cl.Vector(2.0, 3.0, 4.0)
        output[0] = vector.reduce(op, reassociate=True)

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.float32),)),
        assert_in_mlir=(mlir_op, "fastmath <reassoc>"),
    )


@pytest.mark.parametrize(
    "dtype,op,values,expected",
    (
        (cl.int32, cl.VectorReduction.add, (2, -3, 4, 5), 8),
        (cl.int32, cl.VectorReduction.mul, (2, -3, 4, 5), -120),
        (cl.int32, cl.VectorReduction.bitwise_and, (15, 7, 3, 11), 3),
        (cl.int32, cl.VectorReduction.bitwise_or, (8, 4, 2, 1), 15),
        (cl.int32, cl.VectorReduction.bitwise_xor, (8, 4, 2, 1), 15),
        (cl.int32, cl.VectorReduction.max, (2, -3, 4, 5), 5),
        (cl.int32, cl.VectorReduction.min, (2, -3, 4, 5), -3),
        (cl.uint32, cl.VectorReduction.max, (2, 3, 4, 5), 5),
        (cl.uint32, cl.VectorReduction.min, (2, 3, 4, 5), 2),
        (cl.float32, cl.VectorReduction.add, (2, -3, 4, 5), 8),
        (cl.float32, cl.VectorReduction.mul, (2, -3, 4, 5), -120),
        (cl.float32, cl.VectorReduction.max, (2, -3, 4, 5), 5),
        (cl.float32, cl.VectorReduction.min, (2, -3, 4, 5), -3),
        (cl.bool_, cl.VectorReduction.bitwise_and, (True, True, False, True), False),
        (cl.bool_, cl.VectorReduction.bitwise_or, (False, False, True, False), True),
        (cl.bool_, cl.VectorReduction.bitwise_xor, (True, False, True, True), True),
    ),
)
def test_vector_reduce(dtype, op, values, expected):
    @cl.kernel
    def kernel(output):
        vector = cl.Vector(
            dtype(values[0]),
            dtype(values[1]),
            dtype(values[2]),
            dtype(values[3]),
        )
        output[0] = vector.reduce(op)

    output = torch.zeros(1, dtype=to_torch_dtype(dtype), device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (output,))
    assert output.cpu().item() == expected


@pytest.mark.parametrize("op", (cl.VectorReduction.max, cl.VectorReduction.min))
def test_vector_reduce_propagate_nan(op):
    @cl.kernel
    def kernel(output):
        vector = cl.Vector(float("nan"), 3.0, 2.0)
        output[0] = vector.reduce(op)
        output[1] = vector.reduce(op, propagate_nan=True)

    output = torch.zeros(2, dtype=torch.float32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (output,))
    got = output.cpu()
    assert got[0].item() == (3.0 if op is cl.VectorReduction.max else 2.0)
    assert torch.isnan(got[1])


def test_vector_reduce_signed_zero():
    @cl.kernel
    def kernel(output):
        vector = cl.Vector(-0.0, 0.0)
        output[0] = vector.reduce(cl.VectorReduction.max, propagate_nan=True)
        output[1] = vector.reduce(cl.VectorReduction.min, propagate_nan=True)

    output = torch.zeros(2, dtype=torch.float32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (output,))
    got = output.cpu()
    assert not torch.signbit(got[0])
    assert torch.signbit(got[1])


def test_vector_reduce_float_order():
    @cl.kernel
    def kernel(output):
        add_values = cl.Vector(1.0e20, -1.0e20, 1.0)
        mul_values = cl.Vector(1.0e20, 1.0e20, 1.0e-20, 1.0e-20)
        output[0] = add_values.reduce(cl.VectorReduction.add)
        output[1] = mul_values.reduce(cl.VectorReduction.mul)

    output = torch.zeros(2, dtype=torch.float32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (output,))
    got = output.cpu()
    assert got[0].item() == 1.0
    assert torch.isinf(got[1])


def test_vector_reduce_integer_overflow():
    @cl.kernel
    def kernel(output):
        vector = cl.Vector(cl.int8(120), cl.int8(120))
        output[0] = vector.reduce(cl.VectorReduction.add)

    output = torch.zeros(1, dtype=torch.int8, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (output,))
    assert output.cpu().item() == -16


@pytest.mark.parametrize(
    "dtype,op",
    (
        (cl.bool_, cl.VectorReduction.add),
        (cl.bool_, cl.VectorReduction.mul),
        (cl.bool_, cl.VectorReduction.max),
        (cl.bool_, cl.VectorReduction.min),
        (cl.float32, cl.VectorReduction.bitwise_and),
        (cl.float32, cl.VectorReduction.bitwise_or),
        (cl.float32, cl.VectorReduction.bitwise_xor),
    ),
)
def test_vector_reduce_rejects_unsupported_dtype(dtype, op):
    def kernel():
        cl.Vector(dtype(1), dtype(2)).reduce(op)

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TypeCheckingError,
            match=f"Vector reduction {op.value} does not support {dtype}",
        ),
    )


@pytest.mark.parametrize('op', (
    cl.VectorReduction.add,
    cl.VectorReduction.mul,
    cl.VectorReduction.bitwise_and,
    cl.VectorReduction.bitwise_or,
    cl.VectorReduction.bitwise_xor,
))
def test_vector_reduce_rejects_invalid_propagate_nan(op):
    def kernel():
        cl.Vector(1.0, 2.0).reduce(op, propagate_nan=True)

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TypeCheckingError,
            match="propagate_nan is valid only for min and max",
        ),
    )


@pytest.mark.parametrize(
    "dtype,op",
    (
        (cl.int32, cl.VectorReduction.add),
        (cl.int32, cl.VectorReduction.mul),
        (cl.int32, cl.VectorReduction.bitwise_and),
        (cl.float32, cl.VectorReduction.max),
        (cl.float32, cl.VectorReduction.min),
    ),
)
def test_vector_reduce_rejects_invalid_reassociate(dtype, op):
    def kernel():
        cl.Vector(dtype(1), dtype(2)).reduce(op, reassociate=True)

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TypeCheckingError,
            match=(
                "reassociate is valid only for floating-point add and multiply "
                "vector reductions"
            ),
        ),
    )


def test_vector_reduce_rejects_wrong_enum():
    def kernel():
        cl.Vector(1, 2).reduce(cl.BarrierReductionKind.AND)

    compile_kernel(
        kernel,
        raises=pytest.raises(TypeCheckingError, match="Expected VectorReduction"),
    )


def test_vector_reduce_requires_constant_propagate_nan():
    @cl.kernel
    def kernel(propagate_nan):
        cl.Vector(1.0, 2.0).reduce(
            cl.VectorReduction.max,
            propagate_nan=propagate_nan,
        )

    compile_kernel(
        kernel,
        signature=KernelSignature([ScalarConstraint(cl.bool_)]),
        raises=pytest.raises(TypeCheckingError, match="Expected a boolean constant"),
    )


def test_vector_reduce_requires_constant_reassociate():
    def kernel(reassociate):
        cl.Vector(1.0, 2.0).reduce(
            cl.VectorReduction.add,
            reassociate=reassociate,
        )

    compile_kernel(
        kernel,
        signature=KernelSignature([ScalarConstraint(cl.bool_)]),
        raises=pytest.raises(TypeCheckingError, match="Expected a boolean constant"),
    )


class TestVectorSlice:
    def _transform_vector(self, function):

        @cl.kernel
        def kernel(inp: cl.Array, out: cl.Array):
            v = inp.pointer(0).load(count=8)
            v2 = function(v)
            out.pointer(0).store(v2)

        inp = torch.arange(8, dtype=torch.int8).cuda(0)
        out = torch.zeros(8, dtype=torch.int8).cuda(0)
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
        expect = function(list(range(8)))
        out = out.cpu().tolist()
        assert out[: len(expect)] == expect
        assert all(map(lambda x: x == 0, out[len(expect):]))

    def test_start_stop_step(self):
        def f(v):
            return v[1:8:2]

        self._transform_vector(f)

    def test_start_stop_nostep(self):
        def f(v):
            return v[1:6]

        self._transform_vector(f)

    def test_start_nostop_nostep(self):
        def f(v):
            return v[1:]

        self._transform_vector(f)

    def test_nostart_nostop_nostep(self):
        def f(v):
            return v[:]

        self._transform_vector(f)

    def test_nostart_stop_nostep(self):
        def f(v):
            return v[:4]

        self._transform_vector(f)

    def test_nostart_stop_step(self):
        def f(v):
            return v[:6:2]

        self._transform_vector(f)

    def test_nostart_nostop_step(self):
        def f(v):
            return v[::2]

        self._transform_vector(f)

    def test_negative_step(self):
        def f(v):
            return v[::-2]

        self._transform_vector(f)

    def test_negative_start_stop(self):
        def f(v):
            return v[-5:-1]

        self._transform_vector(f)

    def test_same_result_type(self):
        @cl.kernel
        def kernel():
            v = cl.Vector(0, 1, 2, 3, 4, 5, 6, 7, dtype=cl.int8)
            reversed: cl.Vector = v[::-1]
            check = reversed.dtype == cl.int8
            cl.static_assert(check)
            cl.static_assert(reversed.element_count == 8)
            halved: cl.Vector = v[: len(v) // 2]

            check = halved.dtype == cl.int8
            cl.static_assert(check)
            cl.static_assert(halved.element_count == 4)

        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())

    def test_reject_dynamic_slice_start(self):
        def kernel():
            dyn = cl.shared_array(1, cl.int8)[0]
            cl.Vector(0, 1, 2, 3)[dyn:]

        compile_kernel(
            kernel,
            raises=pytest.raises(
                TypeCheckingError, match="Non-constant slices are not supported"
            ),
        )

    def test_reject_dynamic_slice_stop(self):
        def kernel():
            dyn = cl.shared_array(1, cl.int8)[0]
            cl.Vector(0, 1, 2, 3)[:dyn]

        compile_kernel(
            kernel,
            raises=pytest.raises(
                TypeCheckingError, match="Non-constant slices are not supported"
            ),
        )

    def test_reject_dynamic_slice_step(self):
        def kernel():
            dyn = cl.shared_array(1, cl.int8)[0]
            cl.Vector(0, 1, 2, 3)[::dyn]

        compile_kernel(
            kernel,
            raises=pytest.raises(
                TypeCheckingError, match="Non-constant slices are not supported"
            ),
        )

    def test_reject_0_step(self):
        def kernel():
            cl.Vector(0, 1, 2, 3)[::0]

        compile_kernel(
            kernel,
            raises=pytest.raises(InvalidValueError, match="Slice step cannot be zero"),
        )

    def test_reject_float_slice(self):
        def kernel():
            cl.Vector(0, 1, 2, 3)[1.0:]

        compile_kernel(
            kernel,
            raises=pytest.raises(Exception, match="slice indices must be integers"),
        )


def test_reinterpret_as_scalar():
    # Whole-vector reinterpret to a single scalar of the total width.
    @cl.kernel
    def kernel(inp, out):
        v = inp.pointer().load(count=2)
        out[0] = v.reinterpret_as_scalar(cl.int64)

    inp = torch.tensor([1, 2], dtype=torch.int32).cuda(0)
    out = torch.zeros(1, dtype=torch.int64).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    got = out.cpu().item()
    assert got == ((2 << 32) | 1), f"{got:x}"


def test_reinterpret_as_scalar_width_mismatch_errors():
    # The target scalar's bitwidth must equal the vector's total bitwidth.
    @cl.kernel
    def kernel(inp, out):
        v = inp.pointer().load(count=2)     # Vector[int32, 2] = 64 bit
        out[0] = v.reinterpret_as_scalar(cl.int32)   # target is 32 bit != 64

    match = "bitcast requires input value's type and output type to have the same bitwidth"
    with pytest.raises(TypeCheckingError, match=match):
        cl.compile_simt(
            kernel,
            [KernelSignature([make_symbolic_tensor(1, cl.int32),
                              make_symbolic_tensor(1, cl.int32)])],
        )


def test_reinterpret_as_vector_reshape():
    # Whole-vector reinterpret to a differently-shaped vector of the same total
    # width: Vector[float32, 4] -> Vector[int8, 16].
    @cl.kernel
    def kernel(inp, out):
        v = inp.pointer().load(count=4)
        out.pointer().store(v.reinterpret_as_vector(cl.int8, 16))

    values = torch.tensor([1.5, -2.25, 3.75, 0.5], dtype=torch.float32)
    inp = values.cuda(0)
    out = torch.zeros(16, dtype=torch.int8).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out.cpu().tolist() == values.view(torch.int8).tolist()


def test_reinterpret_as_vector_width_mismatch_errors():
    @cl.kernel
    def kernel(inp, out):
        v = inp.pointer().load(count=4)
        out.pointer().store(v.reinterpret_as_vector(cl.int8, 15))

    match = "bitcast requires input value's type and output type to have the same bitwidth"
    with pytest.raises(TypeCheckingError, match=match):
        cl.compile_simt(
            kernel,
            [KernelSignature([make_symbolic_tensor(1, cl.float32),
                              make_symbolic_tensor(1, cl.int8)])],
        )


def test_reinterpret_as_vector_pointer():
    @cl.kernel
    def kernel(inp, out):
        vector = inp.pointer().load(count=4)
        pointers = vector.reinterpret_as_vector(cl.pointer_dtype(cl.float32), 2)
        out.pointer().store(pointers.reinterpret_as_vector(cl.int32, 4))

    inp = torch.arange(4, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(inp)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    torch.testing.assert_close(out, inp)


def test_vector_of_pointer():
    @cl.kernel
    def kernel(N: cl.Constant, out: cl.Array):
        pointers = cl.shared_array(N, cl.pointer_dtype(cl.int8))
        values = cl.shared_array(N, cl.int8)
        for i in range(N):
            pointers[i] = values.pointer(i)

        vector = pointers.pointer().load(count=N)

        for i in range(N):
            vector[i][0] = i
            out[i] = values[i]

    N = 4
    out = torch.zeros(N, dtype=torch.int8, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (N, out))
    assert out.cpu().tolist() == list(range(N))


def test_vector_of_pointer_access_and_store():
    @cl.kernel
    def kernel(N: cl.Constant, out: cl.Array):
        pointers = cl.shared_array(N, cl.pointer_dtype(cl.int8))
        copied = cl.shared_array(N, cl.pointer_dtype(cl.int8))
        values = cl.shared_array(N, cl.int8)
        for i in range(N):
            pointers[i] = values.pointer(i)

        pointers.pointer()[0] = values.pointer()
        pointers.pointer().load()[0] = 10
        pointers.pointer()[1][0] = 11
        vector = pointers.pointer().load(count=N).with_item(2, values.pointer(2))
        vector[1:3][1][0] = 12
        copied.pointer().store(vector)
        copied.pointer().load(count=N)[3][0] = 13

        for i in range(N):
            out[i] = values[i]

    N = 4
    out = torch.zeros(N, dtype=torch.int8, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (N, out))
    assert out.cpu().tolist() == [10, 11, 12, 13]


def test_vector_of_global_pointers():
    @cl.kernel
    def kernel(N: cl.Constant, out: cl.Array):
        pointers = cl.shared_array(N, cl.pointer_dtype(cl.int8))
        for i in range(N):
            pointers[i] = out.pointer(i)

        vector = pointers.pointer().load(count=N)
        for i in range(N):
            vector[i][0] = i

    N = 4
    out = torch.zeros(N, dtype=torch.int8, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (N, out))
    assert out.cpu().tolist() == list(range(N))


def test_pointer_array_rejects_wrong_pointee():
    def kernel():
        pointers = cl.shared_array(1, cl.pointer_dtype(cl.int8))
        pointers[0] = pointers.pointer()

    compile_kernel(
        kernel,
        raises=pytest.raises(TypeCheckingError, match="cannot implicitly cast"),
    )


def test_pointer_array_rejects_wrong_memory_space():
    def kernel(out):
        pointers = cl.shared_array(1, cl.pointer_dtype(cl.int8, 'SHARED'))
        pointers[0] = out.pointer()

    compile_kernel(
        kernel,
        KernelSignature([make_symbolic_tensor(1, cl.int8)]),
        raises=pytest.raises(TypeCheckingError, match="cannot implicitly cast"),
    )


def vector_as_i64(vector):
    return vector.astype(cl.int64)


def vector_reduce(vector):
    return vector.reduce(cl.VectorReduction.add)


@pytest.mark.parametrize(
    "operation",
    (
        vector_as_i64,
        operator.add,
        vector_reduce,
    ),
)
def test_vector_of_pointer_rejects_unsupported_operation(operation):
    def kernel():
        pointers = cl.shared_array(2, cl.pointer_dtype(cl.int8))
        operation(pointers.pointer().load(count=2))

    compile_kernel(kernel, raises=pytest.raises(TypeCheckingError))


def test_vector_of_pointer_bitcast():
    @cl.kernel
    def kernel(N: cl.Constant, out: cl.Array):
        pointers = cl.shared_array(N, cl.pointer_dtype(cl.int8))
        for i in range(N):
            pointers[i] = out.pointer(i)

        vector = pointers.pointer().load(count=N)
        integers = vector.bitcast(cl.uint64)
        generic = integers.bitcast(cl.pointer_dtype(cl.int8))
        global_ = generic.bitcast(cl.pointer_dtype(cl.int8, "GLOBAL"))
        for i in range(N):
            global_[i][0] = i

    N = 4
    out = torch.zeros(N, dtype=torch.int8, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (N, out))
    assert out.cpu().tolist() == list(range(N))


def test_vector_of_pointer_reinterpret():
    @cl.kernel
    def kernel(out):
        pointers = cl.shared_array(2, cl.pointer_dtype(cl.int8, "SHARED"))
        values = cl.shared_array(2, cl.int8)
        for i in range(2):
            pointers[i] = values.pointer(i)

        vector = pointers.pointer().load(count=2)
        floats = vector.bitcast(cl.float32)
        vector = floats.bitcast(cl.pointer_dtype(cl.int8, "SHARED"))
        words = vector.reinterpret_as_vector(cl.uint16, 4)
        roundtrip = words.reinterpret_as_vector(cl.pointer_dtype(cl.int8, "SHARED"), 2)
        roundtrip[0][0] = 10
        roundtrip[1][0] = 11
        out[0] = values[0]
        out[1] = values[1]

    out = torch.zeros(2, dtype=torch.int8, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [10, 11]
