# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest
import torch

import cuda.lang as cl
import cuda.lang._datatype as datatype
from cuda.lang._exception import TypeCheckingError
from cuda.lang.compilation import KernelSignature, ScalarConstraint
from .util import compile_kernel, make_symbolic_tensor


def test_packed_dtype():
    dtype = cl.float4_e2m1fn_x2
    assert dtype.name == "float4_e2m1fn_x2"
    assert dtype.bitwidth == 8
    assert not datatype.is_numeric(dtype)
    assert not datatype.is_arithmetic(dtype)
    assert datatype.to_torch_dtype(dtype) is torch.float4_e2m1fn_x2
    assert pickle.loads(pickle.dumps(dtype)) is dtype
    pointer = cl.pointer_dtype(dtype)
    assert cl.PointerInfo(pointer).pointee_dtype is dtype
    assert pickle.loads(pickle.dumps(pointer)) is pointer
    assert not hasattr(cl, "float4_e2m1fn")


def use_scalar(source, output, array):
    i = cl.thread_index(0) * 4
    cl.static_assert(source.dtype == cl.float4_e2m1fn_x2)
    for j in range(4):
        array[i + j] = source[i + j]
        output[i + j] = array[i + j]


def use_vector(source, output, array):
    i = cl.thread_index(0) * 4
    value = (source.pointer() + i).load(count=4)
    cl.static_assert(cl.dtype_of(value) == cl.float4_e2m1fn_x2)
    (array.pointer() + i).store(value)
    value = (array.pointer() + i).load(count=4)
    (output.pointer() + i).store(value)


def use_shared_scalar(source, output):
    dtype = cl.float4_e2m1fn_x2
    array = cl.shared_array(256, dtype)
    use_scalar(source, output, array)


def use_shared_vector(source, output):
    dtype = cl.float4_e2m1fn_x2
    array = cl.shared_array(256, dtype)
    use_vector(source, output, array)


def use_dynamic_shared_scalar(source, output, size):
    dtype = cl.float4_e2m1fn_x2
    array = cl.shared_array(size, dtype, dynamic=True)
    use_scalar(source, output, array)


def use_dynamic_shared_vector(source, output, size):
    dtype = cl.float4_e2m1fn_x2
    array = cl.shared_array(size, dtype, dynamic=True)
    use_vector(source, output, array)


def use_local_scalar(source, output):
    dtype = cl.float4_e2m1fn_x2
    with cl.local_array(256, dtype) as array:
        use_scalar(source, output, array)


def use_local_vector(source, output):
    dtype = cl.float4_e2m1fn_x2
    with cl.local_array(256, dtype) as array:
        use_vector(source, output, array)


@pytest.mark.parametrize(
    "kernel, extra_args",
    (
        (use_shared_scalar, ()),
        (use_shared_vector, ()),
        (use_dynamic_shared_scalar, (256,)),
        (use_dynamic_shared_vector, (256,)),
        (use_local_scalar, ()),
        (use_local_vector, ()),
    ),
)
def test_array_encoding(kernel, extra_args):
    source = torch.arange(256).to(torch.uint8).cuda(0)
    output = torch.full_like(source, 255)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (64,),
        cl.kernel(kernel),
        (
            source.view(torch.float4_e2m1fn_x2),
            output.view(torch.float4_e2m1fn_x2),
            *extra_args,
        ),
    )
    torch.testing.assert_close(output, source)


def test_strided_global_array():
    @cl.kernel
    def kernel(source, output):
        i = cl.thread_index(0)
        cl.static_assert(source.dtype == cl.float4_e2m1fn_x2)
        output[i] = source[i // 16, i % 16]

    storage = torch.arange(256).to(torch.uint8).cuda(0).reshape(16, 16)
    source = storage.view(torch.float4_e2m1fn_x2).T[::2]
    output = torch.empty(128, dtype=torch.uint8).cuda(0)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (128,),
        kernel,
        (source, output.view(torch.float4_e2m1fn_x2)),
    )
    torch.testing.assert_close(output, storage.T[::2].flatten())


class TestBitcastEncoding:
    def bitcast_scalar(source, output):
        i = cl.thread_index(0) * 4
        for j in range(4):
            pair = cl.bitcast(source[i + j], cl.float4_e2m1fn_x2)
            output[i + j] = cl.bitcast(pair, cl.uint8)

    def bitcast_vector(source, output):
        i = cl.thread_index(0) * 4
        value = (source.pointer() + i).load(count=4)
        pairs = cl.bitcast(value, cl.float4_e2m1fn_x2)
        pairs = pairs.with_item(0, pairs[0])
        (output.pointer() + i).store(cl.bitcast(pairs, cl.uint8))

    def bitcast_pointer(source, output):
        i = cl.thread_index(0) * 4
        array = cl.shared_array(256, cl.float4_e2m1fn_x2)
        for j in range(4):
            array.pointer()[i + j] = cl.bitcast(source[i + j], cl.float4_e2m1fn_x2)
            output[i + j] = cl.bitcast(array.pointer()[i + j], cl.uint8)

    def bitcast_vector_elements(source, output):
        i = cl.thread_index(0) * 4
        value = (source.pointer() + i).load(count=4)
        pairs = cl.bitcast(value, cl.float4_e2m1fn_x2)
        pairs = cl.Vector(*tuple(pairs[:2]), *tuple(pairs[2:]))
        (output.pointer() + i).store(cl.bitcast(pairs, cl.uint8))

    @pytest.mark.parametrize(
        "kernel", (bitcast_scalar, bitcast_vector, bitcast_pointer, bitcast_vector_elements)
    )
    def test_bitcast_encoding(self, kernel):
        source = torch.arange(256).to(torch.uint8).cuda(0)
        output = torch.full_like(source, 255)
        cl.launch(
            torch.cuda.current_stream(),
            (1,),
            (64,),
            cl.kernel(kernel),
            (source, output),
        )
        torch.testing.assert_close(output, source)


def test_dynamic_array_size():
    def kernel(size):
        array = cl.shared_array(size, cl.float4_e2m1fn_x2, dynamic=True)
        array[1] = array[0]

    result = cl.compile_simt(
        kernel,
        [KernelSignature((ScalarConstraint(dtype=cl.int32),))],
        gpu_name="sm_80",
        arch="compute_80",
    )
    program = result.dyn_smem_size_program
    assert program and program.op_attrs and program.opcodes
    assert program.opcodes == ["KernelArgI32"]
    assert program.op_attrs == [0]


class TestRejectNumericOperations:
    @cl.kernel
    def add_pairs(arr):
        arr[0] = arr[0] + arr[1]

    @cl.kernel
    def multiply_pairs(arr):
        arr[0] = arr[0] * arr[1]

    @cl.kernel
    def convert_pairs_to_float(arr):
        arr.pointer().load(count=2).astype(cl.float32)

    @cl.kernel
    def convert_float_to_pairs(_):
        cl.Vector(1.0, 2.0).astype(cl.float4_e2m1fn_x2)

    @cl.kernel
    def store_integer(arr):
        arr[0] = cl.uint8(1)

    @cl.kernel
    def store_integer_literal(arr):
        arr[0] = 1

    @cl.kernel
    def store_integer_through_pointer(arr):
        arr.pointer().store(cl.uint8(1))

    @cl.kernel
    def construct_pair_from_float(arr):
        arr[0] = cl.float4_e2m1fn_x2(1.0)

    @pytest.mark.parametrize(
        "kernel",
        (
            add_pairs,
            multiply_pairs,
            convert_pairs_to_float,
            convert_float_to_pairs,
            store_integer,
            store_integer_literal,
            store_integer_through_pointer,
            construct_pair_from_float,
        ),
    )
    def test_reject_numeric_operations(self, kernel):
        signature = KernelSignature([make_symbolic_tensor(2, cl.float4_e2m1fn_x2)])
        compile_kernel(kernel, signature, raises=pytest.raises(TypeCheckingError))


@pytest.mark.parametrize("dtype", (cl.bool_, cl.int8, cl.uint8, cl.int32, cl.float32))
@pytest.mark.parametrize("vector", (False, True))
def test_reject_cast_from_pairs(dtype, vector):
    @cl.kernel
    def kernel(array):
        if vector:
            array.pointer().load(count=2).astype(dtype)
        else:
            dtype(array[0])

    signature = KernelSignature([make_symbolic_tensor(2, cl.float4_e2m1fn_x2)])
    compile_kernel(kernel, signature, raises=pytest.raises(TypeCheckingError))


@pytest.mark.parametrize("dtype", (cl.bool_, cl.int8, cl.uint8, cl.int32, cl.float32))
@pytest.mark.parametrize("vector", (False, True))
def test_reject_cast_to_pairs(dtype, vector):
    @cl.kernel
    def kernel(array):
        if vector:
            array.pointer().load(count=2).astype(cl.float4_e2m1fn_x2)
        else:
            cl.float4_e2m1fn_x2(array[0])

    signature = KernelSignature([make_symbolic_tensor(2, dtype)])
    compile_kernel(kernel, signature, raises=pytest.raises(TypeCheckingError))


def test_reject_string_constructor():
    @cl.kernel
    def kernel():
        cl.float4_e2m1fn_x2("1.0")

    compile_kernel(kernel, raises=pytest.raises(TypeCheckingError, match="Cannot call"))
