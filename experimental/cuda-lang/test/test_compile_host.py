# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from dataclasses import dataclass
import struct
from typing import Any, Annotated

import cuda.lang as cl
import pytest
import torch

from cuda.tile import ScalarInt64
from cuda.tile._stub import ScalarAnnotation
from cuda.tile._annotated_function import get_annotated_function
from cuda.tile._cext import (
    CallingConvention,
    TileDispatcher,
    cconv_v3_enabled,
    get_parameter_constraints_from_pyargs,
)
from cuda.lang import _compile_host
from cuda.lang._exception import TypeCheckingError
from cuda.lang.compilation import (
    ArrayConstraint,
    ConstantConstraint,
    KernelSignature,
    ListConstraint,
    ScalarConstraint,
)
from cuda.tile.compilation import TupleConstraint
from cuda.tile.compilation._signature import PointerConstraint

ScalarInt32 = Annotated[int, ScalarAnnotation(dtype=cl.int32)]


def _signature_from_pyargs(function, *pyargs):
    annotated = get_annotated_function(function)
    dispatcher = TileDispatcher(annotated.parameter_annotations)
    constraints = get_parameter_constraints_from_pyargs(
        dispatcher,
        tuple(pyargs),
        CallingConvention.cutile_python_v1(),
    )
    return KernelSignature(constraints)


def _compile_host_function(function, *pyargs, **kwargs):
    return _compile_host._compile(
        function,
        _signature_from_pyargs(function, *pyargs),
        **kwargs,
    )


def _array_constraint(dtype):
    return ArrayConstraint(
        dtype=dtype,
        ndim=1,
        index_dtype=cl.int32,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(1,),
    )


def _scalar_ctype(dtype):
    ctype_map = {
        cl.bool_: ctypes.c_bool,
        cl.int8: ctypes.c_int8,
        cl.uint8: ctypes.c_uint8,
        cl.int16: ctypes.c_int16,
        cl.uint16: ctypes.c_uint16,
        cl.int32: ctypes.c_int32,
        cl.uint32: ctypes.c_uint32,
        cl.int64: ctypes.c_int64,
        cl.uint64: ctypes.c_uint64,
        cl.float32: ctypes.c_float,
        cl.float64: ctypes.c_double,
    }
    return ctype_map[dtype]


def _append_arg_ctypes(constraint, ctypes_out):
    if isinstance(constraint, ScalarConstraint):
        ctypes_out.append(_scalar_ctype(constraint.dtype))
    elif isinstance(constraint, PointerConstraint):
        ctypes_out.append(ctypes.c_void_p)
    elif isinstance(constraint, ArrayConstraint):
        ctypes_out.append(ctypes.c_void_p)
        ctypes_out.extend(
            _scalar_ctype(constraint.index_dtype) for _ in range(2 * constraint.ndim)
        )
    elif isinstance(constraint, ConstantConstraint):
        return
    elif isinstance(constraint, TupleConstraint):
        for item in constraint.items:
            _append_arg_ctypes(item, ctypes_out)
    elif hasattr(constraint, "fields"):
        for field in constraint.fields:
            _append_arg_ctypes(field, ctypes_out)
    elif isinstance(constraint, ListConstraint):
        raise TypeError("list arguments are not supported by compiled-host tests")
    else:
        raise TypeError(f"unsupported parameter constraint {constraint}")


def _infer_arg_ctypes(compilation):
    ctypes_out = []
    for constraint in compilation._compilation.signature.parameters:
        _append_arg_ctypes(constraint, ctypes_out)
    return tuple(ctypes_out)


def _call(compilation, args, arg_ctypes=None):
    if arg_ctypes is None:
        arg_ctypes = _infer_arg_ctypes(compilation)
    c_args = tuple(t(x) for x, t in zip(args, arg_ctypes, strict=True))
    addresses = tuple(ctypes.addressof(x) for x in c_args)
    compilation._invoke(addresses)
    ctypes.CDLL(None).fflush(None)
    return 0


def test_scalar_add(capfd):
    @cl.function(host=True, simt=False)
    def add(a, b):
        x = a
        for i in range(b):
            x = x + 1
        print(x)

    compilation = _compile_host_function(add, 15, 27)
    status = _call(compilation, (15, 27))
    assert status == 0
    assert capfd.readouterr().out == "42\n"


def test_scalar_add_constant(capfd):
    @cl.function(host=True, simt=False)
    def add_const(a, x: cl.Constant[int]):
        print(a + x)

    compilation = _compile_host_function(add_const, 0, 42)
    status = _call(compilation, (0,))
    assert status == 0
    assert capfd.readouterr().out == "42\n"


@pytest.mark.parametrize(
    "array_dtype, backing_dtype",
    (
        (cl.int32, torch.int32),
        (cl.float8_e4m3fn, torch.uint8),
    ),
    ids=("int32", "float8_e4m3fn"),
)
def test_kernel_launch(array_dtype, backing_dtype):
    @cl.kernel
    def kernel(output):
        pass

    @cl.function(host=True, simt=False)
    def launch_kernel(stream: ScalarInt64, output):
        cl.launch(
            stream,
            (1,),
            (1,),
            kernel,
            (output,),
            cooperative=True,
        )

    output = torch.zeros(16, dtype=backing_dtype, device="cuda:0")
    stream = torch.cuda.current_stream().cuda_stream
    program = _compile_host._compile(
        launch_kernel,
        KernelSignature((ScalarConstraint(cl.int64), _array_constraint(array_dtype))),
    )
    _call(
        program,
        (stream, output.data_ptr(), output.numel(), 1,),
    )
    torch.cuda.synchronize()


def test_kernel_launch_passes_runtime_scalar_to_const_param():
    @cl.kernel
    def kernel(value: cl.Constant[int], output):
        output[0] = value

    @cl.function(host=True, simt=False)
    def host(value, output):
        cl.launch(None, (1,), (1,), kernel, (value, output))

    compile_count = 0
    original_compile = kernel._compile

    def counted_compile(*args, **kwargs):
        nonlocal compile_count
        compile_count += 1
        return original_compile(*args, **kwargs)

    kernel._compile = counted_compile
    output = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    program = _compile_host_function(host, 7, output)
    assert compile_count == 0

    arguments = (7, output.data_ptr(), 1, 1,)
    _call(program, arguments)
    _call(program, arguments)
    torch.cuda.synchronize()
    assert output.item() == 7
    assert compile_count == 1

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (7, output))
    torch.cuda.synchronize()
    assert compile_count == 1

    _call(program, (9, output.data_ptr(), 1, 1,))
    torch.cuda.synchronize()
    assert output.item() == 9
    assert compile_count == 2


def test_kernel_launch_passes_const_scalar_to_runtime_param():
    @cl.kernel
    def kernel(value, output):
        output[0] = value

    @cl.function(host=True, simt=False)
    def host(value: cl.Constant[int], output):
        cl.launch(None, (1,), (1,), kernel, (value, output))

    compile_count = 0
    original_compile = kernel._compile

    def counted_compile(*args, **kwargs):
        nonlocal compile_count
        compile_count += 1
        return original_compile(*args, **kwargs)

    kernel._compile = counted_compile
    output = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    program = _compile_host_function(host, 42, output)
    assert compile_count == 0

    _call(
        program,
        (output.data_ptr(), 1, 1,),
    )
    _call(
        program,
        (output.data_ptr(), 1, 1,),
    )

    torch.cuda.synchronize()
    assert output.item() == 42
    assert compile_count == 1


@pytest.mark.parametrize(
    "grid_x, message",
    (
        pytest.param(-1, "Grid\\[0\\] value must be non-negative", id="negative"),
        pytest.param(1 << 24, "Grid\\[0\\] exceeds 24-bit limit", id="over-24-bit"),
        pytest.param((1 << 32) + 1, "Grid\\[0\\] value too big", id="over-uint32"),
    ),
)
def test_kernel_launch_rejects_invalid_runtime_grid_dim(grid_x, message):
    @cl.kernel
    def kernel(output):
        output[0] = 1

    @cl.function(host=True, simt=False)
    def host(grid_x: ScalarInt64, output):
        cl.launch(None, (grid_x,), (1,), kernel, (output,))

    output = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    program = _compile_host_function(host, 1, output)
    with pytest.raises(ValueError, match=message):
        _call(program, (grid_x, output.data_ptr(), 1, 1,))


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _uint32_bits(value: int) -> int:
    return value & ((1 << 32) - 1)


def _uint64_bits(value: int) -> int:
    return value & ((1 << 64) - 1)


@pytest.mark.parametrize(
    "source_dtype, source_value, kernel_dtype, expected_bits",
    (
        pytest.param(
            cl.float32, -13.5, cl.float32, _float32_bits(-13.5), id="float32"
        ),
        pytest.param(
            cl.int32, -123456789, cl.int32, _uint32_bits(-123456789), id="int32"
        ),
        pytest.param(
            cl.uint32, 0x7ABCDEFA, cl.int32, 0x7ABCDEFA, id="uint32"
        ),
        pytest.param(
            cl.int64, -0x123456789ABCDEF, cl.int64, _uint64_bits(-0x123456789ABCDEF), id="int64",
        ),
        pytest.param(
            cl.uint64, 0x7FEDCBA987654321, cl.int64, 0x7FEDCBA987654321, id="uint64",
        ),
    ),
)
def test_kernel_launch_preserves_scalar_bits(
    source_dtype, source_value, kernel_dtype, expected_bits
):
    if kernel_dtype is cl.int64:
        @cl.kernel
        def kernel(value: ScalarInt64, output):
            cl.static_assert(cl.dtype_of(value) == kernel_dtype)
            output[0] = cl.bitcast(value, cl.uint64)
    else:
        @cl.kernel
        def kernel(value, output):
            cl.static_assert(cl.dtype_of(value) == kernel_dtype)
            output[0] = cl.uint64(cl.bitcast(value, cl.uint32))

    @cl.function(host=True, simt=False)
    def host(output):
        cl.launch(None, (1,), (1,), kernel, (source_dtype(source_value), output))

    output = torch.zeros(1, dtype=torch.uint64, device="cuda:0")
    program = _compile_host_function(host, output)
    _call(
        program,
        (output.data_ptr(), 1, 1,),
    )
    torch.cuda.synchronize()
    assert output.item() == expected_bits


@pytest.mark.parametrize(
    "source_dtype, source_value, scalar_int64, message",
    (
        pytest.param(
            cl.uint32, 1 << 31, False, "Python int too large to convert to C int32_t",
            id="uint32-to-i32",
        ),
        pytest.param(
            cl.int64, 1 << 31, False, "Python int too large to convert to C int32_t",
            id="int64-to-i32",
        ),
        pytest.param(
            cl.uint64, 1 << 31, False, "Python int too large to convert to C int32_t",
            id="uint64-to-i32",
        ),
        pytest.param(
            cl.uint64, 1 << 63, True, "Python int too large to convert to C int64_t",
            id="uint64-to-i64",
        ),
    ),
)
def test_kernel_launch_scalar_overflow(
    source_dtype, source_value, scalar_int64, message
):
    if scalar_int64:
        @cl.kernel
        def kernel(value: ScalarInt64, output):
            cl.static_assert(cl.dtype_of(value) == cl.int64)
            output[0] = cl.bitcast(value, cl.uint64)
    else:
        @cl.kernel
        def kernel(value: ScalarInt32, output):
            cl.static_assert(cl.dtype_of(value) == cl.int32)
            output[0] = cl.uint64(cl.bitcast(value, cl.uint32))

    @cl.function(host=True, simt=False)
    def host(output):
        cl.launch(None, (1,), (1,), kernel, (source_dtype(source_value), output))

    output = torch.zeros(1, dtype=torch.uint64, device="cuda:0")
    program = _compile_host_function(host, output)
    with pytest.raises(OverflowError, match=message):
        _call(
            program,
            (output.data_ptr(), 1, 1,),
        )


def test_kernel_launch_tuple_argument():
    @cl.kernel
    def add_pair(values: tuple[int, int], output):
        output[0] = values[0] + values[1]

    @cl.function(host=True, simt=False)
    def host(a, b, output):
        cl.launch(None, (1,), (1,), add_pair, ((a, b), output))

    output = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    program = _compile_host_function(host, 15, 27, output)
    _call(
        program,
        (15, 27, output.data_ptr(), 1, 1,),
    )
    torch.cuda.synchronize()
    assert output.item() == 42


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_kernel_launch_dataclass_argument():
    @dataclass(frozen=True)
    class Pair:
        x: int
        y: int

    @cl.kernel
    def add_pair(values: Any, output):
        output[0] = values.x + values.y

    @cl.function(host=True, simt=False)
    def host(a, b, output):
        cl.launch(None, (1,), (1,), add_pair, (Pair(a, b), output))

    output = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    program = _compile_host_function(host, 15, 27, output)
    _call(
        program,
        (15, 27, output.data_ptr(), 1, 1,),
    )
    torch.cuda.synchronize()
    assert output.item() == 42


def test_kernel_launch_rejects_list_argument():
    @cl.kernel
    def kernel(arrays):
        pass

    @cl.function(host=True, simt=False)
    def host(arrays):
        cl.launch(None, (1,), (1,), kernel, (arrays,))

    with pytest.raises(
        TypeCheckingError,
        match="kernel launch with list argument is not supported in compiled host code",
    ):
        _compile_host_function(
            host,
            [torch.zeros(1, dtype=torch.int32, device="cuda:0")],
        )
