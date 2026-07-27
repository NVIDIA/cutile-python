# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import dataclasses
import re
from ctypes import (c_char_p, POINTER, c_void_p, c_int, c_uint64, pointer, CFUNCTYPE, c_uint,
                    c_int32, c_float, byref)
from io import BytesIO

import pytest
import torch.cuda

import cuda.tile as ct
from cuda.tile._cext import cconv_v3_enabled
from cuda.tile._compile import get_sm_arch
from cuda.tile._exception import TileUnsupportedFeatureError
from util import get_bytecode


class CudaDriver:
    def __init__(self):
        from cuda.tile._load_libcuda import _cuGetProcAddress_v2
        _cuGetProcAddress_v2.argtypes = [c_char_p, POINTER(c_void_p), c_int, c_uint64, c_void_p]
        _cuGetProcAddress_v2.restype = c_int

        def get_proc(name: bytes, version: int, ty):
            func_ptr_v = c_void_p()
            res = _cuGetProcAddress_v2(name, byref(func_ptr_v), version, 0, None)
            assert res == 0
            assert func_ptr_v.value is not None
            return ctypes.cast(func_ptr_v, ty)

        functy_cuLibraryLoadData = CFUNCTYPE(
            c_int,
            POINTER(c_void_p), c_void_p,
            c_void_p, c_void_p, c_uint,
            c_void_p, c_void_p, c_uint)
        self._cuLibraryLoadData = get_proc(b"cuLibraryLoadData", 12000, functy_cuLibraryLoadData)

        functy_cuLibraryGetKernel = CFUNCTYPE(c_int, POINTER(c_void_p), c_void_p, c_char_p)
        self._cuLibraryGetKernel = get_proc(b"cuLibraryGetKernel", 12000, functy_cuLibraryGetKernel)

        functy_cuLaunchKernel = CFUNCTYPE(c_int, c_void_p,
                                          c_uint, c_uint, c_uint,
                                          c_uint, c_uint, c_uint,
                                          c_uint, c_void_p, POINTER(c_void_p), POINTER(c_void_p))
        self._cuLaunchKernel = get_proc(b"cuLaunchKernel", 7000, functy_cuLaunchKernel)

    def cuLibraryLoadData(self, code: bytes):
        library = c_void_p()
        res = self._cuLibraryLoadData(byref(library), code, None, None, 0, None, None, 0)
        assert res == 0
        return library

    def cuLibraryGetKernel(self, library, name: str):
        kernel = c_void_p()
        res = self._cuLibraryGetKernel(byref(kernel), library, name.encode())
        assert res == 0
        return kernel

    def cuLaunchKernel(self, f, grid: tuple[int, int, int], block: tuple[int, int, int],
                       shared_mem: int, stream: c_void_p, args: list):
        args_arr = (c_void_p * len(args))()
        for i, x in enumerate(args):
            args_arr[i] = ctypes.cast(pointer(x), c_void_p)
        res = self._cuLaunchKernel(f, *grid, *block, shared_mem, stream, args_arr, None)
        assert res == 0


@ct.kernel
def kernel_1(c1: ct.Constant, s1, c2: ct.Constant, s2,
             a1, a2):
    for i in range(5):
        ct.scatter(a1, (i*2+3, i), c1 * 1000 + s1 * 10 + i)
        ct.scatter(a2, (7 - i, i + 2, i), c2 * 1000.0 + s2 * 10 + i)


def _build_kernel_args(runtime_pyargs) -> list:
    args = []
    for x in runtime_pyargs:
        if isinstance(x, torch.Tensor):
            args.append(c_void_p(x.data_ptr()))
            for s in x.shape:
                args.append(c_int32(s))
            for s in x.stride():
                args.append(c_int32(s))
        elif isinstance(x, tuple):
            for elem in x:
                if isinstance(elem, int):
                    args.append(c_int32(elem))
                elif isinstance(elem, float):
                    args.append(c_float(elem))
                else:
                    assert False, f"Unsupported tuple element type: {type(elem)}"
        elif dataclasses.is_dataclass(x) and not isinstance(x, type):
            args.extend(_build_kernel_args(
                (getattr(x, field.name) for field in dataclasses.fields(x))))
        elif isinstance(x, int):
            args.append(c_int32(x))
        elif isinstance(x, float):
            args.append(c_float(x))
        else:
            assert False
    return args


def _call_kernel(cubin: bytes, kernel_name: str, runtime_pyargs):
    driver = CudaDriver()
    library = driver.cuLibraryLoadData(cubin)
    kernel = driver.cuLibraryGetKernel(library, kernel_name)
    stream = torch.cuda.current_stream()
    driver.cuLaunchKernel(kernel, (1, 1, 1), (1, 1, 1), 0,
                          c_void_p(stream.cuda_stream),
                          _build_kernel_args(runtime_pyargs))


def test_export_compat_cutile_python_v1():
    sig = ct.compilation.KernelSignature(
        parameters=[
            13,
            ct.compilation.ScalarConstraint(ct.int32),
            17.0,
            ct.compilation.ScalarConstraint(ct.float32),
            ct.compilation.ArrayConstraint(ct.int32, 2, index_dtype=ct.int32,
                                           stride_lower_bound_incl=0,
                                           alias_groups=(), may_alias_internally=False,
                                           stride_divisible_by=(4, 1),
                                           stride_constant=(None, 1)),
            ct.compilation.ArrayConstraint(ct.float32, 3, index_dtype=ct.int32,
                                           stride_lower_bound_incl=0,
                                           alias_groups=(), may_alias_internally=False),
        ],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v1(),
    )

    io = BytesIO()
    ct.compilation.export_kernel(kernel_1, [sig], gpu_code=get_sm_arch(), output_file=io,
                                 output_format="cubin")
    a1 = torch.zeros((32, 8), dtype=torch.int32, device="cuda")
    a2 = torch.zeros((8, 8, 8), dtype=torch.float32, device="cuda")
    _call_kernel(io.getvalue(),
                 "kernel_1_Kt1_I13_Si32_F4031000000000000_Sf32_A2i32_1v4l0_2t1_A3f32_7l0",
                 (5, 9.0, a1, a2))
    a1_cpu = a1.cpu()
    a2_cpu = a2.cpu()
    for i in range(5):
        assert a1_cpu[i*2+3, i] == 13 * 1000 + 5 * 10 + i
        assert a2_cpu[7 - i, i + 2, i] == 17.0 * 1000.0 + 9.0 * 10.0 + i


@ct.kernel
def kernel_static_shape(a, out):
    t = ct.load(a, (0,), (8,))
    ct.store(out, (0,), t + 1)


def test_export_compat_static_shape():
    sig = ct.compilation.KernelSignature(
        parameters=[
            ct.compilation.ArrayConstraint(ct.float32, 1, index_dtype=ct.int32,
                                           shape_constant=(8,),
                                           stride_lower_bound_incl=0,
                                           alias_groups=(), may_alias_internally=False),
            ct.compilation.ArrayConstraint(ct.float32, 1, index_dtype=ct.int32,
                                           shape_constant=(8,),
                                           stride_constant=(1,),
                                           stride_lower_bound_incl=0,
                                           alias_groups=(), may_alias_internally=False),
        ],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

    io = BytesIO()
    ct.compilation.export_kernel(kernel_static_shape, [sig], gpu_code=get_sm_arch(),
                                 output_file=io, output_format="cubin")
    a = torch.zeros(8, dtype=torch.float32, device="cuda")
    out = torch.zeros(8, dtype=torch.float32, device="cuda")
    # shape_constant=(8,): shape is still a runtime CUDA parameter — pass it as usual.
    # stride_lower_bound_incl for out is dropped (stride_constant=(1,) makes it redundant).
    _call_kernel(io.getvalue(), "kernel_static_shape_Kt2_A1f32_1s8l0_A1f32_1s8t1", (a, out))
    assert torch.all(out == 1.0).item()


@ct.kernel
def kernel_2(pair, addend: ct.Constant[tuple], out):
    ct.scatter(out, (), pair[0] + pair[1] + addend[0])


def test_export_compat_cutile_python_v2():
    sig = ct.compilation.KernelSignature(
        parameters=[
            ct.compilation.TupleConstraint(
                [
                    ct.compilation.ScalarConstraint(ct.int32),
                    ct.compilation.ScalarConstraint(ct.int32),
                ]),
            (10,),
            ct.compilation.ArrayConstraint(ct.int32, 0, index_dtype=ct.int32,
                                           stride_lower_bound_incl=0,
                                           alias_groups=(), may_alias_internally=False),
        ],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

    io = BytesIO()
    ct.compilation.export_kernel(kernel_2, [sig], gpu_code=get_sm_arch(), output_file=io,
                                 output_format="cubin")
    out = torch.zeros((), dtype=torch.int32, device="cuda")
    _call_kernel(io.getvalue(), "kernel_2_Kt2_T2Si32Si32_T1I10_A0i32", ((3, 7), out))
    assert out.item() == 20


@dataclasses.dataclass(frozen=True)
class Kernel3Args:
    mul: int
    add: float


@ct.kernel
def kernel_3(args, out):
    ct.scatter(out, (), args.mul * 10 + args.add)


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_export_compat_cutile_python_v3_dataclass():
    sig = ct.compilation.KernelSignature(
        parameters=[
            ct.compilation._signature.DataclassConstraint.create(
                Kernel3Args,
                mul=ct.compilation.ScalarConstraint(ct.int32),
                add=ct.compilation.ScalarConstraint(ct.float32)),
            ct.compilation.ArrayConstraint(ct.float32, 0, index_dtype=ct.int32,
                                           stride_lower_bound_incl=0,
                                           alias_groups=(), may_alias_internally=False),
        ],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v3(),
    )

    io = BytesIO()
    ct.compilation.export_kernel(kernel_3, [sig], gpu_code=get_sm_arch(), output_file=io,
                                 output_format="cubin")
    out = torch.zeros((), dtype=torch.float32, device="cuda")
    _call_kernel(io.getvalue(),
                 "kernel_3_Kt3_Dtest__export__compat_e_kernel3Args_e2Si32Sf32_A0f32",
                 (Kernel3Args(3, 7.5), out))
    assert out.item() == 37.5


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_constraint_shorthand():
    cconv = ct.compilation.CallingConvention.cutile_python_v3()
    signature = ct.compilation.KernelSignature(
        [Kernel3Args(3, 7.5)],
        cconv)

    assert signature.parameters == (
        ct.compilation._signature.DataclassConstraint(
            Kernel3Args,
            [ct.compilation.ConstantConstraint(3),
             ct.compilation.ConstantConstraint(7.5)]),
    )


def test_static_shape_with_v1_raises():
    cconv = ct.compilation.CallingConvention.cutile_python_v1()
    expected_message = re.escape("Static array shapes are not supported by calling convention"
                                 " cutile_python_v1; version >= 2 is required")
    with pytest.raises(ValueError, match=expected_message):
        ct.compilation.KernelSignature(
            [ct.compilation.ArrayConstraint(
                ct.float32, 1, index_dtype=ct.int32, stride_lower_bound_incl=0,
                alias_groups=(), may_alias_internally=False,
                shape_constant=(8,))],
            cconv)


def test_tuple_with_v1_raises():
    cconv = ct.compilation.CallingConvention.cutile_python_v1()
    expected_message = re.escape("Tuple parameters are not supported by calling convention"
                                 " cutile_python_v1; version >= 2 is required")
    with pytest.raises(ValueError, match=expected_message):
        ct.compilation.KernelSignature([(ct.compilation.ScalarConstraint(ct.int32),)], cconv)


@pytest.mark.parametrize("cconv,version", [
    (ct.compilation.CallingConvention.cutile_python_v1(), "cutile_python_v1"),
    (ct.compilation.CallingConvention.cutile_python_v2(), "cutile_python_v2"),
])
@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_with_old_cconv_raises(cconv, version):
    expected_message = re.escape(f"Dataclass parameters are not supported by calling convention"
                                 f" {version}; version >= 3 is required")
    with pytest.raises(ValueError, match=expected_message):
        ct.compilation.KernelSignature(
            [ct.compilation._signature.DataclassConstraint(
                Kernel3Args,
                [ct.compilation.ScalarConstraint(ct.int32),
                 ct.compilation.ScalarConstraint(ct.float32)])],
            cconv)


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_with_invalid_field_constraint():
    sig = ct.compilation.KernelSignature(
        [
            ct.compilation._signature.DataclassConstraint(
                Kernel3Args,
                [ct.compilation.ScalarConstraint(ct.int32), 123]),
            ct.compilation.ScalarConstraint(ct.int32),
        ],
        ct.compilation.CallingConvention.cutile_python_v3())
    io = BytesIO()
    expected_msg = re.escape("Invalid field 'add' of kernel parameter 'args': ConstantConstraint"
                             " is only valid for parameters annotated as Constant.")
    with pytest.raises(TypeError, match=expected_msg):
        ct.compilation.export_kernel(kernel_3, [sig], gpu_code=get_sm_arch(), output_file=io,
                                     output_format="cubin")


@ct.kernel
def add_one(x, y):
    tx = ct.load(x, 0, shape=8)
    ct.store(y, 0, tile=tx + 1)


def test_export_bytecode_without_gpu_code():
    x = torch.zeros(8, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    bytecode = get_bytecode(add_one, (x, y), lambda: None, bytecode_version="13.3")
    # Confirm bytecode generation by checking the headers
    assert bytecode.startswith(b"\x7fTileIR\x00")
    assert (bytecode[8], bytecode[9]) == (13, 3)


def test_export_bytecode_without_gpu_code_requires_13_3():
    x = torch.zeros(8, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    with pytest.raises(TileUnsupportedFeatureError, match="13.3 or later"):
        get_bytecode(add_one, (x, y), lambda: None, bytecode_version="13.1")


def test_export_cubin_requires_gpu_code():
    x = torch.zeros(8, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    sig = ct.compilation.KernelSignature.from_kernel_args(
        add_one, (x, y), ct.compilation.CallingConvention.cutile_python_v1())
    with pytest.raises(ValueError, match="gpu_code is required"):
        ct.compilation.export_kernel(add_one, [sig], output_file=BytesIO(),
                                     output_format="cubin")
