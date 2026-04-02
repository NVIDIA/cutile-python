# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from ctypes import (c_char_p, POINTER, c_void_p, c_int, c_uint64, pointer, CFUNCTYPE, c_uint,
                    c_int32, c_float, byref)
from io import BytesIO

import torch.cuda

import cuda.tile as ct
from cuda.tile._compile import get_sm_arch


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


def call_kernel_cutile_python_v1(cubin: bytes, kernel_name: str, runtime_pyargs):
    driver = CudaDriver()
    library = driver.cuLibraryLoadData(cubin)
    kernel = driver.cuLibraryGetKernel(library, kernel_name)

    args = []
    for x in runtime_pyargs:
        if isinstance(x, torch.Tensor):
            args.append(c_void_p(x.data_ptr()))
            for s in x.shape:
                args.append(c_int32(s))
            for s in x.stride():
                args.append(c_int32(s))
        elif isinstance(x, int):
            args.append(c_int32(x))
        elif isinstance(x, float):
            args.append(c_float(x))
        else:
            assert False

    stream = torch.cuda.current_stream()
    driver.cuLaunchKernel(kernel, (1, 1, 1), (1, 1, 1), 0, c_void_p(stream.cuda_stream), args)


def test_export_compat_cutile_python_v1():
    sig = ct.compilation.KernelSignature(
        parameters=[
            13,
            ct.compilation.ScalarConstraint(ct.int32),
            17.0,
            ct.compilation.ScalarConstraint(ct.float32),
            ct.compilation.ArrayConstraint(ct.int32, 2, stride_lower_bound_incl=0,
                                           alias_groups=(), may_alias_internally=False,
                                           stride_divisible_by=(4, 1),
                                           stride_constant=(None, 1)),
            ct.compilation.ArrayConstraint(ct.float32, 3, stride_lower_bound_incl=0,
                                           alias_groups=(), may_alias_internally=False),
        ],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v1(),
    )

    io = BytesIO()
    ct.compilation.export_kernel(kernel_1, [sig], gpu_code=get_sm_arch(), output_file=io,
                                 output_format="cubin")
    a1 = torch.zeros((32, 8), dtype=torch.int32, device="cuda")
    a2 = torch.zeros((8, 8, 8), dtype=torch.float32, device="cuda")
    call_kernel_cutile_python_v1(
            io.getvalue(),
            "kernel_1_Kt1_I13_Si32_F4031000000000000_Sf32_A2i32_1v4l0_2t1_A3f32_7l0",
            (5, 9.0, a1, a2))
    a1_cpu = a1.cpu()
    a2_cpu = a2.cpu()
    for i in range(5):
        assert a1_cpu[i*2+3, i] == 13 * 1000 + 5 * 10 + i
        assert a2_cpu[7 - i, i + 2, i] == 17.0 * 1000.0 + 9.0 * 10.0 + i
