# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.testing import assert_close

from cuda.lang._compile import get_compute_capability
from cuda.lang._compilers import get_nvvm_and_libdevice, PtxCompiler
from cuda.lang._llvm_bitcode import BitcodeBuilder, DATALAYOUT_PTX, CallingConvention, Binop
from cuda.tile._cext import TileDispatcher

from cuda.tile._annotated_function import LeafAnnotationNode
import cuda.lang as cl


def _make_a_plus_b_bitcode() -> bytes:
    builder = _make_builder()
    tt = builder.type_table

    with builder.function("llvm.nvvm.read.ptx.sreg.tid.x",
                          tt.function(tt.I32, ())) as f:
        tid_x = f.value

    with builder.function("__nv_fabsf",
                          tt.function(tt.F32, (tt.F32,))) as f:
        fabsf = f.value

    with builder.function("aplusb",
                          tt.function(tt.VOID,
                                      (tt.P0, tt.I32, tt.I32,
                                       tt.P0, tt.I32, tt.I32,
                                       tt.P0, tt.I32, tt.I32)),
                          calling_convention=CallingConvention.PTX_Kernel) as f:
        aptr, asize, astride, bptr, bsize, bstride, cptr, csize, cstride = f.parameters
        tid = builder.call(tt.function(tt.I32, ()), tid_x, ())
        ap = builder.get_element_ptr(tt.F32, aptr, tid)
        a = builder.load(tt.F32, ap, alignment=4)
        bp = builder.get_element_ptr(tt.F32, bptr, tid)
        b = builder.load(tt.F32, bp, alignment=4)
        cp = builder.get_element_ptr(tt.F32, cptr, tid)
        c = builder.binop(Binop.ADD, a, b)
        res = builder.call(tt.function(tt.F32, (tt.F32,)), fabsf, (c,))
        builder.store(cp, res, alignment=4)
        builder.ret()
    return builder.build()


def test_a_plus_b():
    bitcode = _make_a_plus_b_bitcode()
    cubin = _bitcode_to_cubin(bitcode)

    kernel = _HackKernel(cubin, "aplusb", 3)
    a = torch.ones(32, dtype=torch.float32, device="cuda:0")
    b = torch.arange(-16, 16, dtype=torch.float32, device="cuda:0")
    ref = (a + b).abs()
    c = torch.zeros_like(a)
    cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (a, b, c))
    assert_close(c, ref, rtol=0, atol=0)


def _make_inline_ptx_bitcode() -> bytes:
    builder = _make_builder()
    tt = builder.type_table

    with builder.function("aplusminusb",
                          tt.function(tt.VOID,
                                      (tt.P0, tt.I32, tt.I32,
                                       tt.P0, tt.I32, tt.I32,
                                       tt.P0, tt.I32, tt.I32)),
                          calling_convention=CallingConvention.PTX_Kernel) as f:
        aptr, asize, astride, bptr, bsize, bstride, cptr, csize, cstride = f.parameters
        zero = builder.constants.integer_constant(0, tt.I32)
        one = builder.constants.integer_constant(1, tt.I32)
        ap = builder.get_element_ptr(tt.F32, aptr, zero)
        a = builder.load(tt.F32, ap, alignment=4)
        bp = builder.get_element_ptr(tt.F32, bptr, zero)
        b = builder.load(tt.F32, bp, alignment=4)
        cp0 = builder.get_element_ptr(tt.F32, cptr, zero)
        cp1 = builder.get_element_ptr(tt.F32, cptr, one)

        ptx_functy = tt.function(tt.struct_anonymous([tt.F32, tt.F32]), [tt.F32, tt.F32])
        ptx_func = builder.constants.inline_asm(
            ptx_functy,
            "sub.f32 $3, $0, $2;\nadd.f32 $0, $0, $2;",
            "=f,0,f,=f",
            side_effects=False
        )
        r = builder.call(ptx_functy, ptx_func, [a, b])
        radd = builder.extract_value(r, 0)
        rsub = builder.extract_value(r, 1)
        builder.store(cp0, radd, alignment=4)
        builder.store(cp1, rsub, alignment=4)
        builder.ret()
    return builder.build()


def test_inline_ptx():
    bitcode = _make_inline_ptx_bitcode()
    cubin = _bitcode_to_cubin(bitcode)

    kernel = _HackKernel(cubin, "aplusminusb", 3)
    a = torch.tensor([5], dtype=torch.float32, device="cuda")
    b = torch.tensor([7], dtype=torch.float32, device="cuda")
    c = torch.zeros(2, dtype=torch.float32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b, c))
    assert c.tolist() == [12.0, -2.0]


def _make_builder() -> BitcodeBuilder:
    builder = BitcodeBuilder(target_triple="nvptx64-nvidia-cuda", data_layout=DATALAYOUT_PTX)
    builder.append_nvvm_version_metadata(2, 0)
    return builder


def _bitcode_to_cubin(bitcode: bytes) -> bytes:
    cc = get_compute_capability()

    # Compile bitcode to PTX using NVVM
    nvvm, libdevice = get_nvvm_and_libdevice()
    program = nvvm.create_program()
    program.add_module(bitcode, "main")
    program.add_module(libdevice, "libdevice")
    ptx = program.compile(["-arch=" + cc.arch])
    ptx_compiler: PtxCompiler = PtxCompiler.get()
    cubin = ptx_compiler.compile(ptx, cc.gpu_name)
    return cubin


class _HackKernel(TileDispatcher):
    def __init__(self, cubin: bytes, func_name: str, arity: int):
        self._cubin = cubin
        self._func_name = func_name
        annotations = tuple(LeafAnnotationNode(constant=False) for _ in range(arity))
        super().__init__(annotations)

    def _compile(self, signature, ctx, compute_capability):
        return self._cubin, self._func_name, None, ()
