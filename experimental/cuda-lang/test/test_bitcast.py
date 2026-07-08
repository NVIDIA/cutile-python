# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib

import cuda.lang as cl
import pytest
import torch

from cuda.lang._exception import CompilerExecutionError, TypeCheckingError
from cuda.lang.compilation import KernelSignature
from .util import filecheck, make_symbolic_tensor, require_hopper_or_newer

MEMORY_SPACE_32B = (
    cl.MemorySpace.SHARED,
    cl.MemorySpace.SHARED_CLUSTER,
    cl.MemorySpace.TENSOR,
)


@pytest.mark.parametrize(
    "inp_dtype, out_dtype, check",
    (
        # i2f
        (cl.int16, cl.float16, "(i16) -> f16"),
        (cl.uint16, cl.float16, "(i16) -> f16"),
        (cl.int32, cl.float32, "(i32) -> f32"),
        (cl.uint32, cl.float32, "(i32) -> f32"),
        (cl.int64, cl.float64, "(i64) -> f64"),
        (cl.uint64, cl.float64, "(i64) -> f64"),
        # f2i
        (cl.float16, cl.int16, "(f16) -> i16"),
        (cl.float16, cl.uint16, "(f16) -> i16"),
        (cl.float32, cl.int32, "(f32) -> i32"),
        (cl.float32, cl.uint32, "(f32) -> i32"),
        (cl.float64, cl.int64, "(f64) -> i64"),
        (cl.float64, cl.uint64, "(f64) -> i64"),
    ),
)
def test_bitcast_scalars(inp_dtype, out_dtype, check):
    @cl.kernel
    def kernel(out):
        x = inp_dtype(0)
        out[0] = cl.bitcast(x, out_dtype)

    cres = cl.compile_simt(
        kernel,
        [KernelSignature([make_symbolic_tensor(1, out_dtype)])],
        keep_mlir=True,
    )
    filecheck(cres.mlir, "CHECK: llvm.bitcast{{.+}}" + check)


@pytest.mark.parametrize("mspace", cl.MemorySpace._member_map_.values())
@pytest.mark.parametrize("fail", (True, False))
def test_bitcast_pointer_vector(mspace, fail):
    # this is sort-of a nonsensical test because we don't have a way to represent
    # vectors as dtypes but we need to somehow create a pointer from a vector
    # and do something with it to prevent dce

    ptr_bitwidth = 32 if mspace in MEMORY_SPACE_32B else 64
    dst_dtype = getattr(cl, f"int{ptr_bitwidth}")

    # if we're forcing a type error, make the vector slightly too large for the
    # pointer to the given address space
    count = ptr_bitwidth // 8 + int(fail)

    @cl.kernel
    def kernel(out):
        v = out.get_base_pointer().load(count=count)
        p = cl.bitcast(v, cl.pointer_dtype(cl.float32, mspace))
        i = cl.bitcast(p, dst_dtype)
        out[0] = cl.int8(i)

    if fail:
        match = "bitcast requires input value's type and output type to have the same bitwidth"
        cm = pytest.raises(TypeCheckingError, match=match)
    else:
        cm = contextlib.nullcontext()

    with cm:
        sig = KernelSignature([make_symbolic_tensor(1, cl.int8)])
        cl.compile_simt(kernel, [sig])


@pytest.mark.parametrize("mspace", cl.MemorySpace._member_map_.values())
@pytest.mark.parametrize("float_dtype", (cl.float16, cl.float32, cl.float64))
def test_bitcast_pointer_float(mspace, float_dtype):
    @cl.kernel
    def kernel(out):
        f = out[0]
        p = cl.bitcast(f, cl.pointer_dtype(cl.float32, mspace))
        f = cl.bitcast(p, float_dtype)
        out[0] = f

    ptr_bitwidth = 32 if mspace in MEMORY_SPACE_32B else 64
    if float_dtype.bitwidth == ptr_bitwidth:
        cm = contextlib.nullcontext()
    else:
        match = "bitcast requires input value's type and output type to have the same bitwidth"
        cm = pytest.raises(TypeCheckingError, match=match)

    with cm:
        sig = KernelSignature([make_symbolic_tensor(1, float_dtype)])
        cl.compile_simt(kernel, [sig])


def test_bitcast_between_pointers():
    @cl.kernel
    def kernel(out):
        p1 = cl.shared_array(1, cl.int64).get_base_pointer()
        p1.store(0)
        p2 = cl.bitcast(p1, cl.pointer_dtype(cl.uint16, cl.MemorySpace.SHARED))
        p2.store(0xBEEF)
        (p2 + 1).store(0xDEAD)
        out[0] = p1.load()

    out = torch.zeros(1, dtype=torch.int64).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    got = out.cpu().item()
    assert got == 0xDEADBEEF, f"0x{got:x}"


@require_hopper_or_newer()  # Feature '::cluster' requires .target sm_90 or higher
@pytest.mark.parametrize("from_mspace", cl.MemorySpace._member_map_.values())
@pytest.mark.parametrize("to_mspace", cl.MemorySpace._member_map_.values())
def test_pointer_address_space_bitcast_compile_only(from_mspace, to_mspace):
    from_32b = from_mspace in MEMORY_SPACE_32B
    to_32b = to_mspace in MEMORY_SPACE_32B

    @cl.kernel
    def kernel(out):
        i1 = cl.int32(0) if from_32b else cl.int64(0)
        p1 = cl.bitcast(i1, cl.opaque_pointer_dtype(from_mspace))
        p2 = cl.bitcast(p1, cl.opaque_pointer_dtype(to_mspace))
        i2 = cl.bitcast(p2, (cl.int32 if to_32b else cl.int64))
        out[0] = i2

    if from_32b != to_32b:
        # If bitwidth is not the same, we expect an error
        from_bw = "32" if from_32b else "64"
        to_bw = "32" if to_32b else "64"
        match = (
            "bitcast requires input value's type and output type to have "
            f"the same bitwidth, but input type is {from_bw} bits and output "
            f"dtype has {to_bw} bits"
        )
        cm = pytest.raises(TypeCheckingError, match=match)
    elif [from_mspace, to_mspace].count(cl.MemorySpace.TENSOR) == 1:
        # if bitwidth IS the same but one address is in tensor memory, we'll
        # get an error from codegen
        cm = pytest.raises(
            CompilerExecutionError, match="Bad address space in addrspacecast"
        )
    else:
        # otherwise we should be able to compile
        cm = contextlib.nullcontext()

    with cm:
        cl.compile_simt(
            kernel,
            [
                KernelSignature(
                    [make_symbolic_tensor(1, cl.int32 if to_32b else cl.int64)]
                )
            ],
        )


def test_bitcast_to_bool():
    @cl.kernel
    def kernel(out):
        out[0] = cl.bitcast(0, cl.bool_)

    with pytest.raises(TypeCheckingError, match="bitcast to or from bool is not supported"):
        cl.compile_simt(kernel, [KernelSignature([make_symbolic_tensor(1, cl.int8)])])


def test_bitcast_from_bool():
    @cl.kernel
    def kernel(out):
        out[0] = cl.bitcast(True, cl.int8)

    with pytest.raises(TypeCheckingError, match="bitcast to or from bool is not supported"):
        cl.compile_simt(kernel, [KernelSignature([make_symbolic_tensor(1, cl.bool_)])])


def test_bitcast_from_vector():
    @cl.kernel
    def kernel(inp, out):
        v = inp.get_base_pointer().load(count=2)
        out[0] = cl.bitcast(v, cl.int64)

    inp = torch.tensor([1, 2], dtype=torch.int32).cuda()
    out = torch.zeros(1, dtype=torch.int64).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    got = out.cpu().item()
    assert got == ((2 << 32) | 1), f"{got:x}"
