# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib

import cuda.lang as cl
import pytest
import torch

from cuda.tile._cext import cconv_v3_enabled
from cuda.lang._exception import TypeCheckingError
from cuda.lang.compilation import KernelSignature
from .util import filecheck, make_symbolic_tensor

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
    # reinterpret a whole byte vector as an int, make a pointer from it, then
    # go back to an int -- exercising the vector -> int -> pointer -> int path
    # across memory spaces.

    ptr_bitwidth = 32 if mspace in MEMORY_SPACE_32B else 64
    dst_dtype = getattr(cl, f"int{ptr_bitwidth}")

    # if we're forcing a type error, make the vector slightly too large for the
    # pointer to the given address space
    count = ptr_bitwidth // 8 + int(fail)

    @cl.kernel
    def kernel(out):
        v = out.pointer().load(count=count)
        i = v.reinterpret_as_scalar(dst_dtype)
        p = cl.bitcast(i, cl.pointer_dtype(cl.float32, mspace))
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
        p1 = cl.shared_array(1, cl.int64).pointer()
        p1.store(0)
        p2 = cl.bitcast(p1, cl.pointer_dtype(cl.uint16, cl.MemorySpace.SHARED))
        p2.store(0xBEEF)
        (p2 + 1).store(0xDEAD)
        out[0] = p1.load()

    out = torch.zeros(1, dtype=torch.int64).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    got = out.cpu().item()
    assert got == 0xDEADBEEF, f"0x{got:x}"


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_bitcast_in_host_entry():
    @cl.kernel
    def read_uint16(pointer, output):
        cl.static_assert(pointer.pointee_dtype == cl.uint16)
        output[0] = pointer[0]

    @cl.host_entry
    def launcher(source, output):
        pointer = cl.bitcast(source.pointer(), cl.pointer_dtype(cl.uint16))
        cl.launch(None, (1,), (1,), read_uint16, (pointer, output))

    source = torch.tensor([0x34, 0x12], dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    output = torch.zeros(1, dtype=torch.uint16, device="cuda")
    launcher(source, output)
    assert output.item() == 0x1234


@pytest.mark.parametrize("direct_nvvm", (False, True))
@pytest.mark.parametrize("vector", (False, True))
@pytest.mark.parametrize("opaque", (False, True))
@pytest.mark.parametrize("from_mspace", cl.MemorySpace._member_map_.values())
@pytest.mark.parametrize("to_mspace", cl.MemorySpace._member_map_.values())
def test_pointer_address_space_bitcast_compile_only(
    from_mspace, to_mspace, opaque, vector, direct_nvvm
):
    from_32b = from_mspace in MEMORY_SPACE_32B
    to_32b = to_mspace in MEMORY_SPACE_32B
    src_dtype = (cl.opaque_pointer_dtype(from_mspace) if opaque
                 else cl.pointer_dtype(cl.float32, from_mspace))
    dst_dtype = (cl.opaque_pointer_dtype(to_mspace) if opaque
                 else cl.pointer_dtype(cl.uint16, to_mspace))

    @cl.kernel
    def kernel(out):
        i1 = cl.int32(out[0]) if from_32b else cl.int64(out[0])
        if vector:
            p1 = cl.bitcast(cl.Vector(i1, i1), src_dtype)
            p2 = p1.bitcast(dst_dtype)
            i2 = cl.bitcast(p2, cl.int32 if to_32b else cl.int64)[0]
        else:
            p1 = cl.bitcast(i1, src_dtype)
            p2 = cl.bitcast(p1, dst_dtype)
            i2 = cl.bitcast(p2, cl.int32 if to_32b else cl.int64)
        out[0] = i2

    if from_32b != to_32b:
        match = ("Vector element and target dtype must have the same bitwidth" if vector
                 else "bitcast requires input value's type and output type to have "
                      "the same bitwidth")
        cm = pytest.raises(TypeCheckingError, match=match)
    elif from_mspace != to_mspace:
        cm = pytest.raises(
            TypeCheckingError,
            match="bitcast cannot change pointer memory space .*use address_space_cast",
        )
    else:
        cm = contextlib.nullcontext()

    with cm:
        cl.compile_simt(
            kernel,
            [KernelSignature([make_symbolic_tensor(1, cl.int32 if to_32b else cl.int64)])],
            gpu_name="sm_100a",
            arch="compute_100a",
            direct_nvvm=direct_nvvm,
        )


@pytest.mark.parametrize("direct_nvvm", (False, True))
def test_pointer_reinterpret_as_vector_rejects_memory_space_change(direct_nvvm):
    @cl.kernel
    def kernel(out):
        pointers = cl.bitcast(cl.Vector(cl.int64(0)), cl.pointer_dtype(cl.int8))
        # The total width stays 64 bits, even though the element widths and
        # vector lengths change. Check memory spaces before splitting into integers.
        pointers.reinterpret_as_vector(cl.pointer_dtype(cl.int8, "SHARED"), 2)

    with pytest.raises(
        TypeCheckingError,
        match="bitcast cannot change pointer memory space .*use address_space_cast",
    ):
        cl.compile_simt(
            kernel,
            [KernelSignature([make_symbolic_tensor(1, cl.int8)])],
            direct_nvvm=direct_nvvm,
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


def test_bitcast_vector_elementwise():
    @cl.kernel
    def kernel(inp, out):
        v = inp.pointer().load(count=4)
        r = cl.bitcast(v, cl.float32)
        out.pointer().store(r)

    values = torch.tensor([1.5, -2.25, 3.75, 0.5], dtype=torch.float32)
    inp = values.view(torch.int32).cuda(0)
    out = torch.zeros(4, dtype=torch.float32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out.cpu().tolist() == values.tolist()


def test_bitcast_elementwise_width_mismatch_errors():

    @cl.kernel
    def kernel(inp, out):
        v = inp.pointer().load(count=2)
        out[0] = cl.bitcast(v, cl.int64)[0]

    match = "Vector element and target dtype must have the same bitwidth"
    with pytest.raises(TypeCheckingError, match=match):
        cl.compile_simt(
            kernel,
            [KernelSignature([make_symbolic_tensor(1, cl.int32),
                              make_symbolic_tensor(1, cl.int64)])],
        )
