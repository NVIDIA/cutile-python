# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError
from cuda.lang._stub.tcgen05 import (
    _tcgen05_encode_dtype as encode_dtype,
    _Tcgen05DTypeFormat as DTypeFormat,
)
from test.util import compile_kernel


def encode_tcgen05_instruction_descriptor():
    return cl.Tcgen05InstructionDescriptor(
        sparsity_selector=3,
        sparse=True,
        saturate=True,
        d_type=cl.int32,
        a_type=cl.int8,
        b_type=cl.uint8,
        negate_a=True,
        negate_b=False,
        transpose_a=True,
        transpose_b=False,
        n=248,
        m=240,
        max_shift=cl.Tcgen05InstructionDescriptor.MaxShift.MaxShift16,
    ).encode()


def encode_tcgen05_mxf8f6f4_instruction_descriptor():
    return cl.Tcgen05Mxf8f6f4InstructionDescriptor(
        sparse=True,
        b_scale_id=3,
        a_type=cl.float6_e2m3fn,
        b_type=cl.float6_e3m2fn,
        negate_a=True,
        negate_b=False,
        transpose_a=False,
        transpose_b=True,
        n=128,
        scale_format=cl.Tcgen05Mxf8f6f4InstructionDescriptor.ScaleFormat.UE8M0,
        m=256,
        a_scale_id=2,
    ).encode()


def encode_tcgen05_mxf4_instruction_descriptor():
    return cl.Tcgen05Mxf4InstructionDescriptor(
        sparse=True,
        b_scale_id=2,
        a_type=cl.float4_e2m1fn,
        b_type=cl.float4_e2m1fn,
        negate_a=False,
        negate_b=True,
        transpose_a=True,
        transpose_b=False,
        n=64,
        scale_format=cl.Tcgen05Mxf4InstructionDescriptor.ScaleFormat.UE4M3,
        m=128,
        a_scale_id=2,
        k_dimension=cl.Tcgen05Mxf4InstructionDescriptor.KDimension.DenseK96,
    ).encode()


def encode_tcgen05_shared_memory_descriptor():
    return cl.Tcgen05SharedMemoryDescriptor(
        matrix_start_address=0x12340,
        leading_dimension_byte_offset=0x23450,
        stride_dimension_byte_offset=0x34560,
        base_offset=5,
        leading_dimension_mode=(
            cl.Tcgen05SharedMemoryDescriptor.LeadingDimensionMode.ByteAddressAbsolute
        ),
        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
    ).encode()


@pytest.mark.parametrize(
    "encode_descriptor,expected",
    [
        (
            encode_tcgen05_instruction_descriptor,
            (3 << 0)
            | (1 << 2)
            | (1 << 3)
            | (2 << 4)
            | (1 << 7)
            | (0 << 10)
            | (1 << 13)
            | (1 << 15)
            | ((248 >> 3) << 17)
            | ((240 >> 4) << 24)
            | (2 << 30),
        ),
        (
            encode_tcgen05_mxf8f6f4_instruction_descriptor,
            (1 << 2)
            | (3 << 4)
            | (3 << 7)
            | (4 << 10)
            | (1 << 13)
            | (1 << 16)
            | ((128 >> 3) << 17)
            | (1 << 23)
            | ((256 >> 7) << 27)
            | (2 << 29),
        ),
        (
            encode_tcgen05_mxf4_instruction_descriptor,
            (1 << 2)
            | (2 << 4)
            | (1 << 7)
            | (1 << 10)
            | (1 << 14)
            | (1 << 15)
            | ((64 >> 3) << 17)
            | (0 << 23)
            | ((128 >> 7) << 27)
            | (2 << 29)
            | (1 << 31),
        ),
        (
            encode_tcgen05_shared_memory_descriptor,
            (((0x12340 & 0x3FFFF) >> 4) << 0)
            | (((0x23450 & 0x3FFFF) >> 4) << 16)
            | (((0x34560 & 0x3FFFF) >> 4) << 32)
            | (0b001 << 46)
            | (5 << 49)
            | (1 << 52)
            | (2 << 61),
        ),
    ],
)
def test_tcgen05_instruction_descriptor_encode_on_gpu(encode_descriptor, expected):
    @cl.kernel
    def kernel(out):
        out[0] = encode_descriptor()

    out = torch.zeros(1, dtype=torch.int64, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().item() == expected


def test_tcgen05_dtype_formats_on_host():
    assert encode_dtype(cl.float32, DTypeFormat.Output) == 1
    assert encode_dtype(cl.float6_e2m3fn, DTypeFormat.Input) == 3
    assert encode_dtype(cl.float6_e3m2fn, DTypeFormat.Mxf8f6f4Input) == 4
    assert encode_dtype(cl.float4_e2m1fn, DTypeFormat.Mxf4Input) == 1


def test_tcgen05_fp6_input_types():
    @cl.kernel
    def kernel(out):
        desc = cl.Tcgen05InstructionDescriptor(
            a_type=cl.float6_e2m3fn, b_type=cl.float6_e3m2fn
        )
        out[0] = (desc.encode() >> 7) & 0x3F

    out = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.item() == 3 | (4 << 3)


def test_tcgen05_runtime_input_format():
    @cl.kernel
    def kernel(value, out):
        out[0] = cl.Tcgen05InstructionDescriptor(a_type=value).encode()

    value = encode_dtype(cl.bfloat16, DTypeFormat.Input)
    out = torch.zeros(1, dtype=torch.uint32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (value, out))
    assert out.item() == 1 << 7


def test_tcgen05_rejects_unsupported_dtype():
    def kernel():
        cl.Tcgen05Mxf8f6f4InstructionDescriptor(a_type=cl.float16).encode()

    raises = pytest.raises(TypeCheckingError, match="Mxf8f6f4Input")
    compile_kernel(kernel, raises=raises)


@pytest.mark.parametrize(
    "swizzle_mode,expected_encoding",
    (
        (cl.SwizzleMode.SWIZZLE_NONE, 0),
        (cl.SwizzleMode.SWIZZLE_128B_ATOM_32B, 1),
        (cl.SwizzleMode.SWIZZLE_128B, 2),
        (cl.SwizzleMode.SWIZZLE_64B, 4),
        (cl.SwizzleMode.SWIZZLE_32B, 6),
    ),
)
def test_tcgen05_shared_memory_descriptor_swizzle_encoding(
    swizzle_mode, expected_encoding
):
    @cl.kernel
    def kernel(out):
        descriptor = cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=0,
            leading_dimension_byte_offset=0,
            stride_dimension_byte_offset=0,
            swizzle_mode=swizzle_mode,
        )
        out[0] = descriptor.encode() >> 61

    out = torch.zeros(1, dtype=torch.int64, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().item() == expected_encoding


def test_tcgen05_shared_memory_descriptor_pointer_encoding():
    @cl.kernel
    def kernel(out):
        mat = cl.shared_array(16, cl.int8)
        descriptor = cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=mat.pointer(),
            leading_dimension_byte_offset=0,
            stride_dimension_byte_offset=0,
        )
        out[0] = descriptor.encode()

    out = torch.zeros(1, dtype=torch.int64).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))


def test_tcgen05_shared_memory_descriptor_array_encoding():
    @cl.kernel
    def kernel(out):
        mat = cl.shared_array(16, cl.int8)
        descriptor = cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=mat,
            leading_dimension_byte_offset=0,
            stride_dimension_byte_offset=0,
        )
        out[0] = descriptor.encode()

    out = torch.zeros(1, dtype=torch.int64).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))


@pytest.mark.parametrize("dtype", (cl.int32, cl.uint32, cl.int64, cl.uint64))
def test_tcgen05_shared_memory_descriptor_int_encoding(dtype):
    def cast(pointer, dtype):
        if cl.ensure_constant(dtype.bitwidth == 64):
            pointer = cl.address_space_cast(pointer, cl.MemorySpace.GENERIC)
        return cl.bitcast(pointer, dtype)

    @cl.kernel
    def kernel(out):
        mat = cl.shared_array(16, cl.int8)
        ptr = mat.pointer()
        intval = cast(ptr, dtype)
        descriptor = cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=intval,
            leading_dimension_byte_offset=0,
            stride_dimension_byte_offset=0,
        )
        out[0] = descriptor.encode()

    out = torch.zeros(1, dtype=torch.int64).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
