# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.lang as cl
from cuda.lang.compilation import KernelSignature, ScalarConstraint


@pytest.mark.parametrize("dtype", (cl.float8_e4m3fn, cl.float8_e5m2, cl.float8_e8m0fnu))
@pytest.mark.parametrize("allocation", ("shared", "dynamic_shared", "local"))
@pytest.mark.parametrize("count", (None, 4))
def test_array_fp8_encoding(dtype, allocation, count):
    def copy(source, output, array):
        i = cl.thread_index(0) * 4
        source_pointer = cl.bitcast(source.pointer(), cl.pointer_dtype(dtype))
        output_pointer = cl.bitcast(output.pointer(), cl.pointer_dtype(dtype))
        if count is None:
            for j in range(4):
                array[i + j] = source_pointer[i + j]
                output_pointer[i + j] = array[i + j]
        else:
            (array.pointer() + i).store((source_pointer + i).load(count=count))
            (output_pointer + i).store((array.pointer() + i).load(count=count))

    @cl.kernel
    def kernel(source, output, size):
        if allocation == "local":
            with cl.local_array(64, dtype) as array:
                copy(source, output, array)
        elif allocation == "shared":
            array = cl.shared_array(64, dtype)
            copy(source, output, array)
        else:
            array = cl.shared_array(size, dtype, dynamic=True)
            copy(source, output, array)

    source = (torch.arange(64) * 7 + 131).to(torch.uint8).cuda(0)
    output = torch.zeros_like(source)
    cl.launch(torch.cuda.current_stream(), (1,), (16,), kernel, (source, output, 64))
    torch.testing.assert_close(output, source)


@pytest.mark.parametrize("dtype", (cl.float8_e4m3fn, cl.float8_e5m2, cl.float8_e8m0fnu))
def test_dynamic_array_fp8_size(dtype):
    def kernel(size):
        array = cl.shared_array(size, dtype, dynamic=True)
        array[1] = array[0]

    result = cl.compile_simt(
        kernel,
        [KernelSignature((ScalarConstraint(dtype=cl.int32),))],
        gpu_name="sm_80",
        arch="compute_80",
    )

    program = result.dyn_smem_size_program
    assert program and program.opcodes and program.op_attrs
    assert program.opcodes == ["KernelArgI32"]
    assert program.op_attrs == [0]
