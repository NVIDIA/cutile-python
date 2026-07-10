# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._stub.foreign_function import _call_foreign_function as ffi
import torch
import pytest


def get_device_function(symbolic_value):
    """
    Can inspect the types of values symbolically and determine which
    entrypoint to generate a call to.
    """
    dtype = symbolic_value.dtype
    assert dtype in (cl.float32, cl.float64)
    entrypoint = "__nv_sqrt" if dtype is cl.float64 else "__nv_sqrtf"

    def call_entrypoint(dynamic_value):
        return ffi(entrypoint, dtype, (dynamic_value,))

    return call_entrypoint


def statically_choose_ffi_entrypoint(x):
    return cl.static_eval(get_device_function(x))(x)


@cl.kernel
def kernel(out):
    out[0] = statically_choose_ffi_entrypoint(out[0])


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_ffi_libdevice(dtype):
    out = torch.tensor([4.0], dtype=dtype).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    torch.testing.assert_close(out.cpu().item(), 2.0)


def test_ffi_malloc():
    @cl.kernel
    def kernel(out):
        # TODO: add contextmanager examples
        ptr_dtype = cl.pointer_dtype(cl.int32)
        ptr: cl.Pointer = ffi("malloc", ptr_dtype, (cl.uint64(4),))
        ptr.store(75)
        out[0] = ptr.load()
        ffi("free", None, (ptr,))

    out = torch.zeros(1, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().item() == 75
