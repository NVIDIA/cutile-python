# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch.cuda
import pytest

import cuda.lang as cl
from test.util import compile_kernel


def test_dtype_of():
    @cl.kernel
    def kern(x: cl.Array):
        ptr = x.get_base_pointer()
        cl.static_assert(cl.dtype_of(ptr) == cl.pointer_dtype(cl.int32))
        ptr_dtype = cl.dtype_of(ptr)
        cl.static_assert(ptr_dtype == cl.pointer_dtype(cl.int32))
        cl.static_assert(ptr_dtype.bitwidth == 64)

        val = ptr.load()
        cl.static_assert(cl.dtype_of(val) == cl.int32)
        val_dtype = cl.dtype_of(val)
        cl.static_assert(val_dtype == cl.int32)
        cl.static_assert(val_dtype.bitwidth == 32)

        cl.static_assert(cl.dtype_of(123) == cl.int32)
        int_dtype = cl.dtype_of(123)
        cl.static_assert(int_dtype == cl.int32)

        cl.static_assert(cl.dtype_of(1.5) == cl.float32)
        float_dtype = cl.dtype_of(1.5)
        cl.static_assert(float_dtype == cl.float32)
        cl.static_assert(float_dtype.bitwidth == 32)

        vec = cl.Vector(1, 2, 3, 4, dtype=cl.int8)
        cl.static_assert(cl.dtype_of(vec) == cl.int8)
        vec_dtype = cl.dtype_of(vec)
        cl.static_assert(vec_dtype == cl.int8)
        cl.static_assert(vec_dtype.bitwidth == 8)

    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x,))


@pytest.mark.parametrize(
    "memory_space, bitwidth",
    (
        (cl.MemorySpace.GENERIC, 64),
        (cl.MemorySpace.GLOBAL, 64),
        (cl.MemorySpace.SHARED, 32),
        (cl.MemorySpace.CONSTANT, 64),
        (cl.MemorySpace.LOCAL, 64),
        (cl.MemorySpace.TENSOR, 32),
        (cl.MemorySpace.SHARED_CLUSTER, 32),
    ),
)
def test_pointer_dtype_bitwidth(memory_space, bitwidth):
    def kernel():
        p = cl.shared_array(1, cl.int8).get_base_pointer()
        p = cl.address_space_cast(p, memory_space)
        dtype = cl.dtype_of(p)
        cl.static_assert(dtype.bitwidth == bitwidth)

    compile_kernel(kernel)
