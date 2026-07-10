# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch.cuda

import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError


def test_pointer_dtype():
    @cl.kernel
    def kernel():
        i32_ptr_dtype = cl.pointer_dtype(cl.int32)

        # Check is_pointer_dtype
        res = cl.is_pointer_dtype(i32_ptr_dtype)
        cl.static_assert(res)
        cl.static_assert(cl.is_pointer_dtype(i32_ptr_dtype))

        res = cl.is_pointer_dtype(cl.int32)
        cl.static_assert(not res)
        cl.static_assert(not cl.is_pointer_dtype(cl.int32))

        # Check PointerInfo.opaque
        i32_ptr_info = cl.PointerInfo(i32_ptr_dtype)
        res = i32_ptr_info.opaque
        cl.static_assert(not res)
        cl.static_assert(not i32_ptr_info.opaque)

        # Check PointerInfo.pointee_dtype
        res = i32_ptr_info.pointee_dtype
        cl.static_assert(res == cl.int32)
        cl.static_assert(i32_ptr_info.pointee_dtype == cl.int32)

        # Check PointerInfo.memory_space
        res = i32_ptr_info.memory_space
        cl.static_assert(res == cl.MemorySpace.GENERIC)
        cl.static_assert(i32_ptr_info.memory_space == cl.MemorySpace.GENERIC)

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def test_opaque_pointer_dtype():
    @cl.kernel
    def kernel():
        opaque_ptr_dtype = cl.opaque_pointer_dtype()

        # Check _pointer_dtype
        res = cl.is_pointer_dtype(opaque_ptr_dtype)
        cl.static_assert(res)
        cl.static_assert(cl.is_pointer_dtype(opaque_ptr_dtype))

        # Check PointerInfo.opaque
        ptr_info = cl.PointerInfo(opaque_ptr_dtype)
        res = ptr_info.opaque
        cl.static_assert(res)
        cl.static_assert(ptr_info.opaque)

        # Check PointerInfo.memory_space
        res = ptr_info.memory_space
        cl.static_assert(res == cl.MemorySpace.GENERIC)
        cl.static_assert(ptr_info.memory_space == cl.MemorySpace.GENERIC)

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def test_opaque_pointer_raises_on_pointee_dtype_access():
    @cl.kernel
    def kernel():
        opaque_ptr_dtype = cl.opaque_pointer_dtype()
        ptr_info = cl.PointerInfo(opaque_ptr_dtype)
        ptr_info.pointee_dtype

    with pytest.raises(TypeCheckingError, match="Opaque pointer has no pointee dtype"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())
