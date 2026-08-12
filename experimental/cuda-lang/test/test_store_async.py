# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
import cuda.lang._datatype as datatype

from .util import compile_kernel

HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}
SM100_TARGET = {"gpu_name": "sm_100a", "arch": "compute_100a"}


@pytest.mark.parametrize(
    ("dtype", "count", "instruction"),
    (
        (cl.float32, None, ".b32"),
        (cl.float64, None, ".b64"),
        (cl.float32, 2, ".v2.b32"),
        (cl.float64, 2, ".v2.b64"),
        (cl.float32, 4, ".v4.b32"),
    ),
)
def test_store_async_cluster(dtype, count, instruction):
    storage_count = count or 1

    def kernel():
        values = cl.shared_array(storage_count, dtype).get_base_pointer()
        mbarrier = cl.shared_array(1, cl.mbarrier).get_base_pointer()
        destination = cl.map_shared_to_cluster(values, 1)
        remote_mbarrier = cl.map_shared_to_cluster(mbarrier, 1)
        value = dtype(1) if count is None else values.load(count=count)
        cl.store_async_cluster(destination, value, remote_mbarrier)

    expect = f"st.async.shared::cluster.mbarrier::complete_tx::bytes{instruction}"
    compile_kernel(kernel, assert_in_ptx=expect, **HOPPER_TARGET)


@pytest.mark.parametrize(
    "dtype",
    (
        *datatype.signed_integral_dtypes,
        *datatype.unsigned_integral_dtypes,
        cl.bool_,
        cl.float16,
        cl.bfloat16,
        cl.float32,
        cl.float64,
    ),
)
def test_store_async_global(dtype):
    def kernel():
        ptr = cl.shared_array(1, dtype).get_base_pointer()
        ptr = cl.address_space_cast(ptr, cl.MemorySpace.GENERIC)
        cl.store_async_global(ptr, dtype(1))

    expect = f"st.async.release.gpu.global.b{dtype.bitwidth}"
    compile_kernel(kernel, assert_in_ptx=expect, **SM100_TARGET)


@pytest.mark.parametrize(
    ("scope", "is_multimem", "instruction"),
    (
        (cl.MemoryScope.DEVICE, False, "st.async.release.gpu"),
        (cl.MemoryScope.SYS, False, "st.async.release.sys"),
        (cl.MemoryScope.DEVICE, True, "multimem.st.async.release.gpu"),
        (cl.MemoryScope.SYS, True, "multimem.st.async.release.sys"),
    ),
)
def test_store_async_global_mode(scope, is_multimem, instruction):
    def kernel():
        ptr = cl.shared_array(1, cl.int32).get_base_pointer()
        ptr = cl.address_space_cast(ptr, cl.MemorySpace.GENERIC)
        cl.store_async_global(ptr, 1, scope=scope, is_multimem=is_multimem)

    compile_kernel(
        kernel, assert_in_ptx=f"{instruction}.global.b32", **SM100_TARGET
    )
