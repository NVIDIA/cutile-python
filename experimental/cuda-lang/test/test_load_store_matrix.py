# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._compile import KernelSignature
from cuda.lang._exception import InvalidValueError

from .util import compile_kernel, make_symbolic_tensor


HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}
SM100_TARGET = {"gpu_name": "sm_100a", "arch": "compute_100a"}
SIG_I32 = KernelSignature([make_symbolic_tensor((4,), cl.int32)])


def test_load_matrix_scalar():
    def kernel(output):
        smem = cl.shared_array(64, cl.int16, alignment=16)
        address = smem.get_element_pointer(cl.thread_index(0) * 8)
        value = cl.load_matrix(address, shape=cl.MatrixLoadShape.M8N8)
        output[0] = value

    compile_kernel(
        kernel,
        signature=SIG_I32,
        assert_in_mlir="nvvm.ldmatrix",
        assert_in_ptx="ldmatrix.sync.aligned.m8n8.x1.shared.b16",
    )


def test_load_matrix_vector():
    def kernel(output):
        smem = cl.shared_array(128, cl.int16, alignment=16)
        address = smem.get_element_pointer(cl.thread_index(0) * 8)
        value = cl.load_matrix(
            address,
            shape=cl.MatrixLoadShape.M8N8,
            count=2,
            transpose=True,
        )
        output[0] = value[0]
        output[1] = value[1]

    compile_kernel(
        kernel,
        signature=SIG_I32,
        assert_in_mlir="nvvm.ldmatrix",
        assert_in_ptx="ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
    )


def test_load_matrix_packed_m16n16():
    def kernel(output):
        smem = cl.shared_array(256, cl.int8, alignment=16)
        address = smem.get_element_pointer(cl.thread_index(0) * 16)
        value = cl.load_matrix(
            address,
            shape=cl.MatrixLoadShape.M16N16,
            transpose=True,
            source_format=cl.MatrixLoadSourceFormat.B6X16_P32,
        )
        output[0] = value[0]
        output[1] = value[1]

    compile_kernel(
        kernel,
        signature=SIG_I32,
        assert_in_ptx="ldmatrix.sync.aligned.m16n16.x1.trans",
        **SM100_TARGET,
    )


@pytest.mark.parametrize("trans", (True, False))
def test_store_matrix_vector(trans):
    def kernel():
        smem = cl.shared_array(128, cl.int16, alignment=16)
        values = cl.Vector(cl.uint32(1), cl.uint32(2))
        cl.store_matrix(
            smem.get_base_pointer(),
            values,
            shape=cl.MatrixStoreShape.M8N8,
            transpose=trans,
        )

    compile_kernel(
        kernel,
        assert_in_mlir="nvvm.stmatrix",
        assert_in_ptx=(
            "stmatrix.sync.aligned.m8n8.x2"
            + (".trans" if trans else "")
            + ".shared.b16"
        ),
        **HOPPER_TARGET,
    )


def test_store_matrix_m16n8():
    def kernel():
        smem = cl.shared_array(128, cl.int8, alignment=16)
        cl.store_matrix(
            smem.get_base_pointer(),
            cl.Vector(cl.int32(1), cl.int32(2)),
            shape=cl.MatrixStoreShape.M16N8,
            transpose=True,
        )

    compile_kernel(
        kernel,
        assert_in_ptx="stmatrix.sync.aligned.m16n8.x2.trans",
        **SM100_TARGET,
    )


def test_load_matrix_rejects_invalid_count():
    def kernel():
        smem = cl.shared_array(128, cl.int8, alignment=16)
        cl.load_matrix(
            smem.get_base_pointer(),
            shape=cl.MatrixLoadShape.M8N8,
            count=3,
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(InvalidValueError, match="count must be 1, 2, or 4"),
    )


def test_store_matrix_rejects_invalid_register_count():
    def kernel():
        smem = cl.shared_array(128, cl.int16, alignment=16)
        values = cl.Vector(cl.int32(1), cl.int32(2), cl.int32(3))
        cl.store_matrix(
            smem.get_base_pointer(),
            values,
            shape=cl.MatrixStoreShape.M8N8,
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(InvalidValueError, match="register count"),
    )
