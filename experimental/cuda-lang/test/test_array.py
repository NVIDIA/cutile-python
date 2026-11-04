# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError

import torch
import pytest


def test_array_constant_shape_strides():
    @cl.kernel
    def kern(arr: Annotated[cl.Array, cl.ArrayAnnotation(static_shape_dims=(0,),
                                                         static_stride_dims=(1,))]):
        shape = arr.shape
        strides = arr.strides
        cl.static_assert(shape[0] == 5)
        cl.static_assert(strides[1] == 1)
        cl.static_assert(isinstance(shape[1], cl.Scalar))
        cl.static_assert(isinstance(strides[0], cl.Scalar))

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (torch.ones((5, 7), device="cuda"),))


def test_array_permute():
    @cl.kernel
    def kernel(A: Annotated[cl.Array, cl.ArrayAnnotation(index_dtype=cl.int64)], out):
        A_perm_1 = A.permute()
        A_perm_2 = cl.Array.permute(A, (2, 0, 1))
        out[0] = A_perm_1[0, 0, 0]
        out[1] = A_perm_2[1, 0, 1]
        out[2] = A_perm_1[0, 1, 0]
        out[3] = A_perm_2[0, 0, 1]
        out[4] = A_perm_1[1, 0, 0]
        out[5] = A_perm_2[0, 0, 0]

    A = torch.arange(4, dtype=torch.float32, device="cuda:0").reshape((1, 2, 2))
    out = torch.zeros(6, dtype=torch.float32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.tolist() == [0, 3, 2, 2, 1, 0]


@pytest.mark.parametrize(
    "dims, match", (
        (1, "Expected a tuple, but given value has type"),
        ((0, 1), "do not match array of shape"),
        ((0, 1, 1), "repeated axis in permute dims"),
        ((0, 1, 3), "axis 3 is out of bounds for array of dimension"),
        ((-4, 0, 1), "axis -4 is out of bounds for array of dimension"),
        ((0.0, 1, 2), "Expected a tuple of integers, but element")
    )
)
def test_array_permute_reject_dims(dims, match):
    @cl.kernel
    def kernel(A):
        A.permute(dims)

    A = torch.zeros((2, 3, 4), dtype=torch.float32, device="cuda:0")
    with pytest.raises(TypeCheckingError, match=match):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A,))
