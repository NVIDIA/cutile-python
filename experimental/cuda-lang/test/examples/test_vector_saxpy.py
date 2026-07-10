# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
import torch


@pytest.mark.parametrize("vector_length", (2, 4, 8))
def test_vector_saxpy(vector_length):

    def load_vector_aligned(array, index):
        align = vector_length * cl.static_eval(array.dtype.bitwidth) // 8
        ep = array.get_element_pointer(index)
        return ep.load(count=vector_length, alignment=align)

    def store_vector_aligned(array, index, value):
        align = vector_length * cl.static_eval(array.dtype.bitwidth) // 8
        ep = array.get_element_pointer(index)
        ep.store(value, alignment=align)

    @cl.kernel
    def saxpy(A, X, Y, out):
        cl.static_assert(A.dtype == X.dtype == Y.dtype)
        offset = cl.block_index(0) * vector_length
        a = load_vector_aligned(A, offset)
        x = load_vector_aligned(X, offset)
        y = load_vector_aligned(Y, offset)
        axpy = a * x + y
        store_vector_aligned(out, offset, axpy)

    A, X, Y = (torch.tensor(range(256), dtype=torch.float32).cuda() for _ in range(3))
    assert A.data_ptr() % (vector_length * 4) == 0, (
        "expected alignment of cuda memory to be greater than or eqaul to vector width"
    )
    out = torch.zeros(A.shape[0], dtype=torch.float32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (A.shape[0] // vector_length,),
        (1,),
        saxpy,
        (A, X, Y, out),
    )
    out = out.cpu()
    expect = (A * X + Y).cpu()
    assert torch.allclose(out, expect)
