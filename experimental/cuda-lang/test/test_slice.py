# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import torch
import pytest
from cuda.lang._exception import TypeCheckingError


@pytest.mark.parametrize("index, result",
                         [((None, 4, None), [0, 1, 2, 3]),
                          ((4, 8, None), [4, 5, 6, 7]),
                          ((4, 12, 2), [4, 6, 8, 10])])
def test_array_slice_view(index, result):
    @cl.kernel
    def kernel(a, b):
        i = slice(index[0], index[1], index[2])
        view = a[i]
        b[0] = view[0]
        b[1] = view[1]
        b[2] = view[2]
        b[3] = view[3]

    a = torch.arange(16, dtype=torch.int32).cuda()
    b = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b))
    assert b.cpu().tolist() == result


def test_array_slice_view_2d_array():
    @cl.kernel
    def kernel(a, b):
        view = a[1, 4:]
        b[0] = view[0]
        b[1] = view[1]
        b[2] = view[2]
        b[3] = view[3]

    a = torch.arange(16, dtype=torch.int32).reshape((2, 8)).cuda()
    b = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b))
    assert b.cpu().tolist() == [12, 13, 14, 15]


def test_array_slice_view_runtime_slice():
    @cl.kernel
    def kernel(a, b, i):
        start, step = i[0], i[1]
        view = a[start::step]
        b[0] = view[0]
        b[1] = view[1]
        b[2] = view[2]
        b[3] = view[3]

    a = torch.arange(16, dtype=torch.int32).cuda()
    b = torch.zeros(4, dtype=torch.int32).cuda()
    i = torch.tensor([8, 2], dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b, i))
    assert b.cpu().tolist() == [8, 10, 12, 14]


def test_array_slice_view_static_shape():
    @cl.kernel
    def kernel(a):
        b = a[:, :]
        view = b[:3:1, :]
        cl.static_assert(view.shape[0] == 3)

    a = torch.arange(16, dtype=torch.int32).reshape((4, 4)).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a,))


def test_array_slice_view_stored_slice():
    @cl.kernel
    def kernel(a, b, i):
        if i[0] > 1:
            s = slice(i[1], None, 1)
        else:
            s = slice(i[0], None, 1)
        view = a[s]
        b[0] = view[0]
        b[1] = view[1]
        b[2] = view[2]
        b[3] = view[3]

    a = torch.arange(16, dtype=torch.int32).cuda()
    b = torch.zeros(4, dtype=torch.int32).cuda()
    i = torch.tensor([8, 2], dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b, i))
    assert b.cpu().tolist() == [2, 3, 4, 5]


def test_array_slice_view_external_slice():
    s = slice(1, 3, 1)

    @cl.kernel
    def kernel(a, b):
        view = a[s, ...]
        b[0] = view[0, 0]
        b[1] = view[1, 0]
        b[2] = view[1, 1]
        b[3] = view[0, 1]

    a = torch.arange(16, dtype=torch.int32).reshape((4, 4)).cuda()
    b = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b))
    assert b.cpu().tolist() == [4, 8, 9, 5]


def test_array_slice_view_ellipsis():

    @cl.kernel
    def kernel(a, b):
        view = a[0, ..., 3]
        view = view[..., :, :]
        b[0] = view[0, 0]
        b[1] = view[1, 1]
        b[2] = view[2, 2]
        b[3] = view[3, 3]

    a = torch.arange(256, dtype=torch.int32).reshape((4, 4, 4, 4)).cuda()
    b = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b))
    assert b.cpu().tolist() == [3, 23, 43, 63]


@pytest.mark.parametrize("start, stop, step, message",
                         [(-2, None, None, "A non-negative slice start is required"),
                          (None, -1, None, "A non-negative slice stop is required"),
                          (None, None, 0, "A positive slice step is required")])
def test_array_slice_view_static_index_reject(start, stop, step, message):
    @cl.kernel
    def kernel(a):
        a[start:stop:step]

    a = torch.arange(16, dtype=torch.int32).cuda()
    with pytest.raises(TypeCheckingError, match=message):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a,))


def test_array_slice_view_static_array_bounds_reject():
    @cl.kernel
    def kernel():
        a = cl.shared_array(16, cl.int32)
        a[1:18:2]

    with pytest.raises(TypeCheckingError, match="The provided slice stop"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def test_array_slice_view_static_view_bounds_reject():
    @cl.kernel
    def kernel(a):
        view = a[12:14]
        view[:3]

    a = torch.arange(16, dtype=torch.int32).cuda()
    with pytest.raises(TypeCheckingError, match="The provided slice stop"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a,))


def test_array_slice_view_none_expand_reject():
    @cl.kernel
    def kernel(a):
        a[None, :]

    a = torch.arange(16, dtype=torch.int32).reshape(4, 4).cuda()
    with pytest.raises(TypeCheckingError, match="The provided slice index is not valid"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a,))
