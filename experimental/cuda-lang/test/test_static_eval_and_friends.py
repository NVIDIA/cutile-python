# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest
import torch

import cuda.lang
import cuda.lang as cl
from cuda.lang import static_eval
from cuda.tile import TileStaticAssertionError, StaticException


def test_cl_static_eval():
    @cl.kernel
    def kern(a):
        a[()] = cl.static_eval([2*3].pop())

    a = torch.zeros((), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.item() == 6


def test_cuda_lang_static_eval():
    @cl.kernel
    def kern(a):
        a[()] = cuda.lang.static_eval([2*3].pop())

    a = torch.zeros((), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.item() == 6


def test_imported_static_eval():
    @cl.kernel
    def kern(a):
        a[()] = static_eval([2*3].pop())

    a = torch.zeros((), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.item() == 6


def test_static_assert():
    @cl.kernel
    def kern():
        cl.static_assert(False, "Boo")

    with pytest.raises(TileStaticAssertionError, match="Static assertion failed: Boo"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, ())


def test_static_iter():
    @cl.kernel
    def kern(a):
        for i, x in cl.static_iter(enumerate([10, 20])):
            a[i] = x

    a = torch.zeros(2, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.tolist() == [10, 20]


def test_static_eval_shift_operators():
    @cl.kernel
    def kern(x, y):
        value = x[0]
        y[0] = cl.static_eval(value << 3)
        y[1] = cl.static_eval(value >> 2)

    x = torch.tensor([-16], dtype=torch.int32, device="cuda:0")
    y = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x, y,))
    assert y.tolist() == [-128, -4]


def test_static_eval_pointer_arithmetic():
    @cl.kernel
    def kern(a):
        p = a.pointer()
        p2 = cl.static_eval(p + 3)
        p2[0] = 5
        p3 = cl.static_eval(p2 - 1)
        p3[0] = 7
        tid = cl.thread_index(0) + 1
        p4 = cl.static_eval(tid + p)
        p4[0] = 9
        p5 = cl.static_eval(0 + p)
        p5[0] = 11

    a = torch.zeros((4,), dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.tolist() == [11, 9, 7, 5]


def test_static_exception():
    @cl.kernel
    def kern():
        raise cl.static_exception(ValueError("Hello"))

    with pytest.raises(StaticException,
                       match=re.escape("Exception was raised at compile time (ValueError: Hello)")):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, ())


def test_static_eval_dispatch_binop_via_rhs():
    class Custom:
        def __init__(self, val):
            self.val = val

        def __radd__(self, other):
            return other + self.val

    @cl.kernel
    def kern(x):
        tid = cl.thread_index(0)
        x[tid] = cl.static_eval(tid + Custom(5))

        vec = cl.Vector(tid + 1, tid + 2)
        new_vec = cl.static_eval(vec + Custom(50))
        x[4 + tid] = new_vec[0]
        x[8 + tid] = new_vec[1]

    x = torch.zeros(12, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (4,), kern, (x,))
    assert x.tolist() == [5, 6, 7, 8, 51, 52, 53, 54, 52, 53, 54, 55]
