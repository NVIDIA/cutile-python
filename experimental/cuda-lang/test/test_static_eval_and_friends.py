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

    a = torch.zeros((4,), dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.tolist() == [0, 0, 7, 5]


def test_static_exception():
    @cl.kernel
    def kern():
        raise cl.static_exception(ValueError("Hello"))

    with pytest.raises(StaticException,
                       match=re.escape("Exception was raised at compile time (ValueError: Hello)")):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, ())
