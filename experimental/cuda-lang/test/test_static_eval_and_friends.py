# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import cuda.lang
import cuda.lang as cl
from cuda.lang import static_eval
from cuda.tile import TileStaticAssertionError


def test_cl_static_eval():
    @cl.kernel
    def kern(a):
        a[()] = cl.static_eval([2*3].pop())

    a = torch.zeros((), dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.item() == 6


def test_cuda_lang_static_eval():
    @cl.kernel
    def kern(a):
        a[()] = cuda.lang.static_eval([2*3].pop())

    a = torch.zeros((), dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.item() == 6


def test_imported_static_eval():
    @cl.kernel
    def kern(a):
        a[()] = static_eval([2*3].pop())

    a = torch.zeros((), dtype=torch.int32, device="cuda")
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

    a = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.tolist() == [10, 20]
