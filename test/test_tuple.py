# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import cuda.tile as ct
import torch

from cuda.tile import TileTypeError
from util import assert_equal


def test_tuple_concatenation():
    @ct.kernel
    def kernel(x, y, z):
        a = ct.load(x, (0,), (16,))
        b = ct.load(x, (1,), (16,))
        c = ct.load(x, (2,), (16,))
        t = (a,) + (b, c)
        ct.store(y, (0,), t[0])
        ct.store(y, (1,), t[1])
        ct.store(y, (2,), t[2])
        ct.scatter(z, (), len(t))

    x = torch.arange(48, dtype=torch.int32, device="cuda")
    y = torch.zeros((48,), dtype=torch.int32, device="cuda")
    z = torch.zeros((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, z))
    assert_equal(y, x)
    assert z.item() == 3


def test_tuple_getitem_noninteger():
    @ct.kernel
    def kernel():
        t = (1, 2, 3)
        t[1.0]

    with pytest.raises(TileTypeError, match="Tuple indices must be integers or slices"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_getitem_nonscalar():
    @ct.kernel
    def kernel():
        t = (1, 2, 3)
        i = ct.ones((2,), dtype=ct.int32)
        t[i]

    with pytest.raises(TileTypeError, match="Tuple indices must be integers or slices"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_getitem_nonconstant():
    @ct.kernel
    def kernel():
        t = (1, 2, 3)
        t[ct.bid(0)]

    with pytest.raises(TileTypeError, match="Tuple indices must be constant"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_getitem_nontile():
    @ct.kernel
    def kernel():
        t = (1, 2, 3)
        t[(4, 5)]

    with pytest.raises(TileTypeError, match="Tuple indices must be integers or slices"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_setitem():
    @ct.kernel
    def kernel():
        t = (1, 2, 3)
        t[0] = 7

    with pytest.raises(TileTypeError,
                       match="Tuples are immutable: item assignment is not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
