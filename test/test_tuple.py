# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

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

    x = torch.arange(48, dtype=torch.int32, device="cuda:0")
    y = torch.zeros((48,), dtype=torch.int32, device="cuda:0")
    z = torch.zeros((), dtype=torch.int32, device="cuda:0")
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


def test_build_tuple_starred():
    @ct.kernel
    def kernel(x):
        a = (10, 20, 30)
        b = ()
        c = (40, 50)
        t = (7, *a, 8, *b, 9, *c, 10)
        for i, v in ct.static_iter(enumerate(t)):
            ct.scatter(x, i, v)

    x = torch.zeros(9, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [7, 10, 20, 30, 8, 9, 40, 50, 10]


def test_pass_tuple_starred_to_user_defined_helper():
    def helper(arr, *items):
        for i, v in ct.static_iter(enumerate(items)):
            ct.scatter(arr, i, v)

    @ct.kernel
    def kernel(x):
        helper(x, 123, *(10, 20), 456, *(30,))

    x = torch.zeros(5, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [123, 10, 20, 456, 30]


def test_pass_tuple_starred_to_builtin():
    @ct.kernel
    def kernel(x):
        args = (x, (), 1234)
        ct.scatter(*args)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 1234


def test_pass_non_tuple_starred():
    def helper(*items):
        pass

    @ct.kernel
    def kernel():
        tile = ct.ones((4,), dtype=ct.int32)
        helper(*tile)

    with pytest.raises(TileTypeError, match=re.escape("Expected a tuple after *")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_compare_empty_eq():
    @ct.kernel
    def kernel(x):
        if () == ():
            ct.scatter(x, (), 1)
        else:
            ct.scatter(x, (), 0)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 1


def test_tuple_compare_constants_eq():
    @ct.kernel
    def kernel(x):
        if (1, 2, 3) == (1, 2, 3):
            ct.scatter(x, (), 1)
        else:
            ct.scatter(x, (), 0)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 1


def test_tuple_compare_constants_ne():
    @ct.kernel
    def kernel(x):
        if (1, 2) != (1, 3):
            ct.scatter(x, (), 1)
        else:
            ct.scatter(x, (), 0)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 1


def test_tuple_compare_different_lengths():
    @ct.kernel
    def kernel(x):
        a = ct.bid(0)
        if (a, 1) != (a, 1, 2):
            ct.scatter(x, (), 1)
        else:
            ct.scatter(x, (), 0)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 1


def test_tuple_compare_0d_tiles_eq():
    @ct.kernel
    def kernel(x):
        a = ct.bid(0)
        b = ct.bid(1)
        if (a, b) == (0, 0):
            ct.scatter(x, (a, b), 1)
        else:
            ct.scatter(x, (a, b), -1)

    x = torch.zeros((2, 2), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (2, 2), kernel, (x,))
    assert x.tolist() == [[1, -1], [-1, -1]]


def test_tuple_compare_nd_tile_error():
    @ct.kernel
    def kernel():
        t = ct.ones((4,), dtype=ct.int32)
        if (t,) == (t,):
            pass

    with pytest.raises(TileTypeError, match="not supported for non-scalar elements"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_compare_unsupported_op():
    @ct.kernel
    def kernel():
        if (1, 2) < (3, 4):
            pass

    with pytest.raises(TileTypeError, match="not supported for tuples"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_compare_nested():
    @ct.kernel
    def kernel(x):
        a = ct.bid(0)
        if ((a, 1), 2) == ((0, 1), 2):
            ct.scatter(x, (a, ), 1)
        else:
            ct.scatter(x, (a, ), -1)

    x = torch.zeros((2, ), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (2, ), kernel, (x,))
    assert x.tolist() == [1, -1]


def test_tuple_compare_array_element_error():
    @ct.kernel
    def kernel(x, y):
        if (x,) == (y,):
            pass

    with pytest.raises(TileTypeError, match="not supported for elements of type"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel,
                  (torch.zeros(4, dtype=torch.int32, device="cuda:0"),
                   torch.zeros(4, dtype=torch.int32, device="cuda:0")))


def test_tuple_compare_constant_args():
    @ct.kernel
    def kernel(x, M: ct.Constant[int], N: ct.Constant[int]):
        if (M, N) == (4, 8):
            ct.scatter(x, (), 1)
        else:
            ct.scatter(x, (), -1)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 4, 8))
    assert x.item() == 1
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 4, 9))
    assert x.item() == -1


def test_tuple_truthiness():
    @ct.kernel
    def kernel(x):
        empty = ()
        nonempty = (0,)
        if nonempty:            # True
            ct.scatter(x, 0, 1)
        if empty:               # False
            ct.scatter(x, 1, 1)
        ct.scatter(x, 2, 1 if nonempty else 0)
        ct.scatter(x, 3, 1 if empty else 0)
        if nonempty and ct.bid(0) == 0:
            ct.scatter(x, 4, 1)
        if empty or ct.bid(0) == 1:
            ct.scatter(x, 5, 1)

    x = torch.zeros(6, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [1, 0, 1, 0, 1, 0]


def test_element_in_tuple():
    @ct.kernel
    def kernel(x):
        if 2 in (1, 2, 3):      # True
            ct.scatter(x, 0, 1)
        if 5 in (1, 2, 3):      # False
            ct.scatter(x, 1, 1)
        if 1 in ():              # False (empty)
            ct.scatter(x, 2, 1)
        if 5 not in (1, 2, 3):  # True
            ct.scatter(x, 3, 1)

    x = torch.zeros(4, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [1, 0, 0, 1]


def test_element_in_tuple_runtime():
    @ct.kernel
    def kernel(x):
        a = ct.bid(0)
        if a in (0, 2):
            ct.scatter(x, (a,), 1)

    x = torch.zeros(4, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (4,), kernel, (x,))
    assert x.tolist() == [1, 0, 1, 0]


def test_element_in_tuple_shortcircuit():
    @ct.kernel
    def kernel(x):
        t = ct.ones((4,), dtype=ct.int32)
        if 2 in (2, t):       # match at pos 0
            ct.scatter(x, 0, 1)
        if 2 in (1, 2, t):    # match at pos 1 after a false
            ct.scatter(x, 1, 1)
        if 2 not in (2, t):   # short-circuit → False
            ct.scatter(x, 2, 1)

    x = torch.zeros(3, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [1, 1, 0]


def test_element_in_tuple_tensor_raises():
    # tensor as needle; tensor before match; tensor in middle after a false
    @ct.kernel
    def k_needle():
        ct.ones((4,), dtype=ct.int32) in (1, 2)

    @ct.kernel
    def k_tensor_first():
        2 in (ct.ones((4,), dtype=ct.int32), 2)

    @ct.kernel
    def k_tensor_middle():
        2 in (1, ct.ones((4,), dtype=ct.int32), 2)

    stream = torch.cuda.current_stream()
    for k in (k_needle, k_tensor_first, k_tensor_middle):
        with pytest.raises(TileTypeError, match="'in' requires scalar operands"):
            ct.launch(stream, (1,), k, ())


def test_tuple_in_tuple_nested():
    @ct.kernel
    def kernel(x):
        a = ct.bid(0)
        if (a, 1) in ((0, 1), (2, 1)):
            ct.scatter(x, (a,), 1)
        else:
            ct.scatter(x, (a,), -1)

    x = torch.zeros(4, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (4,), kernel, (x,))
    assert x.tolist() == [1, -1, 1, -1]


def test_tuple_global_capture():
    tup = (100, (101, 102), (103, (104, 105)))

    @ct.kernel
    def kernel(x):
        ct.scatter(x, 0, tup[0])
        ct.scatter(x, 1, tup[2][1][1])

    x = torch.zeros(2, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [100, 105]
