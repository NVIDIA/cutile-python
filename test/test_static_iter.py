# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest

import cuda.tile as ct
import torch

from util import assert_equal


@pytest.mark.parametrize("flavor", ["enumerate", "range"])
def test_static_iter(flavor):
    @ct.kernel
    def kernel_enumerate(x, y):
        a = ct.load(x, (0,), (16,))
        b = ct.load(x, (1,), (16,))
        c = ct.load(x, (2,), (16,))
        tiles = a, b, c
        s = ct.zeros((16,), dtype=x.dtype)
        for i, x in ct.static_iter(enumerate(tiles, 1)):
            s += i * x
        ct.store(y, (0,), s)

    @ct.kernel
    def kernel_range(x, y):
        a = ct.load(x, (0,), (16,))
        b = ct.load(x, (1,), (16,))
        c = ct.load(x, (2,), (16,))
        tiles = a, b, c
        s = ct.zeros((16,), dtype=x.dtype)
        for i in ct.static_iter(range(len(tiles))):
            s += (i + 1) * tiles[i]
        ct.store(y, (0,), s)

    kernel = locals()[f"kernel_{flavor}"]

    x = torch.randint(0, 100, (48,), dtype=torch.int32, device="cuda")
    y = torch.zeros((16,), dtype=torch.int32, device="cuda")
    ref = x[:16] + x[16:32] * 2 + x[32:] * 3
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_static_iter_continue_not_allowed():
    @ct.kernel
    def kernel(x):
        for i in ct.static_iter(range(3)):
            if i == 1:
                continue
            ct.scatter(x, i, 1)

    x = torch.zeros((10,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileSyntaxError,
                       match="Continue in a for loop with static_iter\\(\\) is not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_static_iter_break_not_allowed():
    @ct.kernel
    def kernel(x):
        for i in ct.static_iter(range(3)):
            if i == 1:
                break
            ct.scatter(x, i, 1)

    x = torch.zeros((10,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileSyntaxError,
                       match="Break in a for loop is not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_static_iter_return_not_allowed():
    @ct.kernel
    def kernel(x):
        for i in ct.static_iter(range(3)):
            if i == 1:
                return
            ct.scatter(x, i, 1)

    x = torch.zeros((10,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileSyntaxError,
                       match="Returning from a for loop is not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_static_iter_tile_ops_not_allowed_in_iterable():
    @ct.kernel
    def kernel(x):
        for i in ct.static_iter(ct.ones((4,), dtype=ct.int32)):
            ct.scatter(x, i, 1)

    x = torch.zeros((10,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticEvalError,
                       match="Tile functions cannot be called inside static_iter\\(\\) iterable"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_static_iter_too_many_iterations():
    @ct.kernel
    def kernel(x):
        for i in ct.static_iter(range(1000000)):
            ct.scatter(x, i, 1)

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticEvalError,
                       match=re.escape("Maximum number of iterations (1000) has been reached"
                                       " while unpacking the static_iter() iterable")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_static_iter_non_iterable():
    @ct.kernel
    def kernel(x):
        for i in ct.static_iter(42):
            ct.scatter(x, i, 1)

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileTypeError,
                       match=re.escape("Invalid static_iter() iterable:"
                                       " 'int' object is not iterable")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_static_iter_unsupported_item():
    @ct.kernel
    def kernel(x):
        for i in ct.static_iter(([1, 2], [3, 4])):
            ct.scatter(x, 0, i)

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticEvalError,
                       match=re.escape("Invalid item #0 of static_iter() iterable:"
                                       " Cannot create constant from value of type list.")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_static_iter_outside_for_loop():
    @ct.kernel
    def kernel(x):
        t = ct.static_iter(range(3))
        ct.scatter(x, 0, t[0])

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileSyntaxError,
                       match=re.escape("static_iter() is only allowed as iterable"
                                       " in a `for` loop")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_static_iter_called_indirectly():
    @ct.kernel
    def kernel(x):
        f = ct.static_iter
        for i in f(range(3)):
            ct.scatter(x, i, 1)

    x = torch.zeros((10,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileSyntaxError,
                       match=re.escape("static_iter() must be used directly by name")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_static_iter_dynamic_bound():
    @ct.kernel
    def kernel(x, n):
        for i in ct.static_iter(range(n)):
            ct.scatter(x, i, 1)

    x = torch.zeros((10,), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileValueError,
                       match=re.escape("Symbolic tile has no concrete value"
                                       " and thus cannot be converted to an integer")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 3))


def test_static_iter_with_inner_for_loop():
    @ct.kernel
    def kernel(x, y):
        for col in ct.static_iter(range(3)):
            s = ct.zeros((16,), dtype=x.dtype)
            for row in range(4):
                s += ct.load(x, (row, col), (1, 16)).reshape((16,))
            ct.store(y, (col,), s)

    x = torch.arange(192, dtype=torch.int32, device="cuda").reshape(4, 3 * 16)
    y = torch.zeros((3 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, x.sum(dim=0).to(torch.int32))


def test_static_iter_with_outer_for_loop():
    @ct.kernel
    def kernel(x, y):
        for col in range(3):
            s = ct.zeros((16,), dtype=x.dtype)
            for row in ct.static_iter(range(4)):
                s += ct.load(x, (row, col), (1, 16)).reshape((16,))
            ct.store(y, (col,), s)

    x = torch.arange(192, dtype=torch.int32, device="cuda").reshape(4, 3 * 16)
    y = torch.zeros((3 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, x.sum(dim=0).to(torch.int32))


def test_static_iter_with_inner_while_break():
    @ct.kernel
    def kernel(x, y):
        for col in ct.static_iter(range(3)):
            s = ct.zeros((16,), dtype=x.dtype)
            row = 0
            while row < 4:
                t = ct.load(x, (row, col), (1, 16)).reshape((16,))
                s += t
                if ct.sum(t) == 0:
                    break
                row += 1
            ct.store(y, (col,), s)

    x = torch.arange(192, dtype=torch.int32, device="cuda").reshape(4, 3 * 16)
    y = torch.zeros((3 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, x.sum(dim=0).to(torch.int32))


def test_static_iter_tuple_concatenation():
    @ct.kernel
    def kernel(x, y):
        a = ct.load(x, (0,), (16,))
        b = ct.load(x, (1,), (16,))
        c = ct.load(x, (2,), (16,))
        tiles = (a, b, c)
        doubled = ()
        for t in ct.static_iter(tiles):
            doubled += (t, t * 2)
        for i, d in ct.static_iter(enumerate(doubled)):
            ct.store(y, (i,), d)

    x = torch.arange(3 * 16, dtype=torch.int32, device="cuda")
    a, b, c = x[:16], x[16:32], x[32:]
    ref = torch.cat([a, a * 2, b, b * 2, c, c * 2])

    y = torch.zeros((6 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_static_iter_mixed_types():
    @ct.kernel
    def kernel(x):
        t = 0.0
        for i, val in ct.static_iter(enumerate([2, 3.0, True])):
            t += val
            ct.scatter(x, i, t)

    x = torch.zeros((3,), dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [2.0, 5.0, 6.0]
