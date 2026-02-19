# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import torch
import pytest

import cuda.tile
import cuda.tile as ct
from cuda.tile import TileStaticEvalError


def test_tuple_sum():
    @ct.kernel
    def kernel(y):
        tup = (1, 2, 3)
        s1 = ct.static_eval(sum(tup))
        s2 = cuda.tile.static_eval(sum(tup))
        ct.scatter(y, 0, s1)
        ct.scatter(y, 1, s2)

    y = torch.zeros((2,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == [6, 6]


def test_list_comprehension():
    @ct.kernel
    def kernel(y):
        tup = (1, 2, 3)
        s1 = ct.static_eval(sum([i*i for i in tup]))
        s2 = cuda.tile.static_eval(sum([i*i for i in tup]))
        ct.scatter(y, 0, s1)
        ct.scatter(y, 1, s2)

    y = torch.zeros((2,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == [1*1 + 2*2 + 3*3, 1*1 + 2*2 + 3*3]


def test_mixed_tuple():
    @ct.kernel
    def kernel(y):
        t = ct.gather(y, ())
        tup = (1, t, 3)
        s = ct.static_eval(tup[1])
        ct.scatter(y, (), s + 5)

    y = torch.ones((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.item() == 6


def test_return_dynamic_tile():
    @ct.kernel
    def kernel(x, n: ct.Constant):
        a = ct.gather(x, 0) + 10
        b = ct.gather(x, 1) + 20
        t = ct.static_eval(a if n == 0 else b)
        ct.scatter(x, (2,), t + 100)

    x = torch.zeros((3,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 0))
    assert x[2] == 110

    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 1))
    assert x[2] == 120


def test_symbolic_tile():
    type_string = []

    @ct.kernel
    def kernel(x):
        tile = ct.load(x, (0, 0), (4, 8))
        shape = ct.static_eval(tile.shape)
        dtype = ct.static_eval(tile.dtype)
        ct.static_eval(type_string.append(repr(tile)))
        ct.scatter(x, (0, 0), shape[0])
        ct.scatter(x, (0, 1), shape[1].astype(dtype))

    x = torch.zeros((10, 10), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x[0, 0] == 4
    assert x[0, 1] == 8

    assert type_string == ["<tile[int32, (4, 8)]>"]


def test_symbolic_array():
    type_string = []

    @ct.kernel
    def kernel(x):
        shape = ct.static_eval(x.shape)
        dtype = ct.static_eval(x.dtype)
        ct.static_eval(type_string.append(repr(x)))
        ct.scatter(x, (0, 0), shape[0])
        ct.scatter(x, (0, 1), shape[1].astype(dtype))

    x = torch.zeros((10, 20), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x[0, 0] == 10
    assert x[0, 1] == 20

    assert type_string == ["<array[int32, (?, ?)]>"]


def global_func(x):
    return x + 1


def test_global_func():
    @ct.kernel
    def kernel(x):
        t = ct.load(x, (0, 0), (4, 8))
        f = global_func
        v = ct.static_eval(f(t.shape[0]))
        ct.scatter(x, (0, 0), v)

    x = torch.zeros((10, 20), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x[0, 0] == 5


def test_closure():
    @ct.kernel
    def kernel():
        def f(n):
            return n + 1
        ct.static_eval(f(3))

    with pytest.raises(TileStaticEvalError,
                       match=re.escape("Tile functions cannot be called inside static_eval()")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_static_eval_inside_closure():
    @ct.kernel
    def kernel(x):
        def f(n):
            return ct.static_eval(x.ndim + n)

        v = f(20)
        ct.scatter(x, 0, v)

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 21


def test_static_eval_error_when_called_indirectly():
    @ct.kernel
    def kernel_indirect(y):
        f = ct.static_eval
        v = f(1 * 2)
        ct.scatter(y, (), v)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileSyntaxError, match=re.escape("static_eval() must be used directly")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel_indirect, (y,))


def test_static_eval_error_when_calling_tile_func():
    @ct.kernel
    def kernel(y):
        v = ct.static_eval(ct.ones((4,), dtype=ct.int32).shape[0])
        ct.scatter(y, (), v)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticEvalError,
                       match=re.escape("Tile functions cannot be called inside static_eval()")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))


def test_nested_static_eval():
    @ct.kernel
    def kernel(x):
        v = ct.static_eval(ct.static_eval(20))
        ct.scatter(y, (), v)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticEvalError,
                       match=re.escape("static_eval() cannot be used inside static_eval().")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))


def test_exception_raised_inside_static_eval():
    @ct.kernel
    def kernel(n: ct.Constant):
        ct.static_eval(1 // n)

    with pytest.raises(TileStaticEvalError,
                       match=re.escape("Exception was raised inside static_eval()"
                                       " (ZeroDivisionError: integer division or modulo by zero)")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (0,))


def test_prohibit_walrus():
    @ct.kernel
    def kernel(x):
        ct.static_eval((y := x))  # noqa: F841

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileSyntaxError,
                       match=re.escape("static_eval() expression attempted"
                                       " to modify a local variable 'y'")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_too_many_args():
    @ct.kernel
    def kernel(x):
        ct.static_eval(3, 5)

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileSyntaxError,
                       match=re.escape("static_eval() expects a single expression")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
