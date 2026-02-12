# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest
import torch.cuda

import cuda.tile
import cuda.tile as ct


# A function that can't be evaluated with basic constant folding
def factorial(n):
    ret = 1
    for i in range(1, n + 1):
        ret *= i
    return ret


def test_static_assert_without_message():
    @ct.kernel
    def kernel(x, n: ct.Constant):
        ct.static_assert(factorial(n) < 100)
        ct.scatter(x, (), n)

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 4))
    assert x.item() == 4

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticAssertionError, match=re.escape("Static assertion failed\n")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 7))


def test_static_assert_cuda_tile_spelling():
    @ct.kernel
    def kernel(x, n: ct.Constant):
        cuda.tile.static_assert(factorial(n) < 100)
        ct.scatter(x, (), n)

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 4))
    assert x.item() == 4

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticAssertionError, match=re.escape("Static assertion failed\n")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 7))


def test_static_assert_with_fstring_message():
    @ct.kernel
    def kernel(x, n: ct.Constant):
        ct.static_assert(factorial(n) < 100,
                         f"{n}! = {factorial(n)}, that's too much. And by the way, x is {x}")
        ct.scatter(x, (), n)

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 4))
    assert x.item() == 4

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticAssertionError,
                       match=re.escape("Static assertion failed: 7! = 5040, that's too much."
                                       " And by the way, x is <array[int32, ()]>\n")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 7))


def test_static_assert_empty_string_message():
    @ct.kernel
    def kernel(x, n: ct.Constant):
        ct.static_assert(False, "")
        ct.scatter(x, (), n)

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticAssertionError, match=re.escape("Static assertion failed\n")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 7))


def test_static_assert_proxy_message():
    @ct.kernel
    def kernel(x, n: ct.Constant):
        ct.static_assert(False, x)
        ct.scatter(x, (), n)

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticAssertionError,
                       match=re.escape("Static assertion failed: <array[int32, ()]>\n")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 7))


def test_static_assert_error_when_called_indirectly():
    @ct.kernel
    def kernel_indirect(y):
        f = ct.static_assert
        v = f(1 * 2)
        ct.scatter(y, (), v)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileSyntaxError,
                       match=re.escape("static_assert() must be used directly")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel_indirect, (y,))


def test_static_assert_error_when_condition_is_not_bool():
    @ct.kernel
    def kernel(y, n: ct.Constant):
        ct.static_assert(n)
        ct.scatter(y, (), n)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileTypeError,
                       match=re.escape("static_assert() condition must be a boolean")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y, 42))


def test_static_assert_error_when_condition_is_not_constant():
    @ct.kernel
    def kernel(y, n):
        cond = n > 2
        ct.static_assert(cond)
        ct.scatter(y, (), n)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileTypeError,
                       match=re.escape("static_assert() condition must be"
                                       " a compile-time constant")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y, 42))


def test_static_assert_error_when_calling_tile_func():
    @ct.kernel
    def kernel(y):
        v = ct.static_assert(ct.ones((4,), dtype=ct.int32).dtype == ct.int32)
        ct.scatter(y, (), v)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticEvalError,
                       match=re.escape("Tile functions cannot be called inside static_assert()")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))


def test_static_assert_too_many_args():
    @ct.kernel
    def kernel():
        ct.static_assert(True, "message", 123)

    with pytest.raises(ct.TileSyntaxError,
                       match=re.escape("static_assert(cond, msg=None, /)"
                                       " expects one or two positional arguments")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_static_assert_inside_static_eval():
    @ct.kernel
    def kernel(x):
        v = ct.static_eval(ct.static_assert(20))
        ct.scatter(y, (), v)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileStaticEvalError,
                       match=re.escape("static_assert() cannot be used inside static_eval().")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
