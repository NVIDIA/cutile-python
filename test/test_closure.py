# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import cuda.tile as ct


int_alias = int


def test_nested_function_with_annotation():
    @ct.kernel
    def kernel(x):
        def foo(t: int_alias) -> int_alias:
            return t + 20
        val = ct.gather(x, ())
        val2 = foo(val)
        ct.scatter(x, (), val2)

    x = torch.ones((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 21


def test_pure_nested_function():
    @ct.kernel
    def kernel(x):
        def foo(t):
            return t + 20
        val = ct.gather(x, ())
        val2 = foo(val)
        ct.scatter(x, (), val2)

    x = torch.ones((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 21


def test_pure_nested_function_shadowed_name():
    @ct.kernel
    def kernel(x):
        def foo(x):
            return x + 20
        val = ct.gather(x, ())
        val2 = foo(val)
        ct.scatter(x, (), val2)

    x = torch.ones((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 21


def test_simple_closure():
    @ct.kernel
    def kernel(x, n):
        def foo(t):
            return t + n
        val = ct.gather(x, 0)
        val2 = foo(val)
        ct.scatter(x, 0, val2)
        n = 100
        val3 = foo(val)
        ct.scatter(x, 1, val3)

    x = torch.ones((2,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 15))
    assert x.tolist() == [16, 101]


def test_simple_frozen_capture():
    @ct.kernel
    def kernel(x):
        def make_closure(t):
            def f(y):
                return y + t
            return f

        c = make_closure(30)
        val = ct.gather(x, ())
        val2 = c(val)
        ct.scatter(x, (), val2)

    x = torch.ones((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 31


def test_frozen_capture_returned_via_tuple():
    @ct.kernel
    def kernel(x):
        def make_closure(t):
            def f(y):
                return y + t
            return f, "dummy"

        c = make_closure(30)[0]
        val = ct.gather(x, ())
        val2 = c(val)
        ct.scatter(x, (), val2)

    x = torch.ones((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 31


def test_frozen_capture_type_compatibility():
    @ct.kernel
    def kernel(x):
        def make_closure(t):
            def f(y):
                return y + t
            return f

        i = ct.bid(0)
        if i == 0:
            c = make_closure(30)
        else:
            c = make_closure(40)

        val = ct.gather(x, i)
        val2 = c(val)
        ct.scatter(x, i, val2)

    x = torch.ones((2,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (2,), kernel, (x,))
    assert x.tolist() == [31, 41]


def test_frozen_captures_at_multiple_depths():
    @ct.kernel
    def kernel(x):
        def f0(x0):
            def f1(x1):
                def f2(x2):
                    def f3(x3):
                        def f4(x4):
                            ct.scatter(x, 0, x0)
                            ct.scatter(x, 1, x1)
                            ct.scatter(x, 2, x2)
                            ct.scatter(x, 3, x3)
                            ct.scatter(x, 4, x4)
                        return f4
                    g4 = f3(30)
                    g4(40)
                return f2
            g2 = f1(10)
            g2(20)

        f0(0)

    x = torch.ones((5,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [0, 10, 20, 30, 40]


def test_frozen_capture_that_itself_needs_freezing():
    @ct.kernel
    def kernel(x):
        def make_closure(t):
            def g(y):
                return y * 2 + t

            def f(y):
                return g(y) + 100

            return f

        c = make_closure(30)
        val = ct.gather(x, ())
        val2 = c(val)
        ct.scatter(x, (), val2)

    x = torch.ones((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 132


def test_closure_simple_default_args():
    @ct.kernel
    def kernel(x, y):
        def func(t, a=10, b=3):
            return t * a + b

        tx = ct.gather(x, ())
        y0 = func(tx)
        ct.scatter(y, 0, y0)
        y1 = func(tx, b=5)
        ct.scatter(y, 1, y1)
        y2 = func(tx, 4)
        ct.scatter(y, 2, y2)

    x = torch.ones((), dtype=torch.int32, device="cuda")
    y = torch.zeros((3,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert y.tolist() == [10 + 3, 10 + 5, 4 + 3]


def test_frozen_capture_non_const_value():
    @ct.kernel
    def kernel(x, y):
        def make_closure(t):
            def f(val):
                return val + t
            return f

        tx = ct.gather(x, ())
        c = make_closure(tx)
        result = c(5)
        ct.scatter(y, (), result)

    x = torch.ones((), dtype=torch.int32, device="cuda")
    y = torch.zeros((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert y.item() == 6


def test_closure_nonconst_default_arg():
    def make_closure(default):
        def f(x, i, t=default):
            ct.scatter(x, i, t)
        default = -1
        return f

    @ct.kernel
    def kernel(x):
        if ct.bid(0) == 0:
            c = make_closure(3)
        else:
            c = make_closure(4)
        c(x, ct.bid(0))

    x = torch.zeros((2,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (2,), kernel, (x,))
    assert x.tolist() == [3, 4]


def test_lambda():
    @ct.kernel
    def kernel(x, y):
        f = lambda t, m=100: t * 2 + n + m  # noqa: E731
        tx = ct.gather(x, ())
        n = 5
        ct.scatter(y, 0, f(tx))
        n = 7
        ct.scatter(y, 1, f(tx))
        f2 = lambda t: t * 10 + n  # noqa: E731
        ct.scatter(y, 2, f2(tx))

    x = torch.ones((), dtype=torch.int32, device="cuda")
    y = torch.zeros((3,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert y.tolist() == [2 + 5 + 100, 2 + 7 + 100, 10 + 7]
