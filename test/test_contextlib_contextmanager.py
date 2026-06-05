# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

import cuda.tile as ct
from cuda.tile import static_assert
from cuda.tile._exception import TileSyntaxError, TileTypeError


def append(x, val):
    i = ct.gather(x, 0) + 1
    ct.scatter(x, 0, i)
    ct.scatter(x, i, val)


@contextmanager
def foo(x, start_val, end_val):
    append(x, start_val)
    yield start_val
    append(x, end_val)


def test_user_defined_context_manager():
    @ct.kernel
    def kern(x):
        append(x, 10)
        with foo(x, 20, 30) as val:
            append(x, 40)
            static_assert(val == 20)
        append(x, 50)

    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [5, 10, 20, 40, 30, 50, 0, 0, 0, 0]


def test_nested_with():
    @ct.kernel
    def kern(x):
        append(x, 10)
        with foo(x, 20, 30) as val:
            static_assert(val == 20)
            append(x, 40)
            with foo(x, 50, 60) as val2:
                static_assert(val2 == 50)
                append(x, 70)
            append(x, 80)
        append(x, 90)

    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [9, 10, 20, 40, 50, 70, 60, 80, 30, 90]


def test_break_in_with():
    @ct.kernel
    def kern(x):
        i = 0
        while True:
            with foo(x, 100 + i, 200 + i):
                append(x, i)
                if i == 2:
                    break
            i += 1

    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [9, 100, 0, 200, 101, 1, 201, 102, 2, 202]


@contextmanager
def wrap_foo(x, a, b, c, d, e, f):
    append(x, a)
    with foo(x, b, e) as res:
        append(x, c)
        yield res + 1
        append(x, d)
    append(x, f)


def test_enter_context_inside_context_manager():
    @ct.kernel
    def kern(x):
        append(x, 10)
        with wrap_foo(x, 20, 30, 40, 60, 70, 80) as r:
            append(x, 50)
            static_assert(r == 31)
        append(x, 90)

    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [9, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_break_in_with_nested_context():
    @ct.kernel
    def kern(x):
        i = 0
        while True:
            with wrap_foo(x, 10 + i, 20 + i, 30 + i, 40 + i, 50 + i, 60 + i):
                append(x, i)
                if i == 1:
                    break
            i += 1

    x = torch.zeros(20, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [14,
                          10, 20, 30, 0, 40, 50, 60,   # i=0
                          11, 21, 31, 1, 41, 51, 61,   # i=1
                          0, 0, 0, 0, 0]


@contextmanager
def enter_context_in_cleanup(x, a, b, c, d, e):
    append(x, a)
    yield
    append(x, b)
    with foo(x, c, e):
        append(x, d)


def test_enter_context_in_cleanup():
    @ct.kernel
    def kern(x):
        append(x, 10)
        i = 0
        while True:
            with enter_context_in_cleanup(x, 20 + i, 40 + i, 50 + i, 60 + i, 70 + i):
                append(x, 30 + i)
                if i == 1:
                    break
            i += 1
        append(x, 80)

    x = torch.zeros(16, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [14, 10,
                          20, 30, 40, 50, 60, 70,  # i=0
                          21, 31, 41, 51, 61, 71,  # i=1
                          80, 0]


@contextmanager
def yield_closure(x, a, b, c):
    def func():
        append(x, a)
    append(x, b)
    yield func
    a += 1
    append(x, c)


def test_yield_closure():
    @ct.kernel
    def kern(x):
        with yield_closure(x, 10, 20, 30) as func:
            func()
        func()

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [4, 20, 10, 30, 11, 0, 0, 0]


def test_yield_closure_break():
    @ct.kernel
    def kern(x):
        i = 0
        while True:
            with yield_closure(x, 10 + i * 100, 20 + i * 100, 30 + i * 100) as func:
                func()
                if i == 2:
                    break
            func()
            i += 1

    x = torch.zeros(15, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [11,
                          20, 10, 30, 11,  # i=0
                          120, 110, 130, 111,   # i=1
                          220, 210, 230,  # i=2
                          0, 0, 0]


def test_nested_break_cleanup_order():
    @ct.kernel
    def kern(x):
        i = 0
        while True:
            with foo(x, 1, 2):
                with foo(x, 3, 4):
                    append(x, 9)
                    if ct.bid(0) == 0:
                        break
            i += 1

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [5, 1, 3, 9, 4, 2, 0, 0]


def test_nested_continue_cleanup_order():
    @ct.kernel
    def kern(x):
        i = 0
        while i < 1:
            with foo(x, 1, 2):
                with foo(x, 3, 4):
                    append(x, 9)
                    i += 1
                    continue

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [5, 1, 3, 9, 4, 2, 0, 0]


def test_nested_return_cleanup_order():
    @ct.kernel
    def kern(x):
        with foo(x, 1, 2):
            with foo(x, 3, 4):
                append(x, 9)
                return

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [5, 1, 3, 9, 4, 2, 0, 0]


def test_break_with_outer_context_outside_loop():
    @ct.kernel
    def kern(x):
        with foo(x, 1, 2):
            while True:
                with foo(x, 3, 4):
                    with foo(x, 5, 6):
                        append(x, 9)
                        break
            append(x, 7)

    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [8, 1, 3, 5, 9, 6, 4, 7, 2, 0]


@pytest.mark.parametrize("return_early", [False, True])
def test_yield_closure_helper_function(return_early):
    def helper(x, return_early):
        with yield_closure(x, 10, 20, 30) as func:
            func()
            if return_early:
                return func
        return func

    @ct.kernel
    def kern(x, return_early):
        func = helper(x, return_early)
        func()

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x, return_early))
    assert x.tolist() == [4, 20, 10, 30, 11, 0, 0, 0]


@contextmanager
def two_yields(x):
    append(x, 1)
    yield
    append(x, 2)
    yield
    append(x, 3)


def test_multiple_yields_diagnostic():
    @ct.kernel
    def kern(x):
        with two_yields(x):
            append(x, 99)

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(TileTypeError, match="must have one `yield` statement"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))


@contextmanager
def no_yield(x):
    append(x, 1)


def test_no_yield_diagnostic():
    @ct.kernel
    def kern(x):
        with no_yield(x):
            append(x, 99)

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(TileTypeError, match="no reachable `yield` statement"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))


@contextmanager
def yield_in_if(x, flag):
    append(x, 1)
    if flag:
        yield
    append(x, 2)


def test_yield_in_control_flow_diagnostic():
    @ct.kernel
    def kern(x):
        with yield_in_if(x, True):
            append(x, 99)

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(TileTypeError, match="outside of loops and conditional"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))


@contextmanager
def dead_yield_after_break(x):
    append(x, 1)
    yield
    while True:
        append(x, 3)
        break
        yield  # dead: unreachable after break, must not count as a second yield


def test_dead_yield_after_break():
    @ct.kernel
    def kern(x):
        with dead_yield_after_break(x):
            append(x, 9)

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [3, 1, 9, 3, 0, 0, 0, 0]


@contextmanager
def dead_yield_after_continue(x):
    append(x, 1)
    yield
    i = 0
    while i < 2:
        append(x, 10 + i)
        i += 1
        continue
        yield


def test_dead_yield_after_continue():
    @ct.kernel
    def kern(x):
        with dead_yield_after_continue(x):
            append(x, 9)

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [4, 1, 9, 10, 11, 0, 0, 0]


@contextmanager
def only_dead_yield(x):
    append(x, 1)
    while True:
        append(x, 2)
        break
        yield


def test_only_dead_yield():
    @ct.kernel
    def kern(x):
        with only_dead_yield(x):
            append(x, 9)

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(TileTypeError, match="no reachable `yield` statement"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))


@contextmanager
def with_return():
    if ct.bid(0) == 1:
        return
    yield


def test_disallow_return():
    @ct.kernel
    def kern():
        with with_return():
            pass

    with pytest.raises(TileSyntaxError, match="Returning from a generator-based context manager"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


@dataclass(frozen=True)
class DataclassWithCtxMethod:
    x: ct.Array

    @contextmanager
    def foo(self, start_val, end_val):
        append(self.x, start_val)
        yield start_val
        append(self.x, end_val)


def test_dataclass_method_context_manager():
    @ct.kernel
    def kern(x):
        append(x, 10)
        obj = DataclassWithCtxMethod(x)
        with obj.foo(20, 30) as val:
            append(x, 40)
            static_assert(val == 20)
        append(x, 50)

    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [5, 10, 20, 40, 30, 50, 0, 0, 0, 0]


def test_generator_context_manager_type_str():
    @ct.kernel
    def kern():
        cm = foo(None, 10, end_val=20)
        cm + 1  # trigger a type error

    msg = (re.escape("Unsupported operand types for +: GeneratorContextManager[<function foo at 0x")
           + "[0-9a-fA-F]+"
           + re.escape(">, args=[None, Tile[int32,()]],"
                       " kwargs={end_val: Tile[int32,()]}] and Tile[int32,()]"))
    with pytest.raises(TileTypeError, match=msg):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())
