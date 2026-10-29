# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._cext import hir_run
from cuda.tile._coroutine_util import resume_after
import pytest
import traceback


async def series(n):
    if n == 0:
        return 0
    r = await resume_after(series(n - 1))
    return r + n


def test_run_coroutine():
    n = 10000
    res = hir_run(series(n))
    assert res == sum(range(n + 1))


async def raise_if_zero(n):
    if n == 0:
        raise ValueError("Hello")
    await resume_after(raise_if_zero(n - 1))


def test_propagate_exception():
    with pytest.raises(ValueError, match="Hello"):
        hir_run(raise_if_zero(5))


async def raise_then_catch(n):
    if n == 0:
        raise ValueError("Hello")

    if n == 1:
        try:
            await resume_after(raise_then_catch(0))
        except ValueError as e:
            assert str(e) == "Hello"
            return 100
        assert False

    r = await resume_after(raise_then_catch(n - 1))
    return r + n


def test_raise_then_catch():
    res = hir_run(raise_then_catch(4))
    assert res == 100 + 2 + 3 + 4


async def return_123():
    return 123


async def raise_then_catch_and_call_another(n):
    if n == 0:
        raise ValueError("Hello")

    if n == 1:
        try:
            await resume_after(raise_then_catch_and_call_another(0))
        except ValueError as e:
            assert str(e) == "Hello"
            x = await resume_after(return_123())
            return x
        assert False

    r = await resume_after(raise_then_catch_and_call_another(n - 1))
    return r + n


def test_raise_then_catch_and_call_another():
    res = hir_run(raise_then_catch_and_call_another(4))
    assert res == 123 + 2 + 3 + 4


async def two_calls():
    t1 = await resume_after(series(3))
    t2 = await resume_after(series(4))
    return t1, t2


def test_return_values():
    res = hir_run(two_calls())
    assert res == (1 + 2 + 3, 1 + 2 + 3 + 4)


async def raise_in_leaf():
    raise ValueError("leaf")


async def call_leaf():
    await resume_after(raise_in_leaf())


def test_traceback_preserved():
    try:
        hir_run(call_leaf())
    except ValueError as e:
        traceback.print_tb(e.__traceback__)
        frame_names = [f.name for f in traceback.extract_tb(e.__traceback__)]
        assert "raise_in_leaf" in frame_names
    else:
        assert False


class WeirdAwaitable:
    def __await__(self):
        return iter([123])


async def weird_await():
    await WeirdAwaitable()


async def call_weird_await():
    await weird_await()


def test_unexpected_awaitable():
    try:
        hir_run(call_weird_await())
    except TypeError as e:
        assert "awaitable must be an EvalBlockAwaitable or a coroutine" in str(e)
        traceback.print_tb(e.__traceback__)
        frame_names = [f.name for f in traceback.extract_tb(e.__traceback__)]
        assert "call_weird_await" in frame_names
    else:
        assert False


async def resume_after_call_weird_await(flag):
    try:
        await resume_after(call_weird_await())
    finally:
        flag[0] = True


def test_cleanup_after_internal_error():
    flag = [False]
    coro = resume_after_call_weird_await(flag)
    with pytest.raises(TypeError, match="awaitable must be an EvalBlockAwaitable or a coroutine"):
        hir_run(coro)
    assert flag[0] is True
