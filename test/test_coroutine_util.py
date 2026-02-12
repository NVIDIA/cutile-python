# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._coroutine_util import resume_after, run_coroutine
import pytest
import traceback


async def series(n):
    if n == 0:
        return 0
    r = await resume_after(series(n - 1))
    return r + n


def test_run_coroutine():
    n = 10000
    res = run_coroutine(series(n))
    assert res == sum(range(n + 1))


async def raise_if_zero(n):
    if n == 0:
        raise ValueError("Hello")
    await resume_after(raise_if_zero(n - 1))


def test_propagate_exception():
    with pytest.raises(ValueError, match="Hello"):
        run_coroutine(raise_if_zero(5))


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
    res = run_coroutine(raise_then_catch(4))
    assert res == 100 + 2 + 3 + 4


async def two_calls():
    t1 = await resume_after(series(3))
    t2 = await resume_after(series(4))
    return t1, t2


def test_return_values():
    res = run_coroutine(two_calls())
    assert res == (1 + 2 + 3, 1 + 2 + 3 + 4)


async def raise_in_leaf():
    raise ValueError("leaf")


async def call_leaf():
    await resume_after(raise_in_leaf())


def test_traceback_preserved():
    try:
        run_coroutine(call_leaf())
    except ValueError as e:
        frame_names = [f.name for f in traceback.extract_tb(e.__traceback__)]
        assert "raise_in_leaf" in frame_names
