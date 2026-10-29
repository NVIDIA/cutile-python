# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch.cuda

import cuda.tile as ct
from cuda.tile import TileTypeError


def test_variadic_kwargs_in_helper_function():
    def helper(**kwargs):
        ct.static_assert(kwargs == {"foo": 123, "bar": 456})
        return 789

    @ct.kernel
    def kernel():
        res = helper(foo=123, bar=456)
        ct.static_assert(res == 789)

    ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_forward_variadic_kwargs():
    def leaf(x, foo, bar):
        return x * 100 + foo * 10 + bar

    def forward(f, **kwargs):
        return f(3, **kwargs)

    @ct.kernel
    def kernel():
        res = forward(leaf, foo=4, bar=5)
        ct.static_assert(res == 345)

    ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_dict_access():
    def helper(**kwargs):
        foo1 = kwargs["foo"]
        ct.static_assert(foo1 == 123)

        foo2 = kwargs.get("foo")
        ct.static_assert(foo2 == 123)

        bar1 = kwargs["bar"]
        ct.static_assert(bar1 == 456)

        bar2 = kwargs.get("bar")
        ct.static_assert(bar2 == 456)

        qux1 = kwargs.get("qux")
        ct.static_assert(qux1 is None)

        res1 = "foo" in kwargs
        ct.static_assert(res1)

        res2 = "foo" not in kwargs
        ct.static_assert(not res2)

        res3 = "qux" in kwargs
        ct.static_assert(not res3)

        res4 = "qux" not in kwargs
        ct.static_assert(res4)

        return 789

    @ct.kernel
    def kernel():
        res = helper(foo=123, bar=456)
        ct.static_assert(res == 789)

    ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_dict_getitem_miss():
    def helper(**kwargs):
        return kwargs["qux"]

    @ct.kernel
    def kernel():
        helper(foo=123, bar=456)

    with pytest.raises(TileTypeError, match="Key 'qux' not found in dictionary"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_duplicate_keyword_from_explicit_and_variadic_kwargs():
    def leaf(x, foo=0):
        return x + foo

    def forward(**kwargs):
        return leaf(1, foo=2, **kwargs)

    @ct.kernel
    def kernel():
        forward(foo=3)

    with pytest.raises(TileTypeError, match="Got multiple values for keyword argument 'foo'"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_duplicate_keyword_from_two_variadic_kwargs():
    def leaf(foo):
        return foo

    def make_dict(**kwargs):
        return kwargs

    def forward(kwargs1, kwargs2):
        return leaf(**kwargs1, **kwargs2)

    @ct.kernel
    def kernel():
        forward(make_dict(foo=1), make_dict(foo=2))

    with pytest.raises(TileTypeError, match="Got multiple values for keyword argument 'foo'"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
