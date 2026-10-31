# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from cuda.tile import TileValueError


@ct.kernel
def propagate_constant_int_then_promote(n: ct.Constant[int], out):
    # Do a bunch of operations that should propagate the constant
    #                  n = 5:     n = 50:
    #                  ========   ========
    t = (n + 2) * 3  # t = 21     t = 156
    t = -t           # t = -21    t = -156
    t ^= n           # t = -18    t = -170

    # Now combine it with a concretely typed tile to trigger the type promotion logic
    a = ct.arange(4, dtype=ct.int8)
    b = a - t  # b = [18, 19, 20, 21]

    # Check that the type of `b` is the same as `a`
    ct.static_assert(b.dtype == ct.int8)

    ct.scatter(out, ct.arange(4, dtype=ct.int32), b)


def test_propagate_constant_int_then_promote():
    a = torch.zeros((4,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), propagate_constant_int_then_promote, (5, a))
    assert a.tolist() == [18, 19, 20, 21]


def test_propagate_constant_int_then_promote_out_of_range():
    a = torch.zeros((4,), dtype=torch.int32, device="cuda:0")
    with pytest.raises(TileValueError, match="Value -170 is out of range for int8"):
        ct.launch(torch.cuda.current_stream(), (1,), propagate_constant_int_then_promote, (50, a))


@ct.kernel
def propagate_constant_float_then_promote(n: ct.Constant[int], out):
    # Do a bunch of operations that should propagate the constant
    t = (n + 2) * 3  # n = 5  -> t = 21
    t = -t   # t = -21
    t += 0.5  # t = -20.5

    # Now combine it with a concretely typed tile to trigger the type promotion logic
    a = ct.arange(4, dtype=ct.int8)
    b = a - t  # b = [20.5, 21.5, 22.5, 23.5]

    # Check that the result of `a - t` has been promoted to float32
    ct.static_assert(b.dtype == ct.float32)

    ct.scatter(out, ct.arange(4, dtype=ct.int32), b)


def test_propagate_constant_float_then_promote():
    a = torch.zeros((4,), dtype=torch.float32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), propagate_constant_float_then_promote, (5, a))
    assert a.tolist() == [20.5, 21.5, 22.5, 23.5]


@ct.kernel
def pack_tuple_then_getitem_and_promote(n: ct.Constant[int], out):
    tup = (ct.bid(0), n // 2)
    t = tup[1]

    # Combine `t` with a concretely typed tile to trigger the type promotion logic
    a = ct.arange(4, dtype=ct.int8)
    b = a + t

    # Check that the type of `b` is the same as `a`
    ct.static_assert(b.dtype == ct.int8)

    ct.scatter(out, ct.arange(4, dtype=ct.int32), b)


def test_pack_tuple_then_getitem_and_promote():
    a = torch.zeros((4,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), pack_tuple_then_getitem_and_promote, (11, a))
    assert a.tolist() == [5, 6, 7, 8]


@ct.kernel
def pack_nested_tuple_then_getitem_and_promote(n: ct.Constant[int], out):
    tup = (ct.bid(0), (n // 2, ct.bid(0)))
    t = tup[1][0]

    # Combine `t` with a concretely typed tile to trigger the type promotion logic
    a = ct.arange(4, dtype=ct.int8)
    b = a + t

    # Check that the type of `b` is the same as `a`
    ct.static_assert(b.dtype == ct.int8)

    ct.scatter(out, ct.arange(4, dtype=ct.int32), b)


def test_pack_nested_tuple_then_getitem_and_promote():
    a = torch.zeros((4,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), pack_nested_tuple_then_getitem_and_promote,
              (11, a))
    assert a.tolist() == [5, 6, 7, 8]


@ct.kernel
def propagate_constant_int_through_if_else_then_promote(n: ct.Constant[int], out):
    if ct.bid(0) == 0:
        t = n + 2
    else:
        t = 7  # same constant (assuming n = 5)

    # Now combine it with a concretely typed tile to trigger the type promotion logic
    a = ct.arange(4, dtype=ct.int8)
    b = a + t  # [7, 8, 9, 10]

    # Check that the type of `b` is the same as `a`
    ct.static_assert(b.dtype == ct.int8)

    ct.scatter(out, ct.arange(4, dtype=ct.int32), b)


def test_propagate_constant_int_through_if_else_then_promote():
    a = torch.zeros((4,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,),
              propagate_constant_int_through_if_else_then_promote, (5, a))
    assert a.tolist() == [7, 8, 9, 10]


@ct.kernel
def different_constants_in_if_else_then_promote(n: ct.Constant[int], out):
    if ct.bid(0) == 0:
        t = n
    else:
        t = 7  # different constant (assuming n = 5)

    # Now combine it with a concretely typed tile to trigger the type promotion logic
    a = ct.arange(4, dtype=ct.int8)
    b = a + t  # [5, 6, 7, 8]

    # Since `t` is int32 and not loosely typed, the result should be an int32
    ct.static_assert(b.dtype == ct.int32)

    ct.scatter(out, ct.arange(4, dtype=ct.int32), b)


def test_different_constants_in_if_else_then_promote():
    a = torch.zeros((4,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,),
              different_constants_in_if_else_then_promote, (5, a))
    assert a.tolist() == [5, 6, 7, 8]


@ct.kernel
def combine_loose_and_strict_int(n: ct.Constant[int], out):
    t = n + ct.int16(2)  # int16 because n is loosely typed
    ct.static_assert(t == 7)
    a = ct.arange(4, dtype=ct.int8)  # explicitly int8
    b = a + t  # int16 because `t` is strictly typed
    ct.static_eval(b.dtype == ct.int16)
    ct.scatter(out, ct.arange(4, dtype=ct.int32), b)


def test_combine_loose_and_strict_int():
    a = torch.zeros((4,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), combine_loose_and_strict_int, (5, a))
    assert a.tolist() == [7, 8, 9, 10]


@ct.kernel
def call_float_and_store(out):
    c1 = float("2.0")
    # Since out is f16, this would fail if `c` is a strictly typed f32:
    ct.scatter(out, 0, c1)

    # Calling float() on a strictly typed constant should produce a loosely typed one:
    c2 = float(ct.float32(5.0))
    ct.scatter(out, 1, c2)


def test_float_constructor_produces_loosely_typed_constant():
    a = torch.zeros((2,), dtype=torch.float16, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), call_float_and_store, (a,))
    assert a.tolist() == [2.0, 5.0]
