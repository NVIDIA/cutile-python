# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import cuda.tile as ct
import re

from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._compile import compile_tile
from cuda.tile._exception import TileTypeError


def nd_tensor(nd: int, dtype=None):
    return torch.rand((4,) * nd, dtype=dtype, device='cuda')


def test_tuple_static_getitem_int():

    def kernel():
        t = (2, 2)
        s1, s2 = t
        ct.arange(s1, dtype=ct.int32)
        ct.arange(s2, dtype=ct.int32)

    compile_tile(kernel, (), CompilerOptions())


def test_tuple_static_getitem_slice():

    def kernel():
        t = (1, 1, 2, 2)
        s1, s2 = t[::2]
        ct.arange(s1, dtype=ct.int32)
        ct.arange(s2, dtype=ct.int32)

    compile_tile(kernel, (), CompilerOptions())


def test_tuple_getitem_non_static():

    def kernel(i: int):
        t = (1, 1, 2, 2)
        s1 = t[i]
        ct.arange(s1, dtype=ct.int32)

    with pytest.raises(TileTypeError, match=r"Expected an integer constant"):
        compile_tile(kernel, ((0,)), CompilerOptions())


def test_tuple_getitem_unsupported_key():

    def kernel():
        t = (1, 2, 3, 4)
        t[(0, 1)]

    with pytest.raises(TileTypeError, match=r"Expected an integer constant"):
        compile_tile(kernel, (), CompilerOptions())


def test_tile_attr():

    def kernel():
        val = 0.1
        tx = ct.full((2, 2), val, dtype=ct.float32)
        shape = tx.shape
        dtype = tx.dtype
        ndim = tx.ndim
        ct.full(shape, ndim, dtype=dtype)

    compile_tile(kernel, (), CompilerOptions())


def test_compare_dtype():

    def kernel(x):
        if x.dtype == ct.float64:
            val = 1
        else:
            val = 2
        ct.full((2, 2), val, dtype=ct.float32)

    compile_tile(kernel, (nd_tensor(2),), CompilerOptions())


def test_none_as_constant():

    def kernel():
        x = None
        y = 1
        if x is y:
            ct.printf('done')

    compile_tile(kernel, (), CompilerOptions())


def test_is_op_on_constant():

    def kernel():
        None is None
        None is ct.bid(0)
        ct.bid(0) is None

    compile_tile(kernel, (), CompilerOptions())


def test_is_op_on_none_constant():

    def kernel():
        tx = ct.full((1,), 0, ct.float32)
        ty = ct.full((1,), 0, ct.float32)
        tx is ty

    msg = re.escape("Operator 'is' expects one of the operands to be None")
    with pytest.raises(TileTypeError, match=msg):
        compile_tile(kernel, (), CompilerOptions())


def test_fold_if_expr():

    def kernel(x):
        dtype = ct.float32 if x.dtype == ct.float32 else ct.float16
        ct.full((1,), 0, dtype=dtype)

    x = nd_tensor(1, dtype=torch.float32)
    compile_tile(kernel, (x,), CompilerOptions())


def test_fold_if_stmt():

    def kernel():
        if True:
            shape = (1, 1)
            dtype = ct.float32
        else:
            shape = (2, 2)
            dtype = ct.float64
        ct.full(shape, 0, dtype=dtype)

    compile_tile(kernel, (), CompilerOptions())


def test_fold_if_break_in_loop():

    def kernel():
        while True:
            if False:
                sz = 1
                break
            else:
                sz = 2
                break
        ct.full((sz, sz), 1.0, dtype=ct.float32)

    compile_tile(kernel, (), CompilerOptions())


def plus_one(x):
    return x + 1


def test_fold_if_calling_function():

    def kernel():
        if True:
            sz = plus_one(1)
            sz = 1
        else:
            sz = plus_one(2)
            sz = 2
        ct.full((sz, sz), 1.0, dtype=ct.float32)

    compile_tile(kernel, (), CompilerOptions())


def plus_two(x):
    y = plus_one(x)
    z = plus_one(y)
    return z


def test_fold_if_calling_function_with_function_call():

    def kernel():
        if True:
            sz = plus_one(1)
            sz = 1
        else:
            sz = plus_two(2)
            sz = 2
        ct.full((sz, sz), 1.0, dtype=ct.float32)

    compile_tile(kernel, (), CompilerOptions())


def test_dtype_in_for_loop():

    def kernel():
        for i in range(5):
            dtype = ct.float16
        ct.full((1, 1), 1.0, dtype=dtype)

    compile_tile(kernel, (), CompilerOptions())
