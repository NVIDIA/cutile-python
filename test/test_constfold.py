# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO

import numpy as np
import pytest
import torch
import cuda.tile as ct
import re

from cuda.tile._cext import CallingConvention
from cuda.tile._compile import get_sm_arch
from cuda.tile._exception import TileTypeError
from util import assert_equal


def nd_tensor(nd: int, dtype=None):
    return torch.rand((4,) * nd, dtype=dtype, device='cuda')


def compile(pyfunc, pyargs):
    kernel = ct.kernel(pyfunc)
    sig = ct.compilation.KernelSignature.from_kernel_args(kernel, pyargs,
                                                          CallingConvention.cutile_python_v1())
    ct.compilation.export_kernel(kernel, [sig], BytesIO(), gpu_code=get_sm_arch(),
                                 output_format="cubin")


def test_tuple_static_getitem_int():

    def kernel():
        t = (2, 2)
        s1, s2 = t
        ct.arange(s1, dtype=ct.int32)
        ct.arange(s2, dtype=ct.int32)

    compile(kernel, ())


def test_tuple_static_getitem_slice():

    def kernel():
        t = (1, 1, 2, 2)
        s1, s2 = t[::2]
        ct.arange(s1, dtype=ct.int32)
        ct.arange(s2, dtype=ct.int32)

    compile(kernel, ())


def test_tile_attr():

    def kernel():
        val = 0.1
        tx = ct.full((2, 2), val, dtype=ct.float32)
        shape = tx.shape
        dtype = tx.dtype
        ndim = tx.ndim
        ct.full(shape, ndim, dtype=dtype)

    compile(kernel, ())


def test_compare_array_dtype():

    def kernel(x):
        if x.dtype == ct.float64:
            val = 1
        else:
            val = 2
        ct.full((2, 2), val, dtype=ct.float32)

    compile(kernel, (nd_tensor(2),))


def test_dtype_comparison():
    def kernel():
        a = ct.float32 == ct.float32
        ct.static_assert(a is True)

        b = ct.float32 != ct.float32
        ct.static_assert(b is False)

        c = ct.float32 == ct.float64
        ct.static_assert(c is False)

        d = ct.float32 != ct.float64
        ct.static_assert(d is True)

    compile(kernel, ())


def test_string_comparison():
    def kernel():
        a = "foo" == "foo"
        ct.static_assert(a is True)

        b = "foo" != "foo"
        ct.static_assert(b is False)

        c = "foo" == "bar"
        ct.static_assert(c is False)

        d = "foo" != "bar"
        ct.static_assert(d is True)

        e = "foo" < "bar"
        ct.static_assert(e is False)

        f = "foo" <= "bar"
        ct.static_assert(f is False)

        g = "foo" > "bar"
        ct.static_assert(g is True)

        h = "foo" >= "bar"
        ct.static_assert(h is True)

    compile(kernel, ())


def test_string_concat():
    def kernel():
        a = "foo"
        b = "bar"
        c = a + b
        ct.static_assert(c == "foobar")

    compile(kernel, ())


def test_int_constant_converted_to_str():
    def kernel():
        a = 123
        s = str(a)
        ct.static_assert(s == "123")

    compile(kernel, ())


def test_none_as_constant():

    def kernel():
        x = None
        y = 1
        if x is y:
            ct.printf('done')

    compile(kernel, ())


@pytest.mark.parametrize("negate", [False, True])
def test_is_or_not_op_on_none_constant(negate):

    def kernel():
        tx = ct.full((1,), 0, ct.float32)
        ty = ct.full((1,), 0, ct.float32)
        if negate:
            tx is not ty
        else:
            tx is ty

    op_name = 'is not' if negate else 'is'
    msg = re.escape(f"Operator '{op_name}' expects one of the operands to be None")
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, ())


def test_fold_if_expr():

    def kernel(x):
        dtype = ct.float32 if x.dtype == ct.float32 else ct.float16
        ct.full((1,), 0, dtype=dtype)

    x = nd_tensor(1, dtype=torch.float32)
    compile(kernel, (x,))


def test_fold_if_stmt():

    def kernel():
        if True:
            shape = (1, 1)
            dtype = ct.float32
        else:
            shape = (2, 2)
            dtype = ct.float64
        ct.full(shape, 0, dtype=dtype)

    compile(kernel, ())


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

    compile(kernel, ())


def test_fold_nested_if_both_early_terminators_in_loop():

    @ct.kernel
    def kernel(x):
        i = 0
        a = 10
        while True:
            if ct.bid(0) == i:
                a += 3
                if True:
                    break
                a = 30
            else:
                a = 20
                i += 1
                if True:
                    continue
                a = 40
        ct.scatter(x, ct.bid(0), a)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (2,), kernel, (x,))
    assert x.tolist() == [13, 23]


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

    compile(kernel, ())


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

    compile(kernel, ())


def test_dtype_in_for_loop():

    def kernel():
        dtype = ct.float16
        for i in range(5):
            dtype = ct.float16
        ct.full((1, 1), 1.0, dtype=dtype)

    compile(kernel, ())


def test_semi_constant_tuple_yielded_by_ifelse():
    @ct.kernel
    def kernel(x):
        if ct.bid(0) == 0:
            tup = (ct.bid(1), 4)
        else:
            tup = (ct.bid(0), 4)
        # Use tup[1] in a context that requires it to be constant
        tx = ct.arange(tup[1], dtype=x.dtype)
        ct.store(x, (0,), tx)

    x = torch.zeros((4,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [0, 1, 2, 3]


def test_strictly_typed_integer_constant_truncation():
    @ct.kernel
    def kernel(x, y, z):
        ct.scatter(x, 0, ct.int64(ct.uint32(-3)))
        ct.scatter(x, 1, ct.int64(ct.uint32(0x1abcdef23)))
        ct.scatter(x, 2, ct.int64(ct.int32(-3_000_000_000)))
        ct.scatter(x, 3, ct.int64(ct.int32(3_000_000_000)))

        for i in range(2000):
            ct.scatter(y, i, ct.int64(ct.int8(i - 1000)))
            ct.scatter(z, i, ct.int64(ct.uint8(i - 1000)))

    x = torch.zeros(4, dtype=torch.int64, device="cuda")
    y = torch.zeros(2000, dtype=torch.int64, device="cuda")
    z = torch.zeros(2000, dtype=torch.int64, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, z))
    assert x.tolist() == [0xfffffffd, 0xabcdef23, 1294967296, -1294967296]

    assert_equal(y, torch.arange(-1000, 1000, device="cuda").to(torch.int8).to(torch.int64))
    assert_equal(z, torch.arange(-1000, 1000, device="cuda").to(torch.uint8).to(torch.int64))


def test_strictly_typed_integer_constant_truncation_unary():
    @ct.kernel
    def kernel(x):
        ct.scatter(x, 0, ct.int64(~ct.uint32(3)))
        ct.scatter(x, 1, ct.int64(-ct.uint32(3)))

    x = torch.zeros(2, dtype=torch.int64, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [0xfffffffc, 0xfffffffd]


def test_strictly_typed_boolean_constant_truncation():
    @ct.kernel
    def kernel(x):
        ct.scatter(x, 0, ct.int64(ct.bool_(5)))
        ct.scatter(x, 1, ct.int64(ct.bool_(-3)))

    x = torch.zeros(2, dtype=torch.int64, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [1, 1]


def test_strictly_typed_integer_constant_truncation_binary():
    @ct.kernel
    def kernel(x):
        t = ct.uint8(150) + ct.uint8(110)  # 260 = 4 (mod 256)
        ct.scatter(x, 0, ct.int64(t))

    x = torch.zeros(1, dtype=torch.int64, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [4]


def test_strictly_typed_float_constant_rounding():
    @ct.kernel
    def kernel(x):
        ct.scatter(x, 0, ct.float16(0.2))
        ct.scatter(x, 1, ct.float64(ct.float8_e8m0fnu(0.2)))

    x = torch.zeros(2, dtype=torch.float64, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [float(np.float16(0.2)), 0.25]


def test_strictly_typed_unsigned_float_constant_from_negative():
    @ct.kernel
    def kernel():
        ct.float8_e8m0fnu(-0.2)

    with pytest.raises(ct.TileValueError,
                       match="Negative values cannot be represented in an unsigned float format"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_ensure_constant_ok():
    @ct.kernel
    def kernel(x, c: ct.Constant):
        c2 = ct.ensure_constant(c + 1)
        ct.scatter(x, (), c2)

    x = torch.zeros((), dtype=torch.int64, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 4))
    assert x.item() == 5


def test_ensure_constant_fail():
    @ct.kernel
    def kernel():
        ct.ensure_constant(ct.bid(0))

    with pytest.raises(ct.TileStaticAssertionError,
                       match="is not a compile-time constant"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
