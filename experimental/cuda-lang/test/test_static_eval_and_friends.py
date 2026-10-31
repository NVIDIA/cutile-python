# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest
import torch

import cuda.lang
import cuda.lang as cl
from cuda.lang import static_eval
from cuda.tile import TileStaticAssertionError, StaticException
from test.util import compile_kernel


def test_cl_static_eval():
    @cl.kernel
    def kern(a):
        a[()] = cl.static_eval([2*3].pop())

    a = torch.zeros((), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.item() == 6


def test_cuda_lang_static_eval():
    @cl.kernel
    def kern(a):
        a[()] = cuda.lang.static_eval([2*3].pop())

    a = torch.zeros((), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.item() == 6


def test_imported_static_eval():
    @cl.kernel
    def kern(a):
        a[()] = static_eval([2*3].pop())

    a = torch.zeros((), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.item() == 6


def test_static_assert():
    @cl.kernel
    def kern():
        cl.static_assert(False, "Boo")

    with pytest.raises(TileStaticAssertionError, match="Static assertion failed: Boo"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, ())


def test_static_iter():
    @cl.kernel
    def kern(a):
        for i, x in cl.static_iter(enumerate([10, 20])):
            a[i] = x

    a = torch.zeros(2, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.tolist() == [10, 20]


def test_static_eval_shift_operators():
    @cl.kernel
    def kern(x, y):
        value = x[0]
        y[0] = cl.static_eval(value << 3)
        y[1] = cl.static_eval(value >> 2)

    x = torch.tensor([-16], dtype=torch.int32, device="cuda:0")
    y = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x, y,))
    assert y.tolist() == [-128, -4]


def test_static_eval_pointer_arithmetic():
    @cl.kernel
    def kern(a):
        p = a.pointer()
        p2 = cl.static_eval(p + 3)
        p2[0] = 5
        p3 = cl.static_eval(p2 - 1)
        p3[0] = 7
        tid = cl.thread_index(0) + 1
        p4 = cl.static_eval(tid + p)
        p4[0] = 9
        p5 = cl.static_eval(0 + p)
        p5[0] = 11

    a = torch.zeros((4,), dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a,))
    assert a.tolist() == [11, 9, 7, 5]


def test_static_exception():
    @cl.kernel
    def kern():
        raise cl.static_exception(ValueError("Hello"))

    with pytest.raises(StaticException,
                       match=re.escape("Exception was raised at compile time (ValueError: Hello)")):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, ())


def test_static_eval_dispatch_binop_via_rhs():
    class Custom:
        def __init__(self, val):
            self.val = val

        def __radd__(self, other):
            return other + self.val

    @cl.kernel
    def kern(x):
        tid = cl.thread_index(0)
        x[tid] = cl.static_eval(tid + Custom(5))

        vec = cl.Vector(tid + 1, tid + 2)
        new_vec = cl.static_eval(vec + Custom(50))
        x[4 + tid] = new_vec[0]
        x[8 + tid] = new_vec[1]

    x = torch.zeros(12, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (4,), kern, (x,))
    assert x.tolist() == [5, 6, 7, 8, 51, 52, 53, 54, 52, 53, 54, 55]


def test_static_iter_in_static_eval():
    z = []

    @cl.static_def
    def foo(x):
        for i in cl.static_iter(x):
            print(i)
            z.append(i)

    tup = (1, 2, 3, 4, 5)

    @cl.kernel()
    def kernel():
        foo(tup)

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())
    assert z == list(tup)


def test_strictly_typed_constant_roundtrip():
    def helper(x):
        assert isinstance(x, cl.Scalar)
        assert x.dtype == cl.int16
        return x + 3

    @cl.kernel
    def kern():
        x = cl.int16(5)
        y = cl.static_eval(helper(x))
        cl.static_assert(y.dtype == cl.int16)
        cl.static_assert(y == 8)

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, ())


def test_wrapping_arithmetic_in_static_eval():
    def check_scalar(x):
        assert isinstance(x, cl.Scalar)
        assert x.dtype == cl.uint8
        return x

    @cl.kernel
    def kern(inp, out):
        a = cl.uint8(150)
        b = cl.static_eval(check_scalar(cl.wrapping_add(a, 110)))
        cl.static_assert(b == 4)
        out[0] = b

        value = inp[0]
        out[1] = cl.static_eval(check_scalar(cl.wrapping_sub(value, 1)))
        out[2] = cl.static_eval(check_scalar(cl.wrapping_mul(value, b)))

        v = inp.pointer().load(count=2)
        out.pointer(4).store(cl.static_eval(cl.wrapping_add(v, a)))

    inp = torch.tensor([0, 100], dtype=torch.uint8, device="cuda:0")
    out = torch.zeros(6, dtype=torch.uint8, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (inp, out))
    assert out.tolist() == [4, 255, 0, 0, 150, 250]


def test_static_eval_pointer_comparison():
    def kern():
        p = cl.shared_array(2, cl.int32).pointer()
        q = p + cl.thread_index(0)
        eq = cl.static_eval(p == q)
        cl.static_assert(isinstance(eq, cl.Scalar) and eq.dtype == cl.bool_)

    compile_kernel(kern)


def test_static_eval_pow_and_mod():
    @cl.kernel
    def kern(a, n, out):
        x = a[0]
        y = a[1]
        k = n[0]
        out[0] = cl.static_eval(x ** y)
        out[1] = cl.static_eval(x ** k)
        out[2] = cl.static_eval(x % y)
        v = a.pointer().load(count=2)
        out.pointer(4).store(cl.static_eval(v % 3.0))

    a = torch.tensor([-1.5, 2.0], dtype=torch.float32, device="cuda")
    n = torch.tensor([3], dtype=torch.int32, device="cuda")
    out = torch.zeros(6, dtype=torch.float32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (a, n, out))
    assert out.tolist() == [2.25, -3.375, 0.5, 0.0, 1.5, 2.0]


def test_static_eval_integer_pow():
    def kern():
        x = cl.shared_array(1, cl.int32)[0]
        cl.static_eval(x ** 2)

    compile_kernel(kern, raises=pytest.raises(
        StaticException, match="Exponentiation of non-constant integers is not supported"))


def test_static_eval_opaque_value():
    def kern():
        value = cl.shared_array(1, cl.mbarrier)[0]
        cl.static_assert(isinstance(value, cl.OpaqueValue))
        cl.static_assert(value.dtype == cl.mbarrier)
        cl.static_assert(repr(value) == "<opaque_value[mbarrier]>")
        same = cl.static_eval(value)
        cl.static_assert(cl.dtype_of(same) == cl.mbarrier)

    compile_kernel(kern)
