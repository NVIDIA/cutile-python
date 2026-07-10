# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang.compilation import KernelSignature
from cuda.lang._exception import StaticEvalError
from test.util import compile_kernel, make_symbolic_scalar
import cuda.lang as cl

import pytest


def test_symbolic_pointer():
    def checks(sym):
        assert sym.pointee_dtype is cl.int32
        assert sym.pointer_dtype.bitwidth == 64
        assert not sym.opaque
        assert sym.memory_space is cl.MemorySpace.GENERIC
        assert sym.pointer_dtype == cl.pointer_dtype(cl.int32, cl.MemorySpace.GENERIC)
        assert str(sym) == "<pointer[int32]>"

    def k():
        with cl.local_array(1, cl.int32) as arr:
            ptr = arr.get_base_pointer()
            cl.static_eval(checks(ptr))

    compile_kernel(k)


def test_symbolic_array():
    def checks(sym):
        assert sym.dtype is cl.int32
        assert sym.ndim == 1
        assert sym.shape == (4,)
        assert sym.strides == (1,)
        assert str(sym) == "<array[int32, (4)]>"

    def k():
        with cl.local_array(4, cl.int32) as arr:
            cl.static_eval(checks(arr))

    compile_kernel(k)


def test_symbolic_vector():
    def checks(sym):
        assert sym.element_dtype is cl.int32
        assert sym.element_count == 4
        assert str(sym) == "<vector[int32, count=4]>"

    def k():
        with cl.local_array(4, cl.int32) as arr:
            vector = arr.load_element(0, count=4)
            cl.static_eval(checks(vector))

    compile_kernel(k)


def test_symbolic_scalar():
    def checks(sym):
        assert sym.dtype == cl.int32
        assert str(sym) == "<scalar[int32]>"

    def k(dynamic_value):
        cl.static_eval(checks(dynamic_value))

    compile_kernel(k, signature=KernelSignature([make_symbolic_scalar(cl.int32)]))


def test_symbolic_scalar_getitem():
    def checks(sym):
        sym[0]

    def k(dynamic_value):
        cl.static_eval(checks(dynamic_value))

    compile_kernel(
        k,
        signature=KernelSignature([make_symbolic_scalar(cl.int32)]),
        raises=pytest.raises(
            StaticEvalError, match="'SymbolicScalar' object is not subscriptable"
        ),
    )


def test_static_assertion():
    message = "expected int32"

    def checks(sym):
        assert sym.dtype is cl.int32, message

    def k(dynamic_value):
        cl.static_eval(checks(dynamic_value))

    compile_kernel(
        k,
        signature=KernelSignature([make_symbolic_scalar(cl.int64)]),
        raises=pytest.raises(StaticEvalError, match=message),
    )
