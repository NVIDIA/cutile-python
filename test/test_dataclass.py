# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import re
from dataclasses import dataclass
from typing import Any

import pytest

import cuda.tile as ct
import torch

from cuda.tile import TileTypeError
from cuda.tile._cext import cconv_v3_enabled
from cuda.tile._exception import TypeCheckingError
from cuda.tile._execution import static_def


@dataclass(frozen=True)
class FooBar:
    foo: int
    bar: int
    baz: Any = 5


def test_basic_dataclass():
    @ct.kernel
    def kern(x):
        fb = FooBar(2, bar=7)
        ct.scatter(x, 0, fb.foo)
        ct.scatter(x, 1, fb.bar)
        ct.scatter(x, 2, fb.baz)

    x = torch.zeros((3,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [2, 7, 5]


def test_nested_dataclass():
    @ct.kernel
    def kern(x):
        fb = FooBar(2, bar=7, baz=FooBar(30, 40))
        ct.scatter(x, 0, fb.foo)
        ct.scatter(x, 1, fb.bar)
        ct.scatter(x, 2, fb.baz.foo)
        ct.scatter(x, 3, fb.baz.bar)
        ct.scatter(x, 4, fb.baz.baz)

    x = torch.zeros((5,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [2, 7, 30, 40, 5]


def test_dataclass_global_capture():
    fb = FooBar(2, 7)

    @ct.kernel
    def kern(x):
        ct.scatter(x, (), fb.foo)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.item() == 2


def test_dataclass_with_field_named_self():
    @dataclass(frozen=True)
    class Selfish:
        self: int

    @ct.kernel
    def kern(x):
        s = Selfish(12)
        ct.scatter(x, (), s.self)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.item() == 12


def test_dataclass_static_eval_roundtrip_nonconstant():
    @ct.kernel
    def kern(x):
        v = ct.bid(0) + 10
        fb = FooBar(v, bar=7)
        fb2 = ct.static_eval(fb)
        ct.scatter(x, 0, fb2.foo)
        ct.scatter(x, 1, fb2.bar)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [10, 7]


def test_dataclass_static_eval_roundtrip_constant():
    @ct.kernel
    def kern(x):
        fb = FooBar(10, bar=7)
        fb2 = ct.static_eval(fb)
        ct.scatter(x, 0, fb2.foo)
        ct.scatter(x, 1, fb2.bar)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [10, 7]


def test_dataclass_static_eval_swap_fields():
    @ct.kernel
    def kern(x):
        fb = FooBar(10, 12)
        fb2 = ct.static_eval(FooBar(fb.bar, fb.foo))
        ct.scatter(x, 0, fb2.foo)
        ct.scatter(x, 1, fb2.bar)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [12, 10]


def test_dataclasses_replace():
    @ct.kernel
    def kern(x):
        fb = FooBar(2, 7, 13)
        fb2 = dataclasses.replace(fb, baz=123, bar=(30, 40))
        ct.scatter(x, 0, fb.foo)
        ct.scatter(x, 1, fb.bar)
        ct.scatter(x, 2, fb.baz)
        ct.scatter(x, 3, fb2.foo)
        ct.scatter(x, 4, fb2.bar[0])
        ct.scatter(x, 5, fb2.bar[1])
        ct.scatter(x, 6, fb2.baz)

    x = torch.zeros((7,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [2, 7, 13, 2, 30, 40, 123]


def test_dataclasses_replace_calls_post_init():
    @dataclass(frozen=True)
    class PostInit:
        arr: ct.Array
        index: int
        value: int

        def __post_init__(self):
            ct.scatter(self.arr, self.index, self.value)

    @ct.kernel
    def kern(x):
        obj = PostInit(x, 0, 10)
        obj2 = dataclasses.replace(obj, index=1, value=30)
        ct.static_assert(obj2.index == 1)
        ct.static_assert(obj2.value == 30)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [10, 30]


def test_loop_carried_dataclass_reconstructed_with_field_info():
    @ct.kernel
    def kern(x, n):
        fb = FooBar(1, 10, 100)
        for i in range(n):
            fb = dataclasses.replace(fb, foo=fb.foo + 1, bar=fb.bar + i)
        ct.scatter(x, 0, fb.foo)
        ct.scatter(x, 1, fb.bar)
        ct.scatter(x, 2, fb.baz)

    x = torch.zeros((3,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x, 3))
    assert x.tolist() == [4, 13, 100]


def test_empty_dataclass():
    @dataclass(frozen=True)
    class EmptyFoo:
        val = 5

    @ct.kernel
    def kern(x):
        ef = EmptyFoo()
        ct.scatter(x, 0, ef.val)

    x = torch.zeros((3,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [5, 0, 0]


def test_empty_dataclass_no_fields():
    @dataclass(frozen=True)
    class EmptyFoo:
        ...

    @ct.kernel
    def kern():
        EmptyFoo()
    ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_reject_empty_custom_init_in_dataclass():
    @dataclass(frozen=True)
    class EmptyFoo:
        def __init__(self):
            pass

    @ct.kernel
    def kern():
        EmptyFoo()

    expected_message = re.escape(
        "Dataclass instance creation is only supported for dataclasses with a default generated"
        " __init__() method"
    )
    with pytest.raises(TileTypeError, match=expected_message):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_user_defined_methods_and_constants():
    @dataclass(frozen=True)
    class WithMethod:
        x: int
        y: int

        NUMBER = 123

        def foo(self):
            return self.x * 10 + self.y

    @ct.kernel
    def kern(x, y, z):
        fb = WithMethod(ct.bid(0) + 5, 7)
        ct.scatter(x, ct.bid(0), fb.foo())
        ct.scatter(y, ct.bid(0), fb.NUMBER)
        ct.scatter(z, ct.bid(0), WithMethod.NUMBER)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    y = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    z = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (2,), kern, (x, y, z))
    assert x.tolist() == [57, 67]
    assert y.tolist() == [123, 123]
    assert z.tolist() == [123, 123]


def test_user_defined_property():
    @dataclass(frozen=True)
    class WithProperty:
        x: int
        y: int

        @property
        def foo(self):
            return self.x * 10 + self.y

    @ct.kernel
    def kern(x):
        fb = WithProperty(ct.bid(0) + 5, 7)
        ct.scatter(x, ct.bid(0), fb.foo)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (2,), kern, (x,))
    assert x.tolist() == [57, 67]


def test_dataclasses_replace_no_such_field():
    @ct.kernel
    def kern():
        fb = FooBar(2, 7)
        dataclasses.replace(fb, abracadabra=8)

    with pytest.raises(TileTypeError, match="Dataclass 'FooBar' has no such field 'abracadabra'"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_reject_nonfrozen():
    @dataclass
    class Thawed:
        foo: int

    @ct.kernel
    def kern():
        Thawed(2)

    with pytest.raises(TileTypeError, match="Only frozen dataclasses are supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_reject_nonfrozen_returned_from_static_eval():
    @dataclass
    class Thawed:
        foo: int

    @ct.kernel
    def kern():
        ct.static_eval(Thawed(2))

    with pytest.raises(TileTypeError, match="Only frozen dataclasses are supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_reject_no_init():
    @dataclass(frozen=True, init=False)
    class Initless:
        foo: int

    @ct.kernel
    def kern():
        Initless(2)

    expected_message = re.escape(
        "Dataclass instance creation is only supported for dataclasses with a default generated"
        " __init__() method"
    )
    with pytest.raises(TileTypeError, match=expected_message):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_reject_custom_init():
    @dataclass(frozen=True)
    class CustomizedInit:
        foo: int

        def __init__(self):
            pass

    @ct.kernel
    def kern():
        CustomizedInit()

    expected_message = re.escape(
        "Dataclass instance creation is only supported for dataclasses with a default generated"
        " __init__() method"
    )
    with pytest.raises(TileTypeError, match=expected_message):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_post_init():
    @dataclass(frozen=True)
    class PostInit:
        foo: int
        bar: ct.Array

        def __post_init__(self):
            ct.scatter(self.bar, (), self.foo)

    @ct.kernel
    def kern(x):
        PostInit(3, x)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.item() == 3


def test_reject_custom_new():
    @dataclass(frozen=True)
    class CustomizedNew:
        foo: int

        def __new__(self):
            pass

    @ct.kernel
    def kern():
        CustomizedNew()

    with pytest.raises(TileTypeError,
                       match="Dataclasses with custom __new__ are not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_dataclass_with_base():
    @dataclass(frozen=True)
    class Base:
        x: int

    @dataclass(frozen=True)
    class Derived(Base):
        y: int

    @ct.kernel
    def kern(x):
        d = Derived(3, 5)
        ct.scatter(x, 0, d.x)
        ct.scatter(x, 1, d.y)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [3, 5]


def test_reject_base_with_custom_new():
    @dataclass(frozen=True)
    class Base:
        bar: int

        def __new__(cls):
            pass

    @dataclass(frozen=True)
    class Derived(Base):
        foo: int

    @ct.kernel
    def kern():
        Derived(3)

    with pytest.raises(TileTypeError, match="Dataclasses with custom __new__ are not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_reject_dataclass_base_nondataclass_derived():
    @dataclass(frozen=True)
    class Base:
        foo: int

    class Derived(Base):
        pass

    @ct.kernel
    def kern():
        Derived(3)

    with pytest.raises(TileTypeError,
                       match="Non-dataclass subclasses of a dataclass are not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_reject_no_init_field():
    @dataclass(frozen=True)
    class Initless:
        foo: int
        bar: int = dataclasses.field(init=False)

    @ct.kernel
    def kern():
        Initless(2)

    with pytest.raises(TileTypeError, match="Dataclasses with init=False fields are not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_reject_default_factory_field():
    @dataclass(frozen=True)
    class Initless:
        foo: int
        bar: int = dataclasses.field(default_factory=lambda: 5)

    @ct.kernel
    def kern():
        Initless(2)

    with pytest.raises(TileTypeError,
                       match="Dataclasses with default_factory fields are not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_init_static_def():
    @dataclass(frozen=True)
    class MetaInit:
        x: int
        y: int

        @static_def
        def __init__(self, n):
            object.__setattr__(self, "x", n * 5)
            object.__setattr__(self, "y", n * 7)

    @ct.kernel
    def kern(x):
        d = MetaInit(10)
        ct.scatter(x, 0, d.x)
        ct.scatter(x, 1, d.y)

    x = torch.zeros(2, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [50, 70]


def test_call_dunder():
    @dataclass(frozen=True)
    class WithCall:
        x: int
        y: int

        def __call__(self, z):
            return 100 * self.x + 10 * self.y + z

    @ct.kernel
    def kern(x):
        d = WithCall(3, 5)
        val = d(7)
        ct.scatter(x, (), val)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.item() == 357


def test_call_dunder_static_def():
    @dataclass(frozen=True)
    class WithCallStaticDef:
        x: int
        y: int

        @static_def
        def __call__(self, z):
            items = [self.x, self.y, z]
            res = 0
            while items:
                res = res * 10 + items.pop()
            return res

    @ct.kernel
    def kern(x):
        d = WithCallStaticDef(3, 5)
        val = d(7)
        ct.scatter(x, (), val)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.item() == 753


def test_call_dunder_base_class():
    @dataclass(frozen=True)
    class WithCallBase:
        x: int

        def __call__(self, z):
            return 100 * self.x + 10 * self.y + z

    @dataclass(frozen=True)
    class Derived(WithCallBase):
        y: int

    @ct.kernel
    def kern(x):
        d = Derived(3, 5)
        val = d(7)
        ct.scatter(x, (), val)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.item() == 357


def test_call_dunder_base_class_shadowed():
    @dataclass(frozen=True)
    class WithCallBase:
        x: int

        def __call__(self, z):
            ct.static_assert(False)
            return -1

    @dataclass(frozen=True)
    class WithCallDerived(WithCallBase):
        y: int

        def __call__(self, z):
            return 100 * self.x + 10 * self.y + z

    @ct.kernel
    def kern(x):
        d = WithCallDerived(3, 5)
        val = d(7)
        ct.scatter(x, (), val)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.item() == 357


def test_reject_call_no_dunder():
    @dataclass(frozen=True)
    class NoCall:
        x: int
        y: int

    @ct.kernel
    def kern(x):
        d = NoCall(3, 5)
        d(7)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    with pytest.raises(TypeCheckingError, match="Cannot call an object of type NoCall"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))


def test_getitem_dunder():
    @dataclass(frozen=True)
    class WithGetitem:
        x: int

        def __getitem__(self, i):
            return self.x * i

    @ct.kernel
    def kern(x):
        d = WithGetitem(3)
        res = d[7]
        ct.scatter(x, (), res)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.item() == 21


def test_setitem_dunder():
    @dataclass(frozen=True)
    class WithSetitem:
        x: ct.Array

        def __setitem__(self, i, val):
            ct.scatter(self.x, i, val * 10)

    @ct.kernel
    def kern(x):
        d = WithSetitem(x)
        d[1] = 5

    x = torch.arange(4, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [0, 50, 2, 3]


def test_binary_arithmetic_dunders():
    num_ops = 14
    [
        ADD,       SUB,    MUL,    MATMUL, TRUEDIV,
        FLOORDIV,  MOD,    DIVMOD, POW,    LSHIFT,
        RSHIFT,    AND,    XOR,    OR
    ] = range(num_ops)

    class LhsDundersMixin:
        def __add__(self, other): return ADD, 0, self.x, other.x
        def __sub__(self, other): return SUB, 0, self.x, other.x
        def __mul__(self, other): return MUL, 0, self.x, other.x
        def __matmul__(self, other): return MATMUL, 0, self.x, other.x
        def __truediv__(self, other): return TRUEDIV, 0, self.x, other.x
        def __floordiv__(self, other): return FLOORDIV, 0, self.x, other.x
        def __mod__(self, other): return MOD, 0, self.x, other.x
        def __divmod__(self, other): return DIVMOD, 0, self.x, other.x
        def __pow__(self, other): return POW, 0, self.x, other.x
        def __lshift__(self, other): return LSHIFT, 0, self.x, other.x
        def __rshift__(self, other): return RSHIFT, 0, self.x, other.x
        def __and__(self, other): return AND, 0, self.x, other.x
        def __xor__(self, other): return XOR, 0, self.x, other.x
        def __or__(self, other): return OR, 0, self.x, other.x

    class RhsDundersMixin:
        def __radd__(self, other): return ADD, 1, other.x, self.x
        def __rsub__(self, other): return SUB, 1, other.x, self.x
        def __rmul__(self, other): return MUL, 1, other.x, self.x
        def __rmatmul__(self, other): return MATMUL, 1, other.x, self.x
        def __rtruediv__(self, other): return TRUEDIV, 1, other.x, self.x
        def __rfloordiv__(self, other): return FLOORDIV, 1, other.x, self.x
        def __rmod__(self, other): return MOD, 1, other.x, self.x
        def __rdivmod__(self, other): return DIVMOD, 1, other.x, self.x
        def __rpow__(self, other): return POW, 1, other.x, self.x
        def __rlshift__(self, other): return LSHIFT, 1, other.x, self.x
        def __rrshift__(self, other): return RSHIFT, 1, other.x, self.x
        def __rand__(self, other): return AND, 1, other.x, self.x
        def __rxor__(self, other): return XOR, 1, other.x, self.x
        def __ror__(self, other): return OR, 1, other.x, self.x

    @dataclass(frozen=True)
    class WithLhsDunders(LhsDundersMixin):
        x: int

    @dataclass(frozen=True)
    class WithRhsDunders(RhsDundersMixin):
        x: int

    @dataclass(frozen=True)
    class WithBothDunders(LhsDundersMixin, RhsDundersMixin):
        x: int

    lhs_only_a = WithLhsDunders(1000)
    lhs_only_b = WithLhsDunders(2000)
    rhs_only_a = WithRhsDunders(3000)
    rhs_only_b = WithRhsDunders(4000)
    both_a = WithBothDunders(5000)
    both_b = WithBothDunders(6000)

    a_options = (lhs_only_a, rhs_only_a, both_a)
    b_options = (lhs_only_b, rhs_only_b, both_b)

    @ct.kernel
    def kern(out):
        def put(op_idx, tup):
            for j, val in ct.static_iter(enumerate(tup)):
                ct.scatter(out, (ai, bi, op_idx, j), val)

        for ai, a in ct.static_iter(enumerate(a_options)):
            for bi, b in ct.static_iter(enumerate(b_options)):
                if not ct.static_eval(isinstance(a, WithRhsDunders)
                                      and isinstance(b, WithLhsDunders)):
                    put(ADD, a + b)
                    put(SUB, a - b)
                    put(MUL, a * b)
                    put(MATMUL, a @ b)
                    put(TRUEDIV, a / b)
                    put(FLOORDIV, a // b)
                    put(MOD, a % b)
                    put(DIVMOD, divmod(a, b))
                    put(POW, a ** b)
                    put(LSHIFT, a << b)
                    put(RSHIFT, a >> b)
                    put(AND, a & b)
                    put(XOR, a ^ b)
                    put(OR, a | b)

    out = torch.zeros((3, 3, num_ops, 4), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (out,))
    res = out.tolist()
    for ai, a in enumerate(a_options):
        for bi, b in enumerate(b_options):
            for op_idx in range(num_ops):
                r = res[ai][bi][op_idx]
                if isinstance(a, WithRhsDunders) and isinstance(b, WithLhsDunders):
                    assert r == [0, 0, 0, 0]
                else:
                    assert r == [op_idx, isinstance(a, WithRhsDunders), a.x, b.x]

    # Manual sanity check
    assert res[0][0][MUL] == [MUL, 0, lhs_only_a.x, lhs_only_b.x]
    assert res[1][1][MATMUL] == [MATMUL, 1, rhs_only_a.x, rhs_only_b.x]


def test_binary_arithmetic_dunder_absent():
    @dataclass(frozen=True)
    class Foo:
        x: int

    @ct.kernel
    def kern():
        Foo(4) + 5

    with pytest.raises(ct.TileTypeError, match=re.escape("Unsupported operand types for +:")):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_binary_arithmetic_dunder_absent_both_dataclasses():
    @dataclass(frozen=True)
    class Foo:
        x: int

    @ct.kernel
    def kern():
        Foo(4) + Foo(5)

    with pytest.raises(ct.TileTypeError, match=re.escape("Unsupported operand types for +:")):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_binary_arithmetic_dunder_conditionally_implemented():
    @dataclass(frozen=True)
    class Foo:
        x: int

        def __add__(self, other):
            if other.dtype == ct.int32:
                return self.x + other
            else:
                return NotImplemented

    @ct.kernel
    def kern(rhs, out):
        ct.scatter(out, (), Foo(4) + rhs)

    x = torch.zeros((), dtype=torch.int32, device="cuda")
    with pytest.raises(ct.TileTypeError, match=re.escape("Unsupported operand types for +:")):
        ct.launch(torch.cuda.current_stream(), (1,), kern, (12.0, x))

    ct.launch(torch.cuda.current_stream(), (1,), kern, (15, x))
    assert x.item() == 19


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_instance_as_kernel_arg():
    @dataclass(frozen=True)
    class KernelArg:
        x: Any
        idx: Any
        val: float

    @ct.kernel
    def kern(d):
        ct.scatter(d.x, d.idx, d.val)

    x = torch.zeros((3, 3), dtype=torch.float32, device="cuda:0")
    d = KernelArg(x=x, idx=(1, 2), val=5)
    ct.launch(torch.cuda.current_stream(), (1,), kern, (d,))
    assert x.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_instance_as_constant_kernel_arg():
    @dataclass(frozen=True)
    class ConstKernelArg:
        idx: Any
        val: float

    @ct.kernel
    def kern(x, d: ct.Constant):
        ct.scatter(x, d.idx, d.val)

    x = torch.zeros((3, 3), dtype=torch.float32, device="cuda:0")
    d = ConstKernelArg(idx=(1, 2), val=5)
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x, d))
    assert x.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_reject_nonfrozen_kernel_arg():
    @dataclass
    class NonFrozenArg:
        val: int

    @ct.kernel
    def kern(x, d):
        ct.scatter(x, (), d.val)

    x = torch.zeros((), dtype=torch.int32, device="cuda:0")
    with pytest.raises(TileTypeError, match="Only frozen dataclasses are supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, (x, NonFrozenArg(3)))


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_tuple_of_dataclasses_as_kernel_arg():
    @dataclass(frozen=True)
    class Arg:
        a: int
        b: float

    @ct.kernel
    def kern(x, t):
        ct.scatter(x, 0, t[0].a * 10 + t[0].b)
        ct.scatter(x, 1, t[1].a * 10 + t[1].b)

    x = torch.zeros((2,), dtype=torch.float32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x, (Arg(1, 2.5), Arg(3, 4.5))))
    assert x.tolist() == [12.5, 34.5]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_tuple_of_dataclass_and_array_as_kernel_arg():
    @dataclass(frozen=True)
    class Arg:
        a: int
        b: float

    @ct.kernel
    def kern(t):
        item, out = t
        ct.scatter(out, (), item.a * 10 + item.b)

    x = torch.zeros((), dtype=torch.float32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, ((Arg(3, 0.5), x),))
    assert x.item() == 30.5


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_with_tuple_component_as_kernel_arg():
    @dataclass(frozen=True)
    class Arg:
        idx: Any
        n: int

    @ct.kernel
    def kern(x, d):
        ct.scatter(x, d.idx, d.n)

    x = torch.zeros((3, 3), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x, Arg(idx=(1, 2), n=7)))
    assert x.tolist() == [[0, 0, 0], [0, 0, 7], [0, 0, 0]]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_with_list_of_arrays_as_kernel_arg():
    @dataclass(frozen=True)
    class Arg:
        arrays: Any
        val: float

    @ct.kernel
    def kern(d):
        ct.scatter(d.arrays[0], (), d.val)
        ct.scatter(d.arrays[1], (), d.val * 2)

    x = torch.zeros((), dtype=torch.float32, device="cuda:0")
    y = torch.zeros((), dtype=torch.float32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (Arg([x, y], 2.5),))
    assert x.item() == 2.5
    assert y.item() == 5.0


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_with_dataclass_components_as_kernel_arg():
    # Two sibling dataclasses of *different* types, expanded in the same (nested) round.
    @dataclass(frozen=True)
    class Inner1:
        a: int
        b: float

    @dataclass(frozen=True)
    class Inner2:
        x: float
        y: int
        z: int

    @dataclass(frozen=True)
    class Outer:
        inner1: Any
        inner2: Any
        out: Any

    @ct.kernel
    def kern(d):
        ct.scatter(d.out, 0, d.inner1.a * 10 + d.inner1.b)
        ct.scatter(d.out, 1, d.inner2.x + d.inner2.y * 100 + d.inner2.z * 1000)

    x = torch.zeros((2,), dtype=torch.float32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern,
              (Outer(Inner1(3, 0.5), Inner2(0.25, 4, 5), x),))
    assert x.tolist() == [30.5, 5400.25]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_two_top_level_dataclass_args():
    @dataclass(frozen=True)
    class First:
        a: int
        b: float

    @dataclass(frozen=True)
    class Second:
        x: float
        y: int
        z: int

    @ct.kernel
    def kern(out, d1, d2):
        ct.scatter(out, 0, d1.a * 10 + d1.b)
        ct.scatter(out, 1, d2.x + d2.y * 100 + d2.z * 1000)

    x = torch.zeros((2,), dtype=torch.float32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kern,
              (x, First(3, 0.5), Second(0.25, 4, 5)))
    assert x.tolist() == [30.5, 5400.25]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_unsupported_type_in_dataclass_field():
    @ct.kernel
    def kern(x):
        print(x)

    d = FooBar(foo=123, bar=iter([]))
    with pytest.raises(TypeError, match="Invalid field 'bar' of kernel argument #0:"
                                        " Objects of type 'list_iterator' are not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, (d,))
