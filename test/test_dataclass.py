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

    x = torch.zeros((3,), dtype=torch.int32, device="cuda")
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

    x = torch.zeros((5,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [2, 7, 30, 40, 5]


def test_dataclass_global_capture():
    fb = FooBar(2, 7)

    @ct.kernel
    def kern(x):
        ct.scatter(x, (), fb.foo)

    x = torch.zeros((), dtype=torch.int32, device="cuda")
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

    x = torch.zeros((), dtype=torch.int32, device="cuda")
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

    x = torch.zeros((2,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [10, 7]


def test_dataclass_static_eval_roundtrip_constant():
    @ct.kernel
    def kern(x):
        fb = FooBar(10, bar=7)
        fb2 = ct.static_eval(fb)
        ct.scatter(x, 0, fb2.foo)
        ct.scatter(x, 1, fb2.bar)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [10, 7]


def test_dataclass_static_eval_swap_fields():
    @ct.kernel
    def kern(x):
        fb = FooBar(10, 12)
        fb2 = ct.static_eval(FooBar(fb.bar, fb.foo))
        ct.scatter(x, 0, fb2.foo)
        ct.scatter(x, 1, fb2.bar)

    x = torch.zeros((2,), dtype=torch.int32, device="cuda")
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

    x = torch.zeros((7,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [2, 7, 13, 2, 30, 40, 123]


def test_loop_carried_dataclass_reconstructed_with_field_info():
    @ct.kernel
    def kern(x, n):
        fb = FooBar(1, 10, 100)
        for i in range(n):
            fb = dataclasses.replace(fb, foo=fb.foo + 1, bar=fb.bar + i)
        ct.scatter(x, 0, fb.foo)
        ct.scatter(x, 1, fb.bar)
        ct.scatter(x, 2, fb.baz)

    x = torch.zeros((3,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x, 3))
    assert x.tolist() == [4, 13, 100]


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

    x = torch.zeros((2,), dtype=torch.int32, device="cuda")
    y = torch.zeros((2,), dtype=torch.int32, device="cuda")
    z = torch.zeros((2,), dtype=torch.int32, device="cuda")
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

    x = torch.zeros((2,), dtype=torch.int32, device="cuda")
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


def test_reject_post_init():
    @dataclass(frozen=True)
    class PostInit:
        foo: int

        def __post_init__(self):
            pass

    @ct.kernel
    def kern():
        PostInit(3)

    with pytest.raises(TileTypeError,
                       match="Dataclasses with __post_init__ are not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


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

    x = torch.zeros((2,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [3, 5]


def test_reject_nondataclass_base():
    class Base:
        pass

    @dataclass(frozen=True)
    class Derived(Base):
        foo: int

    @ct.kernel
    def kern():
        Derived(3)

    with pytest.raises(TileTypeError,
                       match="Dataclasses with non-dataclass base are not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


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

    x = torch.zeros(2, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
    assert x.tolist() == [50, 70]
