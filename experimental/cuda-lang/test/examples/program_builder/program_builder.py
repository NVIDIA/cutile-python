# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar
from typing import Callable, Generic
from abc import abstractmethod
from enum import Enum, auto
import cuda.lang as cl
from dataclasses import dataclass, fields, replace
import pprint


Context = TypeVar("Context")


class VisitorIterate(Enum):
    CONTINUE = auto()
    STOP = auto()


@dataclass
class Visitor:
    @abstractmethod
    def __call__(self, node: "ProgramFragment") -> VisitorIterate: ...


@dataclass(frozen=True)
class ProgramFragment(Generic[Context]):
    @abstractmethod
    def __call__(self, context: Context) -> Context: ...

    def __str__(self):
        return pprint.pformat(self, indent=2, width=60)

    @property
    def children(self) -> tuple["ProgramFragment[Context]"]:
        return tuple(
            child for child in self.attributes if isinstance(child, ProgramFragment)
        )

    @property
    def attributes(self):
        return tuple(getattr(self, field.name) for field in fields(self))

    def visit(self, f: Visitor):
        if f(self) is VisitorIterate.STOP:
            return
        for child in self.children:
            child.visit(f)


@dataclass(frozen=True)
class ProgN(ProgramFragment):
    body: tuple

    def __call__(self, context):
        for expr in cl.static_iter(self.body):
            context = expr(context)
        return context

    def visit(self, f):
        if f(self) is VisitorIterate.STOP:
            return
        for expression in self.body:
            expression.visit(f)


@dataclass(frozen=True)
class If(ProgramFragment):
    condition: ProgramFragment
    then: ProgramFragment
    else_: ProgramFragment | None = None

    def __call__(self, context):
        if self.condition(context):
            context = self.then(context)
        elif self.else_ is not None:
            context = self.else_(context)
        return context


@dataclass(frozen=True)
class Loop(ProgramFragment):
    condition: ProgramFragment
    body: ProgramFragment

    def __call__(self, context):
        while self.condition(context):
            context = self.body(context)
        return context


@dataclass(frozen=True)
class ForN(ProgramFragment):
    """Program fragment for a loop over a possibly dynamic value.

    This is fragile because we don't do variable renaming like with hygenic
    lisp macros, but we could pass around scopes on the context to make nested
    loops less fragile.
    """

    get_n: ProgramFragment
    body: ProgramFragment

    def __call__(self, context):
        n = self.get_n(context)
        outer_iv = context.iv
        for iv in range(n):
            context = replace(context, iv=iv)
            context = self.body(context)
        return replace(context, iv=outer_iv)


@dataclass(frozen=True)
class ForStaticN(ProgramFragment):
    n: int
    body: ProgramFragment

    def __call__(self, context):
        for iv in range(self.n):
            context = replace(context, iv=iv)
            context = self.body(context)
        return context


@dataclass(frozen=True)
class Call(ProgramFragment):
    function: Callable

    def __call__(self, context):
        return self.function(context)

    def visit(self, f):
        f(self)
