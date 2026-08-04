# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
from abc import abstractmethod
import torch
import cuda.lang as cl
from dataclasses import dataclass, fields, replace
import pprint

__doc__ = """
Demonstrates the flexibility of frozen dataclasses by constructing a dynamic
program out of dataclasses on the host, analyzing the program on the host,
and executing the program on the device.
"""


@dataclass(frozen=True)
class Context:
    tensor: cl.Array
    iv: int = 0
    n: int = 0


@dataclass(frozen=True)
class AST:
    @abstractmethod
    def __call__(self, context: Context) -> Context: ...

    def __str__(self):
        return pprint.pformat(self, indent=2, width=60)

    def visit(self, f):
        f(self)
        for field in fields(self):
            attr = getattr(self, field.name)
            attr.visit(f)


@dataclass(frozen=True)
class ProgN(AST):
    body: tuple

    def __call__(self, context):
        for expr in cl.static_iter(self.body):
            context = expr(context)
        return context

    def visit(self, f):
        f(self)
        for expression in self.body:
            expression.visit(f)


@dataclass(frozen=True)
class If(AST):
    condition: AST
    then: AST
    else_: AST

    def __call__(self, context):
        if self.condition(context):
            context = self.then(context)
        else:
            context = self.else_(context)
        return context


@dataclass(frozen=True)
class Loop(AST):
    condition: AST
    body: AST

    def __call__(self, context):
        while self.condition(context):
            context = self.body(context)
        return context


@dataclass(frozen=True)
class ForN(AST):
    get_n: AST
    body: AST

    def __call__(self, context):
        context = self.get_n(context)
        for iv in range(context.n):
            context = replace(context, iv=iv)
            context = self.body(context)
        return context


@dataclass(frozen=True)
class Call(AST):
    function: Callable

    def __call__(self, context):
        return self.function(context)

    def visit(self, f):
        f(self)


def assign_to_tensor(context: Context):
    context.tensor[context.iv] = context.iv
    return context


def get_tensor_length(context):
    return replace(context, n=context.tensor.shape[0])


def print_tensor_element(context):
    print("Assigned to tensor element", context.iv)
    return context


def printme(message):
    def do_print(context):
        print(message)
        return context

    return Call(do_print)


def iv_is_even(context):
    return context.iv % 2 == 0


schedule = ProgN(
    (
        printme("start kernel"),
        ForN(
            get_n=Call(get_tensor_length),
            body=If(
                condition=Call(iv_is_even),
                then=ProgN(
                    (
                        Call(assign_to_tensor),
                        Call(print_tensor_element),
                    )
                ),
                else_=printme("skipping odd iteration"),
            ),
        ),
        printme("end kernel"),
    )
)


@dataclass
class Visitor:
    seen_progn: bool = False
    seen_nested_progn: bool = False

    def __call__(self, node):
        got_progn = isinstance(node, ProgN)
        self.seen_nested_progn |= self.seen_progn and got_progn
        self.seen_progn = self.seen_progn or got_progn


def analyze_program(program):
    """Example analysis traversing and analyzing the program on the host"""
    visitor = Visitor()
    program.visit(visitor)
    assert visitor.seen_nested_progn


def test_device_lisp():
    analyze_program(schedule)
    import subprocess
    import sys
    from test.util import filecheck

    args = [sys.executable, __file__]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    filecheck(
        out.stdout,
        """
        CHECK:      start kernel
        CHECK-NEXT: Assigned to tensor element 0
        CHECK-NEXT: skipping odd iteration
        CHECK-NEXT: Assigned to tensor element 2
        CHECK-NEXT: skipping odd iteration
        CHECK-NEXT: Assigned to tensor element 4
        CHECK-NEXT: skipping odd iteration
        CHECK-NEXT: Assigned to tensor element 6
        CHECK-NEXT: skipping odd iteration
        CHECK-NEXT: end kernel
        """,
    )


if __name__ == "__main__":

    @cl.kernel
    def kernel(tensor):
        schedule(Context(tensor))

    out = torch.ones(8, dtype=torch.int8).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    out = out.cpu().tolist()
    assert out == [0, 1, 2, 1, 4, 1, 6, 1]
