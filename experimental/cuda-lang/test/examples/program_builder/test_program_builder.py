# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import sys
from pathlib import Path

import torch
import cuda.lang as cl

if not __package__:
    sys.path.insert(0, str(Path(__file__).parents[3]))
    __package__ = "test.examples.program_builder"

from .program_builder import (
    Call,
    ForN,
    If,
    ProgN,
    Visitor,
    VisitorIterate,
)

__doc__ = """
Demonstrates the flexibility of frozen dataclasses by constructing a dynamic
program out of dataclasses on the host, analyzing the program on the host,
and executing the program on the device.
"""


@dataclass(frozen=True)
class Context:
    tensor: cl.Array
    iv: int = 0


def assign_to_tensor(context: Context):
    context.tensor[context.iv] = context.iv
    return context


def get_tensor_length(context):
    return context.tensor.shape[0]


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
class ProgNAnalysis(Visitor):
    seen_progn: bool = False
    seen_nested_progn: bool = False

    def __call__(self, node):
        got_progn = isinstance(node, ProgN)
        self.seen_nested_progn |= self.seen_progn and got_progn
        self.seen_progn = self.seen_progn or got_progn
        return (
            VisitorIterate.STOP if self.seen_nested_progn else VisitorIterate.CONTINUE
        )


def analyze_program(program):
    """Example analysis traversing and analyzing the program on the host"""
    visitor = ProgNAnalysis()
    program.visit(visitor)
    assert visitor.seen_nested_progn


def test_simple_program_builder():
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
