# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import math
from typing import List

import pytest
import torch

import cuda.tile as ct
from cuda.tile.compilation import CallingConvention
from cuda.tile._ir.ir import Operation
from cuda.tile._ir.core_ops import TypedConst
from cuda.tile._ir.ops import Loop, IfElse, TileExtract, TileReduce
from cuda.tile._ir.arithmetic_ops import Unary
from cuda.tile._compile import compile_tile

from util import assert_close


def _find_ifelse_with_sqrt(block):
    for op in block.traverse():
        if isinstance(op, IfElse):
            if len(_find_sqrt_ops(op.then_block)) == 1:
                return op
    assert False


def _find_nested_loops(block, hoisted_op) -> List[Loop]:
    ret = []
    for op in block:
        if op is hoisted_op:
            continue

        if isinstance(op, Loop):
            assert len(ret) == 0, "Expected loops to be nested"
            ret.append(op)
            ret.extend(_find_nested_loops(op.body, hoisted_op))
    return ret


def _find_sqrt(block) -> Unary:
    sqrt_ops = _find_sqrt_ops(block)
    assert len(sqrt_ops) == 1
    return sqrt_ops[0]


def _find_sqrt_ops(block) -> List[Unary]:
    return [op for op in block.traverse()
            if isinstance(op, Unary) and op.fn == "sqrt"]


def _find_loop_with_extract(block) -> Loop:
    ops = [op for op in block.traverse()
           if isinstance(op, Loop)
           and any(isinstance(inner_op, TileExtract) for inner_op in op.body)]
    assert len(ops) == 1
    return ops[0]


def _find_loop_without_sqrt(block) -> Loop:
    ops = [op for op in block.traverse()
           if isinstance(op, Loop)
           and not any(isinstance(inner_op, Unary) and inner_op.fn == "sqrt"
                       for inner_op in op.body)]
    assert len(ops) == 1
    return ops[0]


def _find_first_ifelse(block) -> IfElse:
    for op in block.traverse():
        if isinstance(op, IfElse):
            return op
    assert False, "No IfElse found in IR"


def _is_inside_loop(op: Operation, loop: Loop):
    for inner_op in loop.body.traverse():
        if op is inner_op:
            return True
    return False


@ct.kernel
def simple_yes(x, a, t):
    for i in range(x.shape[0]):
        val = i + ct.sqrt(t)
        ct.store(x, i, val)


@ct.kernel
def indvar_no(x, a, t):
    for i in range(x.shape[0]):
        val = ct.sqrt(t + i)
        ct.store(x, i, val)


@ct.kernel
def sideeff_no(x, a, t):
    for i in range(x.shape[0]):
        old = ct.atomic_xchg(a, 0, 21)
        val = ct.sqrt(t + old)
        ct.store(x, i, val)


@ct.kernel
def ifelse_yes(x, a, t):
    for i in range(x.shape[0]):
        if t > 0:
            val = ct.sqrt(t)
        else:
            val = 0.0
        ct.store(x, i, val + i)


@ct.kernel
def ifelse_indvar_no(x, a, t):
    for i in range(x.shape[0]):
        if t > 0:
            val = ct.sqrt(t)
        else:
            val = i + 0.0
        ct.store(x, i, val + i)


@ct.kernel
def ifelse_sideeff_no(x, a, t):
    for i in range(x.shape[0]):
        if t > 0:
            val = ct.sqrt(t)
        else:
            val = ct.atomic_xchg(a, 0, 13).item() + 0.0
        ct.store(x, i, val + i)


@ct.kernel
def nested_loops_yes_yes_yes(x, a, t):
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            for k in range(x.shape[0]):
                val = k + ct.sqrt(t)
                ct.store(x, k, val)


@ct.kernel
def nested_loops_no_yes_yes(x, a, t):
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            for k in range(x.shape[0]):
                val = k + ct.sqrt(t + i)
                ct.store(x, k, val)


@ct.kernel
def entire_loop_yes(x, a, t):
    at = ct.load(a, index=(0,), shape=(2,))
    for i in range(x.shape[0]):
        val = 0.0
        for j in range(2):
            val += ct.extract(at, j, ()).item()
        ct.store(x, i, val)


@ct.kernel
def ifelse_cond_indvar_no(x, a, t):
    for i in range(x.shape[0]):
        # condition depends on i
        if i + 1 == x.shape[0]:
            val = ct.sqrt(t)
        else:
            val = 0.0
        ct.store(x, i, val)


@ct.kernel
def ifelse_carry_no(x, a, t):
    for i in range(x.shape[0]):
        # loop-carried value defined in the loop body
        val = i + 1.0
        if t > 0:
            pass
        else:
            val = 0.0
        ct.store(x, i, val)


@ct.kernel
def carried_from_nested_loop_no(x, a, t):
    bid = ct.bid(0)
    for i in range(x.shape[0]):
        k = ct.sqrt(i)
        val = 10.0
        # Can't hoist this loop out of the outer loop because the "continue"
        # operation yield `k` which is not invariant of the outer loop.
        for j in range(bid):
            val = k + k
        ct.store(x, i, val)


def make_cases(tuples):
    return [pytest.param(kernel, op_finder, expected_x, id=kernel._pyfunc.__name__)
            for kernel, op_finder, expected_x in tuples]


@pytest.mark.parametrize("kernel, op_finder, expected_x", make_cases([
    (simple_yes, _find_sqrt, [2.0, 3.0, 4.0]),
    (indvar_no, _find_sqrt, [2.0, math.sqrt(5.0), math.sqrt(6.0)]),
    (sideeff_no, _find_sqrt, [3.0, 5.0, 5.0]),
    (ifelse_yes, _find_ifelse_with_sqrt, [2.0, 3.0, 4.0]),
    (ifelse_indvar_no, _find_ifelse_with_sqrt, [2.0, 3.0, 4.0]),
    (ifelse_sideeff_no, _find_ifelse_with_sqrt, [2.0, 3.0, 4.0]),
    (nested_loops_yes_yes_yes, _find_sqrt, [2.0, 3.0, 4.0]),
    (nested_loops_no_yes_yes, _find_sqrt,
     [math.sqrt(6.0), 1.0 + math.sqrt(6.0), 2.0 + math.sqrt(6.0)]),
    (entire_loop_yes, _find_loop_with_extract, [11.0, 11.0, 11.0]),
    (ifelse_cond_indvar_no, _find_ifelse_with_sqrt, [0.0, 0.0, 2.0]),
    (ifelse_carry_no, _find_first_ifelse, [1.0, 2.0, 3.0]),
    (carried_from_nested_loop_no, _find_loop_without_sqrt, [10.0, 10.0, 10.0])
]))
def test_hoisting(kernel, op_finder, expected_x):
    kernel_name = kernel._pyfunc.__name__
    expected_to_hoist = []
    while True:
        if kernel_name.endswith("_yes"):
            expected_to_hoist.append(True)
            kernel_name = kernel_name[:-4]
        elif kernel_name.endswith("_no"):
            expected_to_hoist.append(False)
            kernel_name = kernel_name[:-3]
        else:
            break

    assert len(expected_to_hoist) > 0, "Please suffix kernel name with _yes or _no"
    expected_to_hoist.reverse()

    x = torch.zeros(3, dtype=torch.float32, device="cuda")
    a = torch.tensor([5, 6, 7], dtype=torch.int32, device="cuda")
    sig = ct.compilation.KernelSignature.from_kernel_args(kernel, (x, a, 4.0),
                                                          CallingConvention.cutile_python_v1())
    [root_block] = compile_tile(kernel._pyfunc, [sig], return_final_ir=True).final_ir

    op = op_finder(root_block)

    nested_loops = _find_nested_loops(root_block, op)
    assert len(nested_loops) == len(expected_to_hoist)

    for loop, expected in zip(nested_loops, expected_to_hoist, strict=True):
        assert _is_inside_loop(op, loop) == (not expected)

    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, a, 4.0))
    ref = torch.tensor(expected_x, dtype=torch.float32, device="cuda")
    assert_close(x, ref)


def test_reduce_body_is_licm_barrier():
    @ct.kernel
    def kernel(x, y):
        xt = ct.load(x, (0, 0), (16, 16))
        yt = ct.reduce(xt, -1, lambda a, b: (a + b) % 5, 0)
        ct.store(y, (0,), yt)

    x = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    y = torch.zeros((16,), dtype=torch.int32, device="cuda")
    sig = ct.compilation.KernelSignature.from_kernel_args(
        kernel, (x, y), CallingConvention.cutile_python_v1()
    )
    [root_block] = compile_tile(
        kernel._pyfunc, [sig], return_final_ir=True, return_cubin=False
    ).final_ir

    [reduce] = [op for op in root_block.traverse() if isinstance(op, TileReduce)]
    modulo_ops = [
        op
        for op in reduce.body.operations
        if isinstance(op, TypedConst) and op.value == 5
    ]
    assert len(modulo_ops) == 1
    [modulo] = modulo_ops

    assert modulo in reduce.body.operations
    assert modulo not in root_block.operations
