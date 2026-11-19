# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from cuda.tile._ir.ops import Loop
from cuda.tile._compile import compile_tile

from util import assert_equal


@ct.kernel
def split_ge_kernel(x):
    for i in range(x.shape[0]):
        val = i
        if i >= 3:
            val *= 10
        ct.store(x, i, val)


def test_split_ge():
    x = torch.zeros(10, dtype=torch.int32, device="cuda:0")
    sig = ct.compilation.KernelSignature.from_kernel_args(
            split_ge_kernel, (x,),
            ct.compilation.CallingConvention.cutile_python_v1())
    [root_block] = compile_tile(split_ge_kernel._pyfunc, [sig],
                                return_final_ir=True, return_cubin=False).final_ir
    loop_ops = [op for op in root_block.traverse() if isinstance(op, Loop)]
    assert len(loop_ops) == 2

    ct.launch(torch.cuda.current_stream(), (1,), split_ge_kernel, (x,))
    ref = torch.tensor([0, 1, 2, 30, 40, 50, 60, 70, 80, 90], dtype=torch.int32, device="cuda:0")
    assert_equal(x, ref)


@ct.kernel
def loop_carried_condition_kernel(output, stop: int):
    col = ct.bid(0)
    for i in range(stop):
        if col > i:
            col -= i + 1
    ct.store(output, (0,), col)


def test_loop_carried_condition():
    output = torch.empty((1,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), loop_carried_condition_kernel, (output, 1))
    assert_equal(output, torch.tensor([0], dtype=torch.int32, device="cuda:0"))


@ct.kernel
def dynamic_split_boundary_kernel(output, split: int):
    for i in range(output.shape[0]):
        value = i
        if i >= split:
            value *= 10
        ct.store(output, i, value)


def test_dynamic_split_boundary():
    output = torch.empty(7, dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), dynamic_split_boundary_kernel, (output, 3))
    ref = torch.tensor([0, 1, 2, 30, 40, 50, 60], dtype=torch.int32, device="cuda:0")
    assert_equal(output, ref)


@ct.kernel
def split_gt_kernel(output, n: int, c: int):
    acc = 0
    for i in range(n):
        if i > c:
            acc = acc + 1000 + i
        else:
            acc = acc + i
    ct.store(output, (0,), acc)


@ct.kernel
def split_le_kernel(output, n: int, c: int):
    acc = 0
    for i in range(n):
        if i <= c:
            acc = acc + 1000 + i
        else:
            acc = acc + i
    ct.store(output, (0,), acc)


@pytest.mark.parametrize("kind", ["gt", "le"])
@pytest.mark.parametrize("c", [-1, 0, 1, 3, 4, 2**31 - 2, 2**31 - 1])
def test_split_point_at_dtype_max(kind, c):
    # The split point for `i > c` and `i <= c` is `c + 1`, which used to wrap
    # around when `c` is the int32 maximum and send every iteration down the
    # wrong branch.
    n = 4
    kernel = split_gt_kernel if kind == "gt" else split_le_kernel
    output = torch.zeros((1,), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (output, n, c))
    taken = (lambda i: i > c) if kind == "gt" else (lambda i: i <= c)
    ref = sum(1000 + i if taken(i) else i for i in range(n))
    assert_equal(output, torch.tensor([ref], dtype=torch.int32, device="cuda:0"))
