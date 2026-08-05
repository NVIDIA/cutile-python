# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    sig = ct.compilation.KernelSignature.from_kernel_args(
            split_ge_kernel, (x,),
            ct.compilation.CallingConvention.cutile_python_v1())
    [root_block] = compile_tile(split_ge_kernel._pyfunc, [sig],
                                return_final_ir=True, return_cubin=False).final_ir
    loop_ops = [op for op in root_block.traverse() if isinstance(op, Loop)]
    assert len(loop_ops) == 2

    ct.launch(torch.cuda.current_stream(), (1,), split_ge_kernel, (x,))
    ref = torch.tensor([0, 1, 2, 30, 40, 50, 60, 70, 80, 90], dtype=torch.int32, device="cuda")
    assert_equal(x, ref)


@ct.kernel
def loop_carried_condition_kernel(output, stop: int):
    col = ct.bid(0)
    for i in range(stop):
        if col > i:
            col -= i + 1
    ct.store(output, (0,), col)


def test_loop_carried_condition():
    output = torch.empty((1,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), loop_carried_condition_kernel, (output, 1))
    assert_equal(output, torch.tensor([0], dtype=torch.int32, device="cuda"))


@ct.kernel
def dynamic_split_boundary_kernel(output, split: int):
    for i in range(output.shape[0]):
        value = i
        if i >= split:
            value *= 10
        ct.store(output, i, value)


def test_dynamic_split_boundary():
    output = torch.empty(7, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), dynamic_split_boundary_kernel, (output, 3))
    ref = torch.tensor([0, 1, 2, 30, 40, 50, 60], dtype=torch.int32, device="cuda")
    assert_equal(output, ref)
