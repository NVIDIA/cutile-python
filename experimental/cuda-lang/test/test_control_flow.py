# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError
import torch


def test_if_else():
    @cl.kernel
    def kernel(X, idx: cl.Constant[int]):
        if X[idx] > 0:
            X[idx] = 2
        else:
            X[idx] = 1

    X = torch.tensor([-1, 1], dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (X, 0))
    assert X[0] == 1

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (X, 1))
    assert X[1] == 2


def test_while_loop():
    @cl.kernel
    def kernel(X):
        while X[0] > 0:
            X[0] = X[0] - 1

    X = torch.tensor([3], dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (X,))
    assert X[0] == 0


@pytest.mark.parametrize("start", [0, 1, 2])
@pytest.mark.parametrize("step", [1, 2])
def test_for_loop(start, step):
    @cl.kernel
    def kernel(X):
        tot = 0
        for i in range(start, X[0], step):
            tot = tot + 2
        X[0] = tot

    x = [3]
    dx = torch.tensor(x, dtype=torch.int32, device="cuda")
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (dx,),
    )
    expect = len(range(start, x[0], step)) * 2
    assert dx[0] == expect


def test_negative_stride():
    @cl.kernel
    def kernel():
        for i in range(2, 1, -1):
            pass

    with pytest.raises(TypeCheckingError, match="Step must be positive, got -1"):
        cl.launch(
            torch.cuda.current_stream(),
            (),
            (),
            kernel,
            (),
        )


def test_static_eval_return_with_static_assert_else_makedummy():
    @cl.function
    def choose(flag: cl.Constant[int]):
        if cl.static_eval(flag == 0):
            return cl.int32(7)
        else:
            cl.static_assert(False, "unsupported flag")

    @cl.kernel
    def kernel(flag: cl.Constant[int], out):
        out[0] = choose(flag)

    out = torch.empty(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (0, out))
    assert out.cpu().item() == 7


# TODO: test while loop inside a for loop to ensure continue correctly
# handles loop-carried variables
