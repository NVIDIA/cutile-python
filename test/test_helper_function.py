# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect

import pytest
import torch
from math import ceil
import cuda.tile as ct
from util import assert_close
from cuda.tile._exception import TileTypeError, TileSyntaxError, TileRecursionError


@pytest.fixture
def shape():
    return (512, 128)


@pytest.fixture
def tile():
    return 16


def helper_function(input):
    return input + 1


def main_kernel_calling_helper(x, y, output, B: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile_x = ct.load(x, index=(px, 0), shape=(B, N))
    tile_y = ct.load(y, index=(px, 0), shape=(B, 1))
    x1 = helper_function(tile_x)
    # This should be a deduped call as tile_y and y1 have the same type.
    y1 = helper_function(tile_y)
    y2 = helper_function(y1)
    out = x1 + y2
    ct.store(output, index=(px, 0), tile=out)


def test_helper_function_multiple_calls(shape, tile):
    x = torch.rand(shape, dtype=torch.float32, device="cuda")
    y = torch.rand((shape[0], 1), dtype=torch.float32, device="cuda")
    z = torch.zeros_like(x)
    kernel = ct.kernel(main_kernel_calling_helper)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, z, tile, shape[1]))
    ref_result = x + y + 3
    assert_close(z, ref_result, atol=1e-4, rtol=1e-5)


def helper_function_no_return():
    # Do nothing for now.
    pass


def helper_function_multiple_returns(input, input1):
    return input + 1, input1 + 1


@ct.kernel
def main_kernel_multiple_returns(x, y, output, B: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile_x = ct.load(x, index=(px, 0), shape=(B, N))
    tile_y = ct.load(y, index=(px, 0), shape=(B, 1))
    helper_function_no_return()
    x1, y1 = helper_function_multiple_returns(tile_x, tile_y)
    out = x1 + y1
    ct.store(output, index=(px, 0), tile=out)


def test_helper_function_multiple_returns(shape, tile):
    x = torch.rand(shape, dtype=torch.float32, device="cuda")
    y = torch.rand((shape[0], 1), dtype=torch.float32, device="cuda")
    z = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, main_kernel_multiple_returns,
              (x, y, z, tile, shape[1]))
    ref_result = x + y + 2
    assert_close(z, ref_result, atol=1e-4, rtol=1e-5)


def helper_function_keyword_args(input, input1, arg=False):
    offset = 1 if arg else -1
    return input + offset, input1 + offset


@ct.kernel
def main_kernel_keyword_args(x, y, output, B: ct.Constant[int], N: ct.Constant[int], arg):
    px = ct.bid(0)
    tile_x = ct.load(x, index=(px, 0), shape=(B, N))
    tile_y = ct.load(y, index=(px, 0), shape=(B, 1))
    x1, y1 = helper_function_keyword_args(tile_x, tile_y, arg=arg)
    out = x1 + y1
    ct.store(output, index=(px, 0), tile=out)


@ct.kernel
def main_kernel_keyword_args_default(x, y, output, B: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile_x = ct.load(x, index=(px, 0), shape=(B, N))
    tile_y = ct.load(y, index=(px, 0), shape=(B, 1))
    x1, y1 = helper_function_keyword_args(tile_x, tile_y)
    out = x1 + y1
    ct.store(output, index=(px, 0), tile=out)


@pytest.mark.parametrize("func", [main_kernel_keyword_args,
                                  main_kernel_keyword_args_default])
def test_helper_function_keyword_args(shape, tile, func):
    x = torch.rand(shape, dtype=torch.float32, device="cuda")
    y = torch.rand((shape[0], 1), dtype=torch.float32, device="cuda")
    z = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    if func is main_kernel_keyword_args:
        ct.launch(torch.cuda.current_stream(), grid, main_kernel_keyword_args,
                  (x, y, z, tile, shape[1], False))
    else:
        ct.launch(torch.cuda.current_stream(), grid, main_kernel_keyword_args_default,
                  (x, y, z, tile, shape[1]))
    ref_result = x + y - 2
    assert_close(z, ref_result, atol=1e-4, rtol=1e-5)


def helper_function_recursive_calls(input, N):
    if N > 0:
        x = helper_function_recursive_calls(input, N - 1) + 1
    else:
        x = input
    return x


@ct.kernel
def main_kernel_recursive_calls(x, output, N: ct.Constant[int]):
    tile_x = ct.gather(x, ())
    x1 = helper_function_recursive_calls(tile_x, N)
    ct.scatter(output, (), x1)


def test_reject_runaway_recursion():
    x = torch.tensor(100.0, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    with pytest.raises(TileRecursionError):
        ct.launch(torch.cuda.current_stream(), (1,), main_kernel_recursive_calls,
                  (x, y, 100000))


def test_accept_reasonable_recursion():
    x = torch.tensor(100.0, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), main_kernel_recursive_calls,
              (x, y, 109))
    assert y.item() == 209.0


def helper_function_array_arguments(x, y, output, B: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile_x = ct.load(x, index=(px, 0), shape=(B, N))
    tile_y = ct.load(y, index=(px, 0), shape=(B, 1))
    out = tile_x + tile_y
    ct.store(output, index=(px, 0), tile=out)


@ct.kernel
def main_kernel_array_arguments_in_helper(x, y, output, B: ct.Constant[int], N: ct.Constant[int]):
    helper_function_array_arguments(x, y, output, B, N)


def test_helper_function_array_arguments(shape, tile):
    x = torch.rand(shape, dtype=torch.float32, device="cuda")
    y = torch.rand((shape[0], 1), dtype=torch.float32, device="cuda")
    z = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, main_kernel_array_arguments_in_helper,
              (x, y, z, tile, shape[1]))
    ref_result = x + y
    assert_close(z, ref_result, atol=1e-4, rtol=1e-5)


def helper_function_early_return(tile_x, tile_y, early_return):
    if early_return:
        return tile_x
    return tile_x + tile_y


@ct.kernel
def helper_function_early_return_kernel(x, y, output,
                                        B: ct.Constant[int],
                                        N: ct.Constant[int],
                                        early_return: bool):
    px = ct.bid(0)
    tile_x = ct.load(x, index=(px, 0), shape=(B, N))
    tile_y = ct.load(y, index=(px, 0), shape=(B, 1))
    out = helper_function_early_return(tile_x, tile_y, early_return)
    ct.store(output, index=(px, 0), tile=out)


@pytest.mark.parametrize("early_return", [True, False])
def test_helper_function_early_return(early_return):
    shape = (512, 128)
    tile = 16
    x = torch.rand(shape, dtype=torch.float32, device="cuda")
    y = torch.rand((shape[0], 1), dtype=torch.float32, device="cuda")
    z = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, helper_function_early_return_kernel,
              (x, y, z, tile, shape[1], early_return))
    ref = x if early_return else x + y
    assert_close(z, ref)


def early_return_inside_while_loop(n):
    a, b = 1, 1
    while True:
        a, b = b, a + b
        if b > n:
            return b


def early_return_inside_for_loop(n):
    a, b = 1, 1
    for i in range(n):
        a, b = b, a + b
        if b > n:
            return b


def early_return_inside_loop(helper_func):
    @ct.kernel
    def early_return_inside_loop_kernel(n, y):
        n = ct.load(n, index=(0,), shape=(1,))
        res = ct.full((1,), helper_func(n.item()), dtype=ct.int32)
        ct.store(y, (0,), res)
    return early_return_inside_loop_kernel


def test_early_return_inside_while_loop():
    n = torch.tensor([15], dtype=torch.int32, device="cuda")
    out = torch.zeros_like(n)
    kernel = early_return_inside_loop(early_return_inside_while_loop)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (n, out))
    assert out.cpu().item() == 21


def test_early_return_inside_for_loop():
    n = torch.tensor([15], dtype=torch.int32, device="cuda")
    out = torch.zeros_like(n)
    kernel = early_return_inside_loop(early_return_inside_for_loop)
    with pytest.raises(TileSyntaxError, match="Returning from a for loop is not supported"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (n, out))


def return_after_while_loop(n):
    while n > 0:
        n = n - 1
    return n


def test_return_after_while_loop():
    n = torch.tensor([3], dtype=torch.int32, device="cuda")
    out = torch.zeros_like(n)
    kernel = early_return_inside_loop(return_after_while_loop)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (n, out))
    assert out.cpu().item() == 0


def loops(n):
    a = 0
    for i in range(n):
        a += 1
        for j in range(n):
            a += 1
    return a


@ct.kernel
def loops_kernel(n, y):
    n = ct.load(n, index=(0,), shape=(1,))
    res = ct.full((1,), loops(n.item()), dtype=ct.int32)
    ct.store(y, (0,), res)


def test_loops_in_helper_function():
    n = torch.tensor([5], dtype=torch.int32, device="cuda")
    out = torch.zeros_like(n)
    ct.launch(torch.cuda.current_stream(), (1,), loops_kernel, (n, out))
    assert out.cpu().item() == 30


def helper_reassign_param(bid):
    bid = bid + 10
    return bid + 5


@ct.kernel
def call_helper_reassign_param(x):
    val = helper_reassign_param(ct.bid(0))
    t = ct.full((1,), val, ct.int32)
    ct.store(x, (0,), t)


def test_helper_function_reassign_param():
    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), call_helper_reassign_param, (x,))
    assert x.cpu().item() == 15


@ct.function
def helper_function_using_ct_api(x, output, B: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile_x = ct.load(x, index=(px, 0), shape=(B, N))
    ct.store(output, index=(px, 0), tile=tile_x + 1)


def test_calling_function_from_host(shape, tile):
    x = torch.rand(shape, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    with pytest.raises(RuntimeError, match="Tile functions can only be called from tile code."):
        helper_function_using_ct_api(x, y, tile, shape[1])


@ct.kernel
def kernel_calling_function_using_ct_api(x, output, B: ct.Constant[int], N: ct.Constant[int]):
    helper_function_using_ct_api(x, output, B, N)


def test_helper_function_using_ct_api(shape, tile):
    x = torch.rand(shape, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel_calling_function_using_ct_api,
        (x, y, tile, shape[1])
    )
    ref_result = x + 1
    assert_close(y, ref_result, atol=1e-4, rtol=1e-5)


def test_error_message_stack_trace():
    def bar(x):  # Line +1
        ct.abracadabra(x)

    def foo(x):  # Line + 4
        bar(x)

    @ct.kernel
    def kernel(x):  # Line +8
        foo(x)

    x = torch.zeros((), device="cuda")
    _, first_line = inspect.getsourcelines(test_error_message_stack_trace)
    msg_regex = (
        "Module 'cuda.tile' has no attribute 'abracadabra'.*\n"
        f".*test_helper_function.py\", line {first_line + 9}.*, in kernel:\n"
        f" *foo\\(x\\)\n"
        f" *\\^\\^\\^\\^\\^\\^\n"
        f".*test_helper_function.py\", line {first_line + 5}.*, in foo:\n"
        f" *bar\\(x\\)\n"
        f" *\\^\\^\\^\\^\\^\\^\n"
        f".*test_helper_function.py\", line {first_line + 2}.*, in bar:\n"
        f"            ct.abracadabra\\(x\\)\n"
        f"            \\^\\^\\^\\^\\^\\^\\^\\^\\^\\^\\^\\^\\^\\^\n"
    )
    with pytest.raises(TileTypeError, match=msg_regex):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
