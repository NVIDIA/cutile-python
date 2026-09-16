# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest

import cuda.tile as ct
import torch


def test_raise_static_exception():
    @ct.kernel
    def kern():
        raise ct.static_exception(ValueError("Hello"))

    with pytest.raises(ct.StaticException,
                       match=re.escape("Exception was raised at compile time (ValueError: Hello)")):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_plain_raise_syntax_error():
    @ct.kernel
    def kern():
        raise ValueError("Hello")

    with pytest.raises(ct.TileSyntaxError,
                       match=re.escape("Raised exception must be wrapped in static_exception()")):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_invalid_exception_type():
    @ct.kernel
    def kern():
        raise ct.static_exception(123)

    with pytest.raises(ct.TileStaticEvalError,
                       match=re.escape("Exception was raised inside static_exception() expression"
                                       " (TypeError: 'int' is not derived from 'BaseException')")):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_not_raised_in_dead_branch():
    @ct.kernel
    def kern():
        if 2 + 2 == 3:
            raise ct.static_exception(ValueError("Hello"))

    ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_static_exception_outside_raise():
    @ct.kernel
    def kernel():
        ct.static_exception(123)

    with pytest.raises(ct.TileSyntaxError,
                       match=re.escape("static_exception() is only allowed after `raise`")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_static_eval_error_when_called_indirectly():
    @ct.kernel
    def kernel_indirect(y):
        f = ct.static_exception
        f(123)

    y = torch.zeros((), dtype=torch.int32, device="cuda:0")
    with pytest.raises(ct.TileSyntaxError,
                       match=re.escape("static_exception() must be used directly")):
        ct.launch(torch.cuda.current_stream(), (1,), kernel_indirect, (y,))
