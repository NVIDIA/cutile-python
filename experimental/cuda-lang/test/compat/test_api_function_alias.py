# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
import torch

from cuda.tile import StaticException


@cl.compat.api_function_alias(cl.static_exception)
def my_static_exception_alias(exc, /):
    return exc


def test_alias_for_static_exception():
    @cl.kernel
    def kern():
        raise my_static_exception_alias(ValueError("Hello"))

    with pytest.raises(StaticException, match="Hello"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, ())
