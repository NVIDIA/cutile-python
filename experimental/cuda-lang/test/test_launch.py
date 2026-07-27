# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import cuda.lang as cl
import torch
import pytest

from cuda.tile._cext import cconv_v3_enabled


def test_too_many_kwargs():
    @cl.kernel()
    def kernel():
        pass

    bad_kwargs = {f"kw{i}": i for i in range(20)}

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (), **bad_kwargs)


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_kernel_arg():
    @dataclass(frozen=True)
    class KernelArgs:
        out: Any
        scale: int
        bias: float

    @cl.kernel()
    def kernel(a):
        i = cl.thread_index(0)
        a.out[i] = i * a.scale + a.bias

    x = torch.zeros((4,), dtype=torch.float32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (4,), kernel, (KernelArgs(x, 10, 0.5),))
    assert x.tolist() == [0.5, 10.5, 20.5, 30.5]
