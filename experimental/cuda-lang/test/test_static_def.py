# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import cuda.lang as cl


def test_static_def():
    @cl.static_def
    def make_contiguous_strides(shape):
        if len(shape) == 0:
            return ()
        ret = [1]
        for x in shape[:-1]:
            ret.append(ret[-1] * x)
        return tuple(ret)

    @cl.kernel
    def kern(x, y):
        s = make_contiguous_strides(x.shape)
        cl.static_assert(len(s) == 3)
        y[0] = s[0]
        y[1] = s[1]
        y[2] = s[2]

    x = torch.zeros((2, 5, 7), device="cuda")
    y = torch.zeros((3,), dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x, y,))
    assert y.tolist() == [1, 2, 10]
