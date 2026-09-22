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

    x = torch.zeros((2, 5, 7), device="cuda:0")
    y = torch.zeros((3,), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x, y,))
    assert y.tolist() == [1, 2, 10]


def test_static_def_shift_operators():
    @cl.static_def
    def shifts(x):
        augmented_left = x
        augmented_right = x
        augmented_left <<= 3
        augmented_right >>= 2
        return x << 1, x >> 1, augmented_left, augmented_right

    @cl.kernel
    def kern(x, y):
        left, right, augmented_left, augmented_right = shifts(x[0])
        y[0] = left
        y[1] = right
        y[2] = augmented_left
        y[3] = augmented_right

    x = torch.tensor([-16], dtype=torch.int32, device="cuda:0")
    y = torch.zeros((4,), dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x, y,))
    assert y.tolist() == [-32, -8, -128, -4]
