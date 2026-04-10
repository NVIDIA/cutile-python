# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import cuda.tile as ct
import torch

from cuda.tile import TileTypeError


@ct.kernel
def array_attr_kernel(X, out):
    ndim = X.ndim
    shape = X.shape
    strides = X.strides
    ct.static_assert(ndim == 3)
    ct.static_assert(len(shape) == ndim)
    ct.static_assert(len(strides) == ndim)

    ct.store(out, (0,), shape[0])
    ct.store(out, (1,), shape[1])
    ct.store(out, (2,), shape[2])
    ct.store(out, (3,), strides[0])
    ct.store(out, (4,), strides[1])
    ct.store(out, (5,), strides[2])


def test_array_attr():
    x = torch.zeros((2, 3, 4), device='cuda')
    out = torch.zeros(6, device='cuda', dtype=torch.int64)
    ct.launch(torch.cuda.current_stream(),
              (1,),
              array_attr_kernel, (x, out))
    assert list(out[0:3]) == list(x.shape)
    assert list(out[3:6]) == list(x.stride())


def test_array_getitem():
    @ct.kernel
    def kernel(x):
        x[0]

    x = torch.zeros((10,), device='cuda')
    with pytest.raises(TileTypeError, match="Arrays are not directly subscriptable"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_array_setitem():
    @ct.kernel
    def kernel(x):
        x[0] = 3.0

    x = torch.zeros((10,), device='cuda')
    with pytest.raises(TileTypeError, match="Arrays do not support item assignment. Use store()"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
