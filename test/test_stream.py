# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.testing import make_tensor
import cuda.tile as ct


@ct.kernel
def array_copy_1d(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(y, index=(bid,), tile=tx)


def _test_stream(stream, sync):
    x = make_tensor(4096, dtype=torch.float32, device="cuda:0")
    y = torch.zeros_like(x)
    torch.cuda.synchronize()
    ct.launch(stream, (1,), array_copy_1d, (x, y, 4096))
    sync()
    torch.testing.assert_close(x, y)


# -- Test PyTorch Stream --
def test_torch_pass_stream():
    stream = torch.cuda.Stream()
    _test_stream(stream, stream.synchronize)


def test_torch_pass_stream_ptr():
    stream = torch.cuda.Stream()
    _test_stream(stream.cuda_stream, stream.synchronize)


# -- Test CuPy Stream --
def test_cupy_pass_stream(cupy):
    stream = cupy.cuda.Stream()
    _test_stream(stream, stream.synchronize)


def test_cupy_pass_stream_ptr(cupy):
    stream = cupy.cuda.Stream()
    _test_stream(stream.ptr, stream.synchronize)
