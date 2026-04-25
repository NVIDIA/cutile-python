# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.testing import make_tensor
import cupy
import cuda.tile as ct
from ctypes import c_void_p
from cuda.bindings.driver import CUstream


@ct.kernel
def array_copy_1d(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(y, index=(bid,), tile=tx)


def _test_stream(stream, sync):
    x = make_tensor(4096, dtype=torch.float32, device="cuda")
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
def test_cupy_pass_stream():
    stream = cupy.cuda.Stream()
    _test_stream(stream, stream.synchronize)


def test_cupy_pass_stream_ptr():
    stream = cupy.cuda.Stream()
    _test_stream(stream.ptr, stream.synchronize)


# -- Test Numba Stream --
def test_numba_pass_stream(numba_cuda):
    stream = numba_cuda.stream()
    _test_stream(stream, stream.synchronize)


def test_numba_pass_stream_ptr(numba_cuda):
    stream = numba_cuda.stream()
    handle = stream.handle
    # numba-cuda < 0.30: handle is ctypes c_void_p
    # numba-cuda >= 0.30: handle is cuda.bindings.driver.CUstream
    if isinstance(handle, c_void_p):
        stream_ptr = handle.value
    elif isinstance(handle, CUstream):
        stream_ptr = int(handle)
    else:
        raise ValueError(f"Unsupported stream type: {type(handle)}")
    _test_stream(stream_ptr, stream.synchronize)
