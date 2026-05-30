# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import torch
import pytest


class DlpackProxy:
    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        return self.tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


@ct.kernel
def add_one(x):
    xi = ct.load(x, 0, ())
    xi = xi + 1
    ct.store(x, 0, xi)


def test_simple():
    x = torch.zeros(1, device='cuda')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream = torch.cuda.current_stream()
        ct.launch(stream, (1,), add_one, (x,))

    assert x.item() == 0
    for _ in range(10):
        graph.replay()
    assert x.item() == 10


def test_proxy_dlpack():
    x = torch.zeros(1, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), add_one, (DlpackProxy(x),))
    torch.cuda.synchronize()
    assert x.item() == 1


def test_proxy_dlpack_cudagraph():
    x = torch.zeros(1, device='cuda')
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match=r"DLPack array argument in CUDAGraph isn't supported yet"):
        with torch.cuda.graph(graph):
            ct.launch(torch.cuda.current_stream(), (1,), add_one, (DlpackProxy(x),))


@ct.kernel
def matmul_accumulate(x, y, z):
    acc = ct.load(z, (0, 0), (16, 16))
    for k in range(4):
        tx = ct.load(x, (0, k), (16, 4))
        ty = ct.load(y, (k, 0), (4, 16))
        acc = ct.mma(tx, ty, acc)
    ct.store(z, (0, 0), acc)


def test_matmul():
    x = torch.ones((16, 16), dtype=torch.float16, device='cuda')
    y = torch.ones((16, 16), dtype=torch.float16, device='cuda')
    z = torch.zeros_like(x)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream = torch.cuda.current_stream()
        ct.launch(stream, (1,), matmul_accumulate, (x, y, z))

    N = 4
    for _ in range(N):
        graph.replay()

    ref = torch.full((16, 16), 16 * N, dtype=torch.float16, device='cuda')
    assert torch.all(z == ref)


@ct.kernel
def list_copy(As, Bs, N: ct.Constant[int]):
    for i in range(len(As)):
        tx = ct.load(As[i], (0,), (N,))
        ct.store(Bs[i], (0,), tile=tx)


def test_list_of_array():
    N = 8
    x = torch.rand((N, ), device='cuda')
    y = torch.empty_like(x)

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match=r'List argument in CUDAGraph isn\'t supported yet'):
        with torch.cuda.graph(graph):
            ct.launch(torch.cuda.current_stream(),
                      (1,),
                      list_copy, ([x,], [y,], N))
