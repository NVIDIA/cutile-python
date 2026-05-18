# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from math import ceil
from conftest import dtype_id, shape_id
import torch
import pytest

import cuda.tile as ct
from util import estimate_bench_iter
from kernels.transpose import transpose_kernel


@pytest.fixture(params=[
    torch.float16, torch.float32
], ids=dtype_id)
def dtype(request):
    return request.param


def _run_transpose_benchmark(shape, dtype, backend, benchmark, atol=1e-3, rtol=1e-3):
    m, n = shape
    A = torch.rand((m, n), dtype=dtype, device="cuda")
    B = torch.zeros((n, m), dtype=dtype, device=A.device)

    backend(A, B)
    torch.testing.assert_close(B, A.T, atol=atol, rtol=rtol)
    torch.cuda.synchronize()
    warmup_rounds, iterations, rounds = estimate_bench_iter(backend, (A, B), cudagraph=True)
    benchmark.pedantic(
        backend, (A, B),
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
        cudagraph=True,
    )

    flop_count = m * n
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (A, B)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


@pytest.fixture(params=[
    (1024, 1024),
    (2048, 8000),
    (12288, 4096),
], ids=shape_id)
def shape(request):
    return request.param


@pytest.mark.benchmark(group='transpose')
def bench_transpose(shape, dtype, backend, benchmark):
    _run_transpose_benchmark(shape, dtype, backend, benchmark)


def cutile_transpose(A, B):
    tm, tn = 256, 256
    m, n = A.shape[0], A.shape[1]
    grid = (ceil(m / tm), ceil(n / tn), 1)
    ct.launch(torch.cuda.current_stream(), grid, transpose_kernel, (A, B, tm, tn))


def torch_transpose(A, B):
    out = torch.transpose(A, 0, 1)
    B.copy_(out)
