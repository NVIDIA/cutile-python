# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from conftest import dtype_id, shape_size_id
import torch
import pytest

from util import estimate_bench_iter
from kernels.vec_add import vec_add


@pytest.fixture(params=[
    # 32 MiB in 1D and 2D
    (1024 * 1024 * 32,),
    (1024 * 1024, 32),
    (1024 * 1024 * 2, 16),
    # 128 MiB in 1D and 2D
    (1024 * 1024 * 128,),
    (1024 * 1024, 128),
    (1024 * 1024 * 4, 32),
    # 1 GiB in 1D and 2D
    (1024 * 1024 * 1024,),
    (1024 * 1024, 1024),
    (1024 * 1024 * 8, 128),
    (1024 * 1024 * 32, 32),
], ids=shape_size_id)
def shape(request):
    return request.param


@pytest.fixture(params=[
    torch.float16, torch.float32
], ids=dtype_id)
def dtype(request):
    return request.param


@pytest.mark.parametrize("use_gather", [False, True])
@pytest.mark.benchmark(group='vec_add')
def bench_vec_add(shape, dtype, backend, use_gather, benchmark):
    if len(shape) == 1:
        n = shape[0]
        a = torch.randn((n,), dtype=dtype, device="cuda")
        b = torch.randn((n,), dtype=dtype, device="cuda")
    else:
        m, n = shape
        a = torch.randn((m, n), dtype=dtype, device="cuda")
        b = torch.randn((m, n), dtype=dtype, device="cuda")

    c = backend(a, b, use_gather)
    ref = a + b
    torch.testing.assert_close(c, ref, atol=1e-3, rtol=1e-3)
    torch.cuda.synchronize()

    warmup_rounds, iterations, rounds = estimate_bench_iter(backend, (a, b, use_gather),
                                                            cudagraph=True)

    benchmark.pedantic(
        backend, (a, b, use_gather),
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
        cudagraph=True
    )

    flop_count = 0
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (a, b, c)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def cutile_vec_add(a, b, use_gather):
    return vec_add(a, b, use_gather=use_gather)


def torch_vec_add(a, b, use_gather):
    return a + b
