# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from math import ceil
from conftest import dtype_id, shape_id
import torch
import pytest
import cuda.tile as ct

from util import estimate_bench_iter, torch_use_tf32_matmul, is_ampere_or_ada
from kernels.matmul import (
    matmul_kernel, matmul_split_k_kernel, batch_matmul_kernel, persistent_matmul_kernel
)


@pytest.fixture(params=[
    torch.float16, torch.float32
], ids=dtype_id)
def dtype(request):
    return request.param


def _run_matmul_benchmark(shape, dtype, backend, benchmark, extra_args=(), atol=1e-3, rtol=1e-3):
    m, n, k = shape
    A = torch.rand((m, k), dtype=dtype, device="cuda")
    B = torch.rand((k, n), dtype=dtype, device="cuda")
    C = torch.zeros((m, n), dtype=dtype, device="cuda")

    args = (A, B, C) + extra_args

    with torch_use_tf32_matmul():
        backend(*args)
        torch.testing.assert_close(C, A @ B, atol=atol, rtol=rtol)

    torch.cuda.synchronize()
    warmup_rounds, iterations, rounds = estimate_bench_iter(backend, args)
    benchmark.pedantic(
        backend, args,
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
    )

    flop_count = 2 * m * n * k
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (A, B, C)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def _run_batch_matmul_benchmark(
    shape, dtype, backend, benchmark, extra_args=(), atol=1e-3, rtol=1e-3
):
    b, m, n, k = shape
    A = torch.rand((b, m, k), dtype=torch.float32, device="cuda").to(dtype)
    B = torch.rand((b, k, n), dtype=torch.float32, device="cuda").to(dtype)
    C = torch.zeros((b, m, n), dtype=torch.float32, device="cuda")

    args = (b, A, B, C) + extra_args

    with torch_use_tf32_matmul():
        backend(*args)
        if dtype != torch.float8_e5m2:
            ref = ref_batch_matmul(b, A, B)
            torch.testing.assert_close(C, ref, atol=atol, rtol=rtol)

    torch.cuda.synchronize()
    warmup_rounds, iterations, rounds = estimate_bench_iter(backend, args)
    benchmark.pedantic(
        backend, args,
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
    )

    flop_count = 2 * b * m * n * k
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (A, B, C)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


# =============================== Matmul =============================

@pytest.fixture(params=[
    (1024, 1024, 1024),
    (8192, 8192, 8192),
    (12288, 4096, 2560),
], ids=shape_id)
def shape(request):
    return request.param


@pytest.mark.benchmark(group='matmul')
def bench_matmul(shape, dtype, backend, benchmark):
    _run_matmul_benchmark(shape, dtype, backend, benchmark)


def cutile_matmul(A, B, C):
    tm, tn, tk = 256, 256, 64
    m, n, _ = A.shape[0], B.shape[1], A.shape[1]
    grid = (ceil(m / tm) * ceil(n / tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, C, tm, tn, tk))


def torch_matmul(A, B, C):
    with torch_use_tf32_matmul():
        torch.matmul(A, B, out=C)


# =============================== Matmul Split K =============================


@pytest.fixture(params=[
    (256, 256, 4096),
    (128, 128, 8192)
], ids=shape_id)
def split_k_shape(request):
    return request.param


@pytest.mark.benchmark(group='matmul_split_k')
def bench_matmul_split_k(split_k_shape, dtype, backend, benchmark):
    m, n, _ = split_k_shape
    tile_sizes = (32, 64, 256)
    LOCKS = torch.zeros(ceil(m / tile_sizes[0]) * ceil(n / tile_sizes[1]),
                        dtype=torch.int32, device="cuda")
    COUNTS = torch.zeros_like(LOCKS)
    extra_args = (LOCKS, COUNTS, tile_sizes)
    _run_matmul_benchmark(split_k_shape, dtype, backend, benchmark, extra_args, rtol=2e-3)


def cutile_matmul_split_k(A, B, C, LOCKS, COUNTS, tile_sizes):
    tm, tn, tk = tile_sizes
    split_k = 4
    m, n, _ = A.shape[0], B.shape[1], A.shape[1]
    grid = (ceil(m / tm) * ceil(n / tn), split_k, 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_split_k_kernel,
              (A, B, C, LOCKS, COUNTS, tm, tn, tk, split_k))


def torch_matmul_split_k(A, B, C, *args):
    torch_matmul(A, B, C)


# =============================== Batch Matmul in FP8=============================

@pytest.fixture(params=[
    (2, 1024, 1024, 1024),
    (4, 8192, 8192, 2000),
    (8, 12288, 4096, 2560),
], ids=shape_id)
def batch_matmul_shape(request):
    return request.param


@pytest.fixture(params=[
    torch.float8_e4m3fn, torch.float8_e5m2
], ids=dtype_id)
def fp8_dtype(request):
    return request.param


@pytest.mark.skipif(is_ampere_or_ada(), reason="float8 not supported on Ampere or Ada")
@pytest.mark.benchmark(group='batch_matmul')
def bench_batch_matmul(batch_matmul_shape, fp8_dtype, backend, benchmark):
    _run_batch_matmul_benchmark(batch_matmul_shape, fp8_dtype, backend, benchmark)


def cutile_batch_matmul(bs, A, B, C):
    tm, tn, tk = 256, 256, 64
    m, n = A.shape[1], B.shape[2]
    grid = (bs, ceil(m / tm), ceil(n / tn))
    ct.launch(torch.cuda.current_stream(), grid, batch_matmul_kernel, (A, B, C, tm, tn, tk))


def torch_batch_matmul(bs, A, B, C):
    if A.dtype == torch.float8_e5m2:
        pytest.skip("float8_e5m2 matmul on torch is not supported")
    inv_sa = torch.tensor(1.0, device=A.device, dtype=torch.float32)
    inv_sb = torch.tensor(1.0, device=B.device, dtype=torch.float32)
    with torch_use_tf32_matmul():
        for i in range(bs):
            # Only multiplication of row-major and column-major matrices is supported by cuBLASLt
            # So we need to transpose B to column-major view
            A_row = A[i].contiguous()
            B_col = B[i].transpose(-2, -1).contiguous().transpose(-2, -1)
            C[i] = torch._scaled_mm(
                A_row, B_col, scale_a=inv_sa, scale_b=inv_sb, out_dtype=torch.float32
            )


def ref_batch_matmul(bs, A, B):
    ref = torch.zeros((bs, A.shape[1], B.shape[2]), dtype=torch.float32, device="cuda")
    torch_batch_matmul(bs, A, B, ref)
    return ref


# =============================== Persistent Matmul =============================

@pytest.mark.benchmark(group='persistent_matmul')
def bench_persistent_matmul(shape, dtype, backend, benchmark):
    _run_matmul_benchmark(shape, dtype, backend, benchmark)


def cutile_persistent_matmul(A, B, C):
    NUM_SMS = torch.cuda.get_device_properties(
            "cuda"
        ).multi_processor_count
    M, N = A.shape[0], B.shape[1]
    tm, tn, tk = 256, 256, 64

    grid_size = min(
        NUM_SMS,
        ceil(M / tm) * ceil(N / tn),
    )
    grid = (grid_size,)
    ct.launch(torch.cuda.current_stream(), grid, persistent_matmul_kernel, (A, B, C, tm, tn, tk))


def torch_persistent_matmul(A, B, C, *args):
    torch_matmul(A, B, C)
