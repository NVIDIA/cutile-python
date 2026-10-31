# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from conftest import shape_id, requires_tileiras
import torch
import pytest
import cuda.tile as ct
from util import estimate_bench_iter, require_blackwell_or_newer
from functools import cache

from kernels.kernel_utils import (
    block_quantize_f4e2m1fn_f8e4m3fn, block_quantize_f8e4m3fn_f8e8m0fnu, swizzle_32_4_4,
)
from kernels.scaled_matmul import block_scaled_matmul_kernel, nvfp4_block_scaled_matmul_kernel
from kernels.quantize_scaled_matmul import (
    mxfp8_quantize_kernel, nvfp4_quantize_kernel, qunatize_mxfp8_matmul_kernel,
    qunatize_nvfp4_matmul_kernel,
)
from cuda.tile._bytecode.version import BytecodeVersion


def _run_quantize_matmul_benchmark(shape, backend, benchmark, scaling_block_size, extra_args=(),
                                   atol=0.5, rtol=0.05):
    m, n, k = shape

    A = torch.rand((m, k), device='cuda:0')
    B = torch.rand((n, k), device='cuda:0')
    C = torch.zeros((m, n), dtype=torch.float32, device="cuda:0")
    args = (A, B, C, scaling_block_size) + extra_args

    ref = A @ B.T
    res = backend(*args)
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol)

    torch.cuda.synchronize()
    warmup_rounds, iterations, rounds = estimate_bench_iter(backend, args, cudagraph=True)
    benchmark.pedantic(
        backend, args,
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations, cudagraph=True
    )

    flop_count = 2 * m * n * k
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (A, B, C)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


@cache
def get_kernel(kernel, num_ctas):
    return kernel.replace_hints(num_ctas=num_ctas)


@pytest.fixture(params=[
    (1024, 1024, 1024),
    (8192, 8192, 8192),
    (12288, 4096, 2560),
], ids=shape_id)
def shape(request):
    return request.param

# =============================== MXFP8 Matmul =============================


@pytest.fixture(params=[
    "quantize_matmul_mxfp8_quantize_kernel",
    "quantize_matmul_mxfp8_host_quantize",
    "quantize_matmul_mxfp8_fused_kernel",
    ])
def MXFP8_backends(request):
    return getattr(request.module, request.param)


@require_blackwell_or_newer()
@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.benchmark(group='quantize_mma_mxfp8')
def bench_quantize_matmul_mxfp8(shape, MXFP8_backends, benchmark):
    # check _mma_scaled_supported_dtypes in _datatype.py for allowed mma scaled block sizes
    scaling_block_size = 32
    _run_quantize_matmul_benchmark(shape, MXFP8_backends, benchmark,
                                   scaling_block_size=scaling_block_size)


def quantize_matmul_mxfp8_quantize_kernel(A, B, C, scaling_block_size):
    kernel = get_kernel(block_scaled_matmul_kernel, num_ctas=1)
    tm, tn, tk = 128, 128, 128
    tm_q, tn_q, tk_q = 128, 128, 128
    m, n, k = A.shape[0], B.shape[0], A.shape[1]

    tkc = ct.cdiv(k, scaling_block_size)

    A_q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device='cuda:0')
    A_s_swizzled = torch.empty(
        (ct.cdiv(m, 32 * 4), ct.cdiv(tkc, 4), 32, 16),
        dtype=torch.float8_e8m0fnu, device='cuda:0'
    )
    grid_aq = (ct.cdiv(m, tm_q), ct.cdiv(k, tk_q))
    ct.launch(torch.cuda.current_stream(), grid_aq, mxfp8_quantize_kernel,
              (A, A_q, A_s_swizzled, tm_q, tk_q, scaling_block_size))

    B_q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device='cuda:0')
    B_s_swizzled = torch.empty(
        (ct.cdiv(n, 32 * 4), ct.cdiv(tkc, 4), 32, 16),
        dtype=torch.float8_e8m0fnu, device='cuda:0'
    )
    grid_bq = (ct.cdiv(n, tn_q), ct.cdiv(k, tk_q))
    ct.launch(torch.cuda.current_stream(), grid_bq, mxfp8_quantize_kernel,
              (B, B_q, B_s_swizzled, tn_q, tk_q, scaling_block_size))

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (A_q, A_s_swizzled, B_q, B_s_swizzled, C, tm, tn, tk, scaling_block_size))
    return C


def quantize_matmul_mxfp8_host_quantize(A, B, C, scaling_block_size):
    kernel = get_kernel(block_scaled_matmul_kernel, num_ctas=1)
    tm, tn, tk = 128, 128, 128
    m, n = A.shape[0], B.shape[0]

    A_q, A_s = block_quantize_f8e4m3fn_f8e8m0fnu(A, scaling_block_size)
    B_q, B_s = block_quantize_f8e4m3fn_f8e8m0fnu(B, scaling_block_size)
    A_s_swizzled = swizzle_32_4_4(A_s)
    B_s_swizzled = swizzle_32_4_4(B_s)

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (A_q, A_s_swizzled, B_q, B_s_swizzled, C, tm, tn, tk, scaling_block_size))
    return C


def quantize_matmul_mxfp8_fused_kernel(A, B, C, scaling_block_size):
    kernel = get_kernel(qunatize_mxfp8_matmul_kernel, num_ctas=1)
    tm, tn, tk = 128, 128, 128
    tm_q, tn_q, tk_q = 128, 128, 128
    m, n, k = A.shape[0], B.shape[0], A.shape[1]
    tkc = ct.cdiv(k, scaling_block_size)

    A_q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device='cuda:0')
    A_s_swizzled = torch.empty(
        (ct.cdiv(m, 32 * 4), ct.cdiv(tkc, 4), 32, 16),
        dtype=torch.float8_e8m0fnu, device='cuda:0'
    )
    B_q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device='cuda:0')
    B_s_swizzled = torch.empty(
        (ct.cdiv(n, 32 * 4), ct.cdiv(tkc, 4), 32, 16),
        dtype=torch.float8_e8m0fnu, device='cuda:0'
    )

    done = torch.zeros((1,), dtype=torch.int32, device='cuda:0')
    vbid = torch.zeros((1,), dtype=torch.int32, device='cuda:0')

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (A, A_q, A_s_swizzled, B, B_q, B_s_swizzled, C,
               vbid, done, tm, tn, tk, tm_q, tn_q, tk_q, scaling_block_size))
    return C


# =============================== NVFP4 Matmul =============================


@pytest.fixture(params=[
    "quantize_matmul_nvfp4_quantize_kernel",
    "quantize_matmul_nvfp4_host_quantize",
    "quantize_matmul_nvfp4_fused_kernel",
    ])
def NVFP4_backends(request):
    return getattr(request.module, request.param)


@require_blackwell_or_newer()
@requires_tileiras(BytecodeVersion.V_13_4)
@pytest.mark.benchmark(group='quantize_mma_nvfp4')
def bench_quantize_matmul_nvfp4(shape, NVFP4_backends, benchmark):
    # check _mma_scaled_supported_dtypes in _datatype.py for allowed mma scaled block sizes
    scaling_block_size = 16
    _run_quantize_matmul_benchmark(shape, NVFP4_backends, benchmark,
                                   scaling_block_size=scaling_block_size)


def quantize_matmul_nvfp4_host_quantize(A, B, C, scaling_block_size):
    kernel = get_kernel(nvfp4_block_scaled_matmul_kernel, num_ctas=1)
    tm, tn, tk = 128, 128, 128
    m, n = A.shape[0], B.shape[0]

    A_q, A_s, A_g = block_quantize_f4e2m1fn_f8e4m3fn(A, scaling_block_size)
    B_q, B_s, B_g = block_quantize_f4e2m1fn_f8e4m3fn(B, scaling_block_size)

    A_s_swizzled = swizzle_32_4_4(A_s)
    B_s_swizzled = swizzle_32_4_4(B_s)

    A_q = A_q.view(torch.uint8)
    B_q = B_q.view(torch.uint8)

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (A_q, A_s_swizzled, A_g, B_q, B_s_swizzled, B_g, C, tm, tn, tk, scaling_block_size))
    return C


def quantize_matmul_nvfp4_quantize_kernel(A, B, C, scaling_block_size):
    kernel = get_kernel(nvfp4_block_scaled_matmul_kernel, num_ctas=1)
    tm, tn, tk = 128, 128, 128
    tm_q, tn_q, tk_q = 128, 128, 128
    m, n, k = A.shape[0], B.shape[0], A.shape[1]
    tkc = ct.cdiv(k, scaling_block_size)

    done = torch.zeros((1,), dtype=torch.int32, device='cuda:0')
    vbid = torch.zeros((1,), dtype=torch.int32, device='cuda:0')

    A_gmax_bits = torch.zeros((1,), dtype=torch.uint32, device='cuda:0')
    A_g_scale = torch.zeros((1,), dtype=torch.float32, device='cuda:0')
    A_q = torch.empty((m, k // 2), dtype=torch.uint8, device='cuda:0')
    A_s_swizzled = torch.empty(
        (ct.cdiv(m, 32 * 4), ct.cdiv(tkc, 4), 32, 16),
        dtype=torch.float8_e4m3fn, device='cuda:0'
    )
    grid_aq = (ct.cdiv(m, tm_q), ct.cdiv(k, tk_q))
    ct.launch(torch.cuda.current_stream(), grid_aq, nvfp4_quantize_kernel,
              (A, A_q, A_s_swizzled, A_g_scale, A_gmax_bits,
               tm_q, tk_q, scaling_block_size, done, vbid))

    done = torch.zeros((1,), dtype=torch.int32, device='cuda:0')
    vbid = torch.zeros((1,), dtype=torch.int32, device='cuda:0')
    B_gmax_bits = torch.zeros((1,), dtype=torch.uint32, device='cuda:0')
    B_g_scale = torch.zeros((1,), dtype=torch.float32, device='cuda:0')

    B_q = torch.empty((n, k // 2), dtype=torch.uint8, device='cuda:0')
    B_s_swizzled = torch.empty(
        (ct.cdiv(n, 32 * 4), ct.cdiv(tkc, 4), 32, 16),
        dtype=torch.float8_e4m3fn, device='cuda:0'
    )
    grid_bq = (ct.cdiv(n, tn_q), ct.cdiv(k, tk_q))
    ct.launch(torch.cuda.current_stream(), grid_bq, nvfp4_quantize_kernel,
              (B, B_q, B_s_swizzled, B_g_scale, B_gmax_bits,
               tn_q, tk_q, scaling_block_size, done, vbid))

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (A_q, A_s_swizzled, A_g_scale, B_q, B_s_swizzled, B_g_scale, C,
               tm, tn, tk, scaling_block_size))
    return C


def quantize_matmul_nvfp4_fused_kernel(A, B, C, scaling_block_size):
    kernel = get_kernel(qunatize_nvfp4_matmul_kernel, num_ctas=1)
    tm, tn, tk = 128, 128, 128
    tm_q, tn_q, tk_q = 128, 128, 128
    m, n, k = A.shape[0], B.shape[0], A.shape[1]
    tkc = ct.cdiv(k, scaling_block_size)

    A_q = torch.empty((m, k // 2), dtype=torch.uint8, device='cuda:0')
    A_s_swizzled = torch.empty(
        (ct.cdiv(m, 32 * 4), ct.cdiv(tkc, 4), 32, 16),
        dtype=torch.float8_e4m3fn, device='cuda:0'
    )
    A_gmax_bits = torch.zeros((1,), dtype=torch.uint32, device='cuda:0')

    B_q = torch.empty((n, k // 2), dtype=torch.uint8, device='cuda:0')
    B_s_swizzled = torch.empty(
        (ct.cdiv(n, 32 * 4), ct.cdiv(tkc, 4), 32, 16),
        dtype=torch.float8_e4m3fn, device='cuda:0'
    )
    B_gmax_bits = torch.zeros((1,), dtype=torch.uint32, device='cuda:0')

    done_1 = torch.zeros((1,), dtype=torch.int32, device='cuda:0')
    vbid_1 = torch.zeros((1,), dtype=torch.int32, device='cuda:0')
    done_2 = torch.zeros((1,), dtype=torch.int32, device='cuda:0')
    vbid_2 = torch.zeros((1,), dtype=torch.int32, device='cuda:0')

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (A, A_q, A_s_swizzled, A_gmax_bits, B, B_q, B_s_swizzled, B_gmax_bits, C,
               vbid_1, done_1, vbid_2, done_2,
               tm, tn, tk, tm_q, tn_q, tk_q, scaling_block_size))
    return C
