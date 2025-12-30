# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from math import ceil, sqrt
from conftest import dtype_id, shape_id
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
import cuda.tile as ct

import pytest
import torch

from util import estimate_bench_iter
from kernels.attention import fmha_kernel, fmha_bwd_dq_kernel, fmha_bwd_dk_dv_kernel, fmha_bwd_preprocess_kernel
has_backward = True

def qkv_id(qkv_shape: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    q_shape, kv_shape = qkv_shape
    if q_shape[2] == 1:
        prefix = "decode-"
    else:
        prefix = "prefill-"
    b, q_head, q_len, d = q_shape
    _, k_head, k_len, _ = kv_shape
    return prefix + shape_id((b, q_head, k_head, q_len, k_len, d))


@pytest.fixture(
    params=[
        # B, H, L, D
       ((6, 24, 67000, 64), (6, 24, 67000, 64)), # tanlan dim 64
       ((6, 24, 67000, 128), (6, 24, 67000, 128)), # tanlan dim 128
    #    ((6, 32, 1024, 128), (6, 32, 1024, 128)),  # prefill
    #    ((1, 32, 1024, 64), (1, 32, 1024, 64)),
    #    ((1, 32, 1024, 64), (1, 8, 1024, 64)),  # prefill + gqa
    #    ((1, 32, 1024, 128), (1, 8, 1024, 128)),  # prefill + gqa
    #    ((1, 32, 8192, 64), (1, 32, 8192, 64)),
    #    ((1, 32, 8192, 128), (1, 32, 8192, 128)),
    #    ((1, 32, 8192, 128), (1, 8, 8192, 128)),  # prefill + gqa
    #    ((1, 32, 1, 128), (1, 32, 1024, 128)),  # decode
    #    ((1, 32, 1, 64), (1, 32, 1024, 64)),  # decode
    #    ((8, 32, 1, 128), (8, 32, 1024, 128)),  # decode
    #    ((1, 32, 1, 128), (1, 8, 1024, 128)),  # decode + gqa
    #    ((8, 32, 1, 128), (8, 8, 1024, 128)),  # decode + gqa
    ],
    ids=qkv_id)
def qkv_shape(request):
    return request.param


@pytest.fixture(params=[torch.bfloat16], ids=dtype_id)
def dtype(request):
    return request.param


@pytest.mark.benchmark(group='attention')
def bench_fmha(qkv_shape, dtype, backend, benchmark):
    q_shape, kv_shape = qkv_shape

    q = torch.randn(q_shape, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(kv_shape, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(kv_shape, dtype=dtype, device='cuda', requires_grad=True)
    lse = torch.randn(q_shape[:-1], dtype=torch.float32, device='cuda')
    grad = torch.randn(q_shape, dtype=dtype, device='cuda', requires_grad=True)
    o = torch.zeros_like(q)
    ref = torch.zeros_like(q)
    is_causal = q_shape[2] == kv_shape[2]
    enable_gqa = q_shape[1] != kv_shape[1]

    if has_backward:
        dq, dk, dv = backend(q, k, v, o, grad, lse, is_causal, enable_gqa)
        dq_ref, dk_ref, dv_ref = ref_fmha(q, k, v, ref, grad, is_causal, enable_gqa)
        torch.testing.assert_close(o, ref, atol=1e-2, rtol=5e-2)
        torch.testing.assert_close(dq, dq_ref, atol=6e-2, rtol=5e-2)
        # torch.testing.assert_close(dk, dk_ref, atol=6e-2, rtol=5e-2)
        # torch.testing.assert_close(dv, dv_ref, atol=6e-2, rtol=5e-2)
    else:
        backend(q, k, v, o, grad, lse, is_causal, enable_gqa)
        ref_fmha(q, k, v, ref, grad, is_causal, enable_gqa)
        torch.testing.assert_close(o, ref, atol=1e-2, rtol=5e-2)
    torch.cuda.synchronize()

    warmup_rounds, iterations, rounds = estimate_bench_iter(
        backend, (q, k, v, o, grad, lse, is_causal, enable_gqa),
    )

    benchmark.pedantic(
        backend, (q, k, v, o, grad, lse, is_causal, enable_gqa),
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
    )

    B, H, L, D = q.shape
    # first gemm mma(q, k): 2 * B * H * L * L * D
    # second gemm mma(p, v): 2 * B * H * L * L * D
    flop_count = 4 * B * H * L * L * D

    if is_causal:
        flop_count /= 2

    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (q, k, v, o)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def cutile_fmha(q, k, v, o, grad, lse, is_causal, enable_gqa):
    b, qh, q_len, d = q.shape
    _, kh, k_len, _ = k.shape
    qk_scale = 1 / sqrt(d)
    TILE_M, TILE_N = (256, 128) if is_causal else (64, 128)
    query_group_size = qh // kh
    grid = (ceil(q_len / TILE_M), b * qh, 1)
    input_pos = 0 if q_len == k_len else (k_len - 1)
    EVEN_K = (k_len % TILE_N) == 0
    ct.launch(torch.cuda.current_stream(), grid, fmha_kernel,
              (q, k, v, o, lse,
               qk_scale, input_pos, d, qh, TILE_M, TILE_N,
               query_group_size, is_causal, EVEN_K))
    if has_backward:
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        delta = torch.zeros_like(lse)
        ct.launch(torch.cuda.current_stream(), (ceil(q_len / TILE_M), b * qh, 1), 
                  fmha_bwd_preprocess_kernel,
                  (o, grad, delta, qh, TILE_M, d))
        ct.launch(torch.cuda.current_stream(), (ceil(k_len / TILE_N), b * qh, 1), 
                  fmha_bwd_dk_dv_kernel,
                  (q, k, v, grad, delta, lse, dk, dv,
                   qk_scale, input_pos, d, qh, TILE_M, TILE_N,
                   query_group_size, is_causal, EVEN_K))
        ct.launch(torch.cuda.current_stream(), (ceil(q_len / TILE_M), b * qh, 1), 
                  fmha_bwd_dq_kernel,
                  (q, k, v, grad, delta, lse, dq,
                   qk_scale, input_pos, d, qh, TILE_M, TILE_N,
                   query_group_size, is_causal, EVEN_K))
        return dq, dk, dv

def torch_fmha(q, k, v, o, grad, lse, is_causal, enable_gqa):
    backend = SDPBackend.CUDNN_ATTENTION \
            if (q.shape[2] == k.shape[2]) \
            else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(backend):
        ret = scaled_dot_product_attention(q, k, v,
                                           is_causal=is_causal,
                                           enable_gqa=enable_gqa)
        o.copy_(ret)
        if has_backward:
            ret.backward(grad, retain_graph=True)
            dq, dk, dv = q.grad, k.grad, v.grad
            return dq, dk, dv


def ref_fmha(q, k, v, o, grad, is_causal, enable_gqa):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ret = scaled_dot_product_attention(q, k, v,
                                           is_causal=is_causal,
                                           enable_gqa=enable_gqa)
        o.copy_(ret)
        if has_backward:
            ret.backward(grad, retain_graph=True)
            dq, dk, dv = q.grad, k.grad, v.grad
            return dq, dk, dv