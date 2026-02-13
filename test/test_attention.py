# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math

import cuda.tile as ct
from kernels.attention import fmha_kernel


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("q_heads", [8, 16])
@pytest.mark.parametrize("k_heads", [8])
@pytest.mark.parametrize("q_len", [1, 15, 32])
@pytest.mark.parametrize("k_len", [32, 63])
@pytest.mark.parametrize("head_dim", [32])
@pytest.mark.parametrize("tile_size", [(8, 16)])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("use_input_pos", [True, False])
def test_flash_attention(batch_size, q_heads, k_heads,
                         q_len, k_len,
                         head_dim, tile_size, is_causal,
                         use_input_pos,
                         float_dtype):
    query_group_size = q_heads // k_heads
    TILE_M, TILE_N = tile_size
    qk_scale = 1 / math.sqrt(head_dim)
    q = torch.randn((batch_size, q_heads, q_len, head_dim), dtype=float_dtype, device='cuda')
    k = torch.randn((batch_size, k_heads, k_len, head_dim), dtype=float_dtype, device='cuda')
    v = torch.randn((batch_size, k_heads, k_len, head_dim), dtype=float_dtype, device='cuda')
    o = torch.zeros_like(q)
    grid = (math.ceil(q_len / TILE_M), batch_size * q_heads, 1)
    if use_input_pos:
        # encode input position for q
        # for decoding kernel the starting position q is determined by k_len - 1
        input_pos = k_len - 1
    else:
        input_pos = 0
    EVEN_K = (k_len % TILE_N) == 0
    ct.launch(torch.cuda.current_stream(), grid, fmha_kernel,
              (q, k, v, o,
               qk_scale,
               input_pos,
               head_dim, head_dim, q_heads,
               TILE_M, TILE_N,
               query_group_size, is_causal, EVEN_K))
    if is_causal:
        mask = (input_pos + torch.arange(q_len)[:, None]) >= torch.arange(k_len)[None, :]
        mask = torch.where(mask, 0.0, -math.inf).to(float_dtype).to('cuda')
    else:
        mask = None
    ref_result = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                  attn_mask=mask,
                                                                  scale=qk_scale,
                                                                  enable_gqa=(q_heads != k_heads))
    if float_dtype == torch.float32:
        atol, rtol = 5e-5, 5e-4
    elif float_dtype == torch.float16:
        atol, rtol = 1e-3, 5e-3
    elif float_dtype == torch.bfloat16:
        atol, rtol = 1e-2, 5e-2
    torch.testing.assert_close(o, ref_result, atol=atol, rtol=rtol)
