# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Paged HND decode launch for SQ1 or packed Q and a PyTorch reference."""

import itertools

import cuda.lang as cl
import torch

from .fmha_decode_config import FmhaDecodeConfig, validate_config
from .fmha_decode_kernel import make_fmha_decode_launcher


def prepare_tensors(
    lengths, heads_q=8, heads_kv=1, cfg=FmhaDecodeConfig(), seed=1111, *, q_lengths=None
):
    validate_config(cfg)
    if not lengths or min(lengths) < 1:
        raise ValueError("decode requires at least one K/V token per request")
    if cfg.use_variable_seqlens_q:
        if q_lengths is None or len(q_lengths) != len(lengths):
            raise ValueError("packed Q requires one query length per request")
        if min(q_lengths) < 1 or max(q_lengths) > cfg.max_seq_len_q:
            raise ValueError("query lengths must be in [1,max_seq_len_q]")
        if cfg.mask_type == "causal" and any(q > k for q, k in zip(q_lengths, lengths)):
            raise ValueError("causal attention requires q_len <= kv_len for each request")
    elif q_lengths is not None:
        raise ValueError("q_lengths requires use_variable_seqlens_q=True")
    torch.manual_seed(seed)
    page = cfg.num_tokens_per_page
    counts = [(n + page - 1) // page for n in lengths]
    shape = (sum(counts), heads_kv, page, 128)
    dtype = torch.bfloat16 if cfg.q_dtype == cl.bfloat16 else torch.float16
    total_q = sum(q_lengths) if cfg.use_variable_seqlens_q else len(lengths)
    tensors = dict(
        q=(torch.randn(total_q, heads_q, 128, device="cuda") * 0.2).to(dtype),
        k=(torch.randn(shape, device="cuda") * 0.2).to(dtype),
        v=(torch.randn(shape, device="cuda") * 0.2).to(dtype),
        o=torch.empty(total_q, heads_q, 128, device="cuda", dtype=dtype),
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        paged_kv_indptr=torch.tensor(
            [0] + list(itertools.accumulate(counts)), device="cuda", dtype=torch.int32
        ),
        paged_kv_indices=torch.arange(sum(counts) - 1, -1, -1, device="cuda", dtype=torch.int32),
    )
    if cfg.use_variable_seqlens_q:
        tensors["qo_indptr"] = torch.tensor(
            [0] + list(itertools.accumulate(q_lengths)), device="cuda", dtype=torch.int32
        )
    return tensors


def run(tensors, cfg=FmhaDecodeConfig(), stream=None):
    validate_config(cfg)
    q, k, v, o = (tensors[name] for name in ("q", "k", "v", "o"))
    dtype = torch.bfloat16 if cfg.q_dtype == cl.bfloat16 else torch.float16
    if any(
        t.dtype != dtype or not t.is_cuda or not t.is_contiguous() for t in (q, k, v, o)
    ):
        raise ValueError(f"Q/K/V/O must be contiguous CUDA tensors with config dtype {dtype}")
    if q.ndim != 3 or k.ndim != 4 or q.shape[-1] != 128 or k.shape[-1] != 128:
        raise ValueError("expected Q[B or total_q,Hq,128] and K/V[pages,Hkv,page,128]")
    if min(q.shape) < 1 or min(k.shape) < 1:
        raise ValueError("Q and K/V tensor extents must be positive")
    if v.shape != k.shape or o.shape != q.shape or k.shape[2] != cfg.num_tokens_per_page:
        raise ValueError("inconsistent Q/K/V/O geometry")
    if q.shape[1] % k.shape[1] or not 1 <= q.shape[1] // k.shape[1] <= 32:
        raise ValueError("Hq/Hkv must be an integer in [1,32]")
    metadata = tuple(tensors[name] for name in ("seq_lens", "paged_kv_indptr", "paged_kv_indices"))
    if cfg.use_variable_seqlens_q:
        if "qo_indptr" not in tensors:
            raise ValueError("packed Q requires qo_indptr")
        metadata += (tensors["qo_indptr"],)
    elif "qo_indptr" in tensors:
        raise ValueError("qo_indptr requires use_variable_seqlens_q=True")
    if any(t.device != q.device for t in (k, v, o, *metadata)):
        raise ValueError("all tensors must reside on the same CUDA device")
    if any(t.dtype != torch.int32 or t.ndim != 1 or not t.is_contiguous() for t in metadata):
        raise ValueError("metadata must be contiguous one-dimensional int32 tensors")
    batch = metadata[0].numel()
    if batch < 1 or metadata[1].numel() != batch + 1:
        raise ValueError("sequence lengths and page indptr must match the request batch")
    if cfg.use_variable_seqlens_q:
        if metadata[3].numel() != batch + 1:
            raise ValueError("qo_indptr must have B+1 entries")
        if not batch <= q.shape[0] <= batch * cfg.max_seq_len_q:
            raise ValueError("packed query extent must be between B and B*max_seq_len_q")
    elif batch != q.shape[0]:
        raise ValueError("sequence lengths must match the SQ1 query batch")
    if batch * cfg.max_seq_len_q * q.shape[1] > 2147483647:
        raise ValueError("planned query/head extent must fit in int32")
    ratio = q.shape[1] // k.shape[1]
    launcher = make_fmha_decode_launcher(ratio, k.shape[1], cfg, batch=batch)
    args = tuple(
        tensors[name]
        for name in ("q", "k", "v", "o", "seq_lens", "paged_kv_indptr", "paged_kv_indices")
    )
    if cfg.use_variable_seqlens_q:
        args += (tensors["qo_indptr"],)
    launcher(torch.cuda.current_stream() if stream is None else stream, *args)
    return o


def torch_reference(tensors, cfg=FmhaDecodeConfig()):
    q, k, v = (tensors[name] for name in ("q", "k", "v"))
    result = torch.empty_like(q)
    indptr = tensors["paged_kv_indptr"].tolist()
    ratio = q.shape[1] // k.shape[1]
    q_offsets = (
        tensors["qo_indptr"].tolist() if cfg.use_variable_seqlens_q
        else list(range(q.shape[0] + 1))
    )
    for batch, length in enumerate(tensors["seq_lens"].tolist()):
        ids = tensors["paged_kv_indices"][indptr[batch]:indptr[batch + 1]].long()
        q_begin, q_end = q_offsets[batch:batch + 2]
        q_length = q_end - q_begin
        keys = torch.arange(length, device=q.device)
        rows = torch.arange(q_length, device=q.device)
        upper = length - q_length + rows + 1
        visible = torch.ones(q_length, length, device=q.device, dtype=torch.bool)
        if cfg.mask_type == "causal":
            visible &= keys[None, :] < upper[:, None]
            if cfg.window_left >= 0:
                visible &= keys[None, :] >= upper[:, None] - cfg.window_left - 1
        for head in range(k.shape[1]):
            kk = k[ids, head].reshape(-1, 128)[:length].float()
            vv = v[ids, head].reshape(-1, 128)[:length].float()
            qq = q[q_begin:q_end, head * ratio:(head + 1) * ratio].float()
            scores = qq @ kk.T / 128**0.5
            scores.masked_fill_(~visible[:, None, :], float("-inf"))
            result[q_begin:q_end, head * ratio:(head + 1) * ratio] = (
                torch.softmax(scores, -1) @ vv
            ).to(q.dtype)
    return result
