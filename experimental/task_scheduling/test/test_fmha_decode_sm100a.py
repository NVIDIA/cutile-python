# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end paged decode tests with runtime ragged Q and KV bounds.

Set CUDA_LANG_DECODE_EXHAUSTIVE=1 to run all 164 cases instead of the 56 CI cases.
"""

import importlib
import os
from dataclasses import replace
from itertools import product

import pytest
import torch
from task_scheduling_test_utils import require_blackwell_cc100

_PACKAGE = "experimental.task_scheduling.tutorial.08_fmha_decode_sm100a"


def pytest_generate_tests(metafunc):
    # Each distinct configuration compiles a large task-scheduled kernel. Cover
    # every parameter value in CI; retain the full cross product for longer runs.
    name = metafunc.function.__name__
    dtypes = ("float16", "bfloat16")
    if name == "test_paged_decode":
        names = "decode_case,page,heads_q,heads_kv,persistent"
        full = (
            (dtype, page, hq, hkv, persistent)
            for dtype, page, (hq, hkv), persistent in product(
                dtypes, (16, 32, 64, 128), ((2, 2), (8, 2), (8, 1), (32, 1)),
                (False, True),
            )
        )
        cases = (
            ("float16", 16, 2, 2, False),
            ("bfloat16", 16, 8, 1, True),
            ("bfloat16", 32, 8, 2, False),
            ("float16", 32, 32, 1, True),
            ("float16", 64, 8, 1, False),
            ("bfloat16", 64, 2, 2, True),
            ("bfloat16", 128, 32, 1, False),
            ("float16", 128, 8, 2, True),
        )
    elif name in ("test_paged_decode_causal_window", "test_packed_query_causal_window"):
        names = "decode_case,window,persistent"
        first_window = -1 if name == "test_paged_decode_causal_window" else 1
        full = product(dtypes, (first_window, 0, 31, 128, 256), (False, True))
        cases = (
            ("float16", first_window, False),
            ("float16", 0, False),
            ("bfloat16", 0, True),
            ("float16", 31, True),
            ("bfloat16", 128, False),
            ("bfloat16", 256, True),
        )
    elif name == "test_packed_query_decode":
        names = "decode_case,page,ratio,grouped,persistent,mask"
        layouts = (
            (16, 1, True, False), (32, 2, True, True),
            (64, 4, True, False), (128, 8, True, True),
            (16, 3, False, False), (32, 10, False, True),
            (64, 32, False, False), (128, 4, False, True),
        )
        full = (
            (dtype, *layout, mask)
            for dtype, layout, mask in product(dtypes, layouts, ("dense", "causal"))
        )
        cases = (
            ("float16", 16, 1, True, False, "dense"),
            ("bfloat16", 32, 2, True, True, "causal"),
            ("bfloat16", 64, 4, True, False, "dense"),
            ("float16", 128, 8, True, True, "causal"),
            ("bfloat16", 16, 3, False, False, "causal"),
            ("float16", 32, 10, False, True, "dense"),
            ("float16", 64, 32, False, False, "causal"),
            ("bfloat16", 128, 4, False, True, "dense"),
        )
    else:
        if "decode_case" in metafunc.fixturenames:
            metafunc.parametrize("decode_case", dtypes, indirect=True, ids=("fp16", "bf16"))
        return
    if os.environ.get("CUDA_LANG_DECODE_EXHAUSTIVE") == "1":
        cases = tuple(full)
    ids = [
        "-".join(("fp16" if case[0] == "float16" else "bf16", *(str(v) for v in case[1:])))
        for case in cases
    ]
    metafunc.parametrize(names, cases, indirect=["decode_case"], ids=ids)


@pytest.fixture
def decode_case(request):
    runner = importlib.import_module(_PACKAGE + ".fmha_decode_run")
    dtype = getattr(runner.cl, request.param)
    cfg = runner.FmhaDecodeConfig(q_dtype=dtype, kv_dtype=dtype, out_dtype=dtype)
    return runner, cfg


@require_blackwell_cc100()
def test_paged_decode(page, heads_q, heads_kv, persistent, decode_case):
    runner, cfg = decode_case
    cfg = replace(cfg, num_tokens_per_page=page, use_persistent_scheduler=persistent)
    tensors = runner.prepare_tensors(
        (1, 17, 127, 128, 129, 255, 256, 257, 1023), heads_q, heads_kv, cfg
    )
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
    )


@require_blackwell_cc100()
def test_paged_decode_causal_window(window, persistent, decode_case):
    runner, cfg = decode_case
    cfg = replace(
        cfg, mask_type="causal", window_left=window, use_persistent_scheduler=persistent
    )
    tensors = runner.prepare_tensors((1, 129, 257, 8192), 8, 1, cfg)
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
    )


@require_blackwell_cc100()
def test_paged_decode_runtime_metadata(decode_case):
    runner, cfg = decode_case
    tensors = runner.prepare_tensors((8192,) * 4, cfg=cfg)
    runner.run(tensors, cfg)
    tensors["seq_lens"].copy_(torch.tensor((1, 129, 1025, 8191), device="cuda", dtype=torch.int32))
    tensors["paged_kv_indptr"].copy_(
        torch.tensor((0, 1, 6, 39, 295), device="cuda", dtype=torch.int32)
    )
    tensors["paged_kv_indices"].copy_(tensors["paged_kv_indices"].flip(0))
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
    )


@pytest.mark.parametrize("head_ratio", (3, 10))
@require_blackwell_cc100()
def test_paged_decode_partial_head_group(head_ratio, decode_case):
    runner, cfg = decode_case
    tensors = runner.prepare_tensors((129, 1024), head_ratio * 2, 2, cfg)
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
    )


@pytest.mark.parametrize("grouped", (False, True))
@require_blackwell_cc100()
def test_paged_decode_clc_recycles_requests(grouped, decode_case):
    runner, cfg = decode_case
    cfg = replace(
        cfg, mask_type="causal", use_persistent_scheduler=True, groups_tokens_heads_q=grouped
    )
    # More than three waves, with different request lengths on recycled CTAs.
    lengths = tuple((1, 129, 1025, 8192)[i % 4] for i in range(64))
    tensors = runner.prepare_tensors(lengths, 32, 8, cfg)
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        runner.run(tensors, cfg)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
    )

    # Replay the same compiled graph with different CSR boundaries and page IDs.
    lengths = torch.tensor(
        tuple((257, 1, 513, 33)[i % 4] for i in range(64)), device="cuda", dtype=torch.int32
    )
    tensors["seq_lens"].copy_(lengths)
    tensors["paged_kv_indptr"][1:].copy_(((lengths + 31) // 32).cumsum(0))
    tensors["paged_kv_indices"].copy_(tensors["paged_kv_indices"].flip(0))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
    )


@pytest.mark.parametrize("head_ratio", (1, 2, 8))
@require_blackwell_cc100()
def test_paged_decode_grouped_q(head_ratio, decode_case):
    runner, cfg = decode_case
    cfg = replace(
        cfg, use_persistent_scheduler=True, groups_tokens_heads_q=True
    )
    tensors = runner.prepare_tensors((1, 129, 1025), head_ratio * 2, 2, cfg)
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
    )


@require_blackwell_cc100()
def test_packed_query_decode(page, ratio, grouped, persistent, mask, decode_case):
    runner, cfg = decode_case
    cfg = replace(
        cfg, use_variable_seqlens_q=True, max_seq_len_q=17, num_tokens_per_page=page,
        groups_tokens_heads_q=grouped, use_persistent_scheduler=persistent, mask_type=mask,
    )
    tensors = runner.prepare_tensors(
        (1, 129, 255, 256, 257, 513), ratio * 2, 2, cfg, q_lengths=(1, 3, 7, 8, 9, 17)
    )
    tensors["o"].fill_(float("nan"))
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
    )


@require_blackwell_cc100()
def test_packed_query_causal_window(window, persistent, decode_case):
    runner, cfg = decode_case
    cfg = replace(
        cfg, use_variable_seqlens_q=True, max_seq_len_q=17, groups_tokens_heads_q=True,
        use_persistent_scheduler=persistent, mask_type="causal", window_left=window,
    )
    tensors = runner.prepare_tensors(
        (1, 129, 257, 8192), 4, 2, cfg, q_lengths=(1, 3, 9, 17)
    )
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
    )


@pytest.mark.parametrize("persistent", (False, True))
@pytest.mark.parametrize("grouped", (False, True))
@require_blackwell_cc100()
def test_packed_query_graph_reloads_metadata(grouped, persistent, decode_case):
    runner, cfg = decode_case
    cfg = replace(
        cfg, use_variable_seqlens_q=True, max_seq_len_q=9, groups_tokens_heads_q=grouped,
        use_persistent_scheduler=persistent, mask_type="causal", window_left=31,
    )
    # Active and inactive tiles cross multiple resident waves and rotate on replay.
    lengths = tuple((33, 129, 257, 513)[i % 4] for i in range(24))
    q_lengths = tuple((1, 3, 7, 9)[i % 4] for i in range(24))
    ratio = 4 if grouped else 10
    tensors = runner.prepare_tensors(lengths, ratio * 4, 4, cfg, q_lengths=q_lengths)
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.run(tensors, cfg)
    for replay in range(2):
        if replay:
            new_q_lengths = torch.tensor(q_lengths[::-1], device="cuda", dtype=torch.int32)
            tensors["qo_indptr"][1:].copy_(new_q_lengths.cumsum(0))
            new_lengths = torch.tensor(lengths[::-1], device="cuda", dtype=torch.int32)
            tensors["seq_lens"].copy_(new_lengths)
            tensors["paged_kv_indptr"][1:].copy_(((new_lengths + 31) // 32).cumsum(0))
            tensors["paged_kv_indices"].copy_(tensors["paged_kv_indices"].flip(0))
        tensors["o"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            tensors["o"], runner.torch_reference(tensors, cfg), atol=0.003, rtol=0.02
        )


@pytest.mark.parametrize("persistent", (False, True))
@require_blackwell_cc100()
def test_packed_query_window_zero_selects_exact_key(persistent, decode_case):
    runner, cfg = decode_case
    cfg = replace(
        cfg, use_variable_seqlens_q=True, max_seq_len_q=9, groups_tokens_heads_q=True,
        use_persistent_scheduler=persistent, mask_type="causal", window_left=0,
    )
    lengths, q_lengths = (17, 129, 257, 33), (1, 7, 3, 9)
    tensors = runner.prepare_tensors(lengths, 4, 2, cfg, q_lengths=q_lengths)
    tensors["q"].zero_()
    tensors["k"].zero_()
    tensors["v"].fill_(float("nan"))
    expected = torch.empty_like(tensors["o"])
    q_begin = 0
    for batch, (length, q_length) in enumerate(zip(lengths, q_lengths)):
        begin, end = tensors["paged_kv_indptr"][batch:batch + 2].tolist()
        ids = tensors["paged_kv_indices"][begin:end].long()
        # Unique keys make an off-by-one query/causal offset observable exactly.
        values = torch.arange((end - begin) * 32, device="cuda", dtype=torch.float32)
        values = values + batch * 16
        tensors["v"][ids] = values.reshape(-1, 1, 32, 1).to(tensors["v"].dtype)
        expected[q_begin:q_begin + q_length] = values[length - q_length:length, None, None]
        q_begin += q_length
    runner.run(tensors, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(tensors["o"], expected, atol=0, rtol=0)
