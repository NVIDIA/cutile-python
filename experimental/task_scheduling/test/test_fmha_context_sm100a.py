# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Kernel tests for SM100a query/head-paired FMHA task scheduling."""

import importlib

import pytest
import torch

from task_scheduling_test_utils import require_blackwell_cc100


_PACKAGE = "experimental.task_scheduling.tutorial.07_fmha_context_sm100a"


def _module(name):
    return importlib.import_module(f"{_PACKAGE}.{name}")


def require_available_sm100():
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="requires a Blackwell CC 10.x GPU")
    return require_blackwell_cc100()


@pytest.mark.parametrize(
    ("scheduler", "q_shape", "k_shape"),
    [
        pytest.param(
            scheduler,
            q_shape,
            k_shape,
            id=f"{scheduler}-{case}",
        )
        for scheduler in (
            "direct",
            "static_persistent",
            "clc_dynamic_persistent",
        )
        for q_shape, k_shape, case in (
            ((1, 256, 1, 128), (1, 128, 1, 128), "n1-head-tail"),
            ((1, 512, 2, 128), (1, 512, 2, 128), "n4-steady-state"),
        )
    ],
)
@require_available_sm100()
def test_fmha_d128_accuracy(scheduler, q_shape, k_shape):
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(scheduler=scheduler)
    tensors = run.prepare_tensors(q_shape, k_shape, cfg=cfg)
    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors)


@pytest.mark.parametrize("scheduler", ("static_persistent", "clc_dynamic_persistent"))
@require_available_sm100()
def test_fmha_d128_head_paired_gqa_accuracy(scheduler):
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(
        scheduler=scheduler,
        head_paired=True,
        h_r=4,
    )
    tensors = run.prepare_tensors(
        (1, 256, 8, 128),
        (1, 256, 2, 128),
        cfg=cfg,
        seed=20260905,
    )
    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors, cfg=cfg)


@pytest.mark.parametrize(
    ("scheduler", "q_length", "k_length"),
    [
        pytest.param("direct", 128, 128, id="direct-single-kv-tile"),
        pytest.param("static_persistent", 512, 512, id="static-triangular"),
        pytest.param("clc_dynamic_persistent", 513, 769, id="clc-bottom-right-tail"),
    ],
)
@require_available_sm100()
def test_fmha_d128_query_paired_causal_accuracy(
    scheduler,
    q_length,
    k_length,
):
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(scheduler=scheduler, is_causal=True)
    tensors = run.prepare_tensors(
        (1, q_length, 2, 128),
        (1, k_length, 2, 128),
        cfg=cfg,
        seed=20260906,
    )
    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors, cfg=cfg)


@pytest.mark.parametrize(
    ("window_size_left", "q_length", "k_length"),
    (
        pytest.param(0, 385, 641, id="causal-bottom-right"),
        pytest.param(127, 385, 641, id="window-127-bottom-right"),
        pytest.param(256, 385, 641, id="window-256-bottom-right"),
        pytest.param(256, 512, 512, id="window-256-equal-length"),
    ),
)
@require_available_sm100()
def test_fmha_d128_head_paired_causal_window_accuracy(
    window_size_left,
    q_length,
    k_length,
):
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(
        scheduler="clc_dynamic_persistent",
        is_causal=True,
        head_paired=True,
        h_r=4,
        window_size_left=window_size_left,
    )
    tensors = run.prepare_tensors(
        (1, q_length, 8, 128),
        (1, k_length, 2, 128),
        cfg=cfg,
        seed=20260907 + window_size_left,
    )
    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors, cfg=cfg)


@require_available_sm100()
def test_fmha_d128_packed_ragged_accuracy():
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(
        scheduler="clc_dynamic_persistent",
        has_varlen=True,
    )
    tensors = run.prepare_ragged_tensors(
        (1, 257),
        (129, 257),
        heads=2,
        cfg=cfg,
        seed=20260901,
    )
    tensors["scale"].mul_(0.75)
    tensors["output_scale"].fill_(0.5)

    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors, cfg=cfg)


@require_available_sm100()
def test_fmha_d128_packed_ragged_head_paired_gqa_accuracy():
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(
        scheduler="clc_dynamic_persistent",
        has_varlen=True,
        head_paired=True,
        h_r=4,
    )
    tensors = run.prepare_ragged_tensors(
        (33, 257),
        (65, 257),
        heads=8,
        cfg=cfg,
        seed=20260905,
    )
    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors, cfg=cfg)


@pytest.mark.parametrize(
    ("head_paired", "h_r", "window_size_left", "heads"),
    [
        pytest.param(False, 1, 0, 2, id="query-paired-causal"),
        pytest.param(True, 4, 129, 8, id="head-paired-left-window"),
    ],
)
@require_available_sm100()
def test_fmha_d128_packed_ragged_causal_accuracy(
    head_paired,
    h_r,
    window_size_left,
    heads,
):
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(
        scheduler="clc_dynamic_persistent",
        has_varlen=True,
        is_causal=True,
        has_q_offset=True,
        head_paired=head_paired,
        h_r=h_r,
        window_size_left=window_size_left,
    )
    tensors = run.prepare_ragged_tensors(
        (33, 257),
        (65, 385),
        heads=heads,
        cfg=cfg,
        seed=20260908 + window_size_left,
    )
    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors, cfg=cfg)


@require_available_sm100()
def test_fmha_d128_packed_ragged_bounds_exclude_peer_requests():
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(
        scheduler="clc_dynamic_persistent",
        has_varlen=True,
    )
    tensors = run.prepare_ragged_tensors(
        (33, 257),
        (65, 257),
        heads=2,
        cfg=cfg,
        seed=2026071423,
    )
    tensors["q"].zero_()
    tensors["k"].zero_()
    tensors["v"][:65].fill_(1.0)
    tensors["v"][65:].fill_(2.0)

    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors, cfg=cfg)
    torch.testing.assert_close(
        tensors["out"][:33].float(),
        torch.ones_like(tensors["out"][:33].float()),
        rtol=0.0,
        atol=1e-3,
    )
    torch.testing.assert_close(
        tensors["out"][33:].float(),
        torch.full_like(tensors["out"][33:].float(), 2.0),
        rtol=0.0,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    ("page_size", "max_pages"),
    (
        pytest.param(16, 24, id="page-16"),
        pytest.param(32, 12, id="page-32"),
        pytest.param(64, 6, id="page-64"),
        pytest.param(128, 3, id="page-128"),
    ),
)
@pytest.mark.parametrize(
    ("head_paired", "h_r", "heads"),
    (
        pytest.param(False, 1, 2, id="query-paired"),
        pytest.param(True, 4, 8, id="head-paired-gqa"),
    ),
)
@require_available_sm100()
def test_fmha_d128_paged_ragged_accuracy(
    page_size, max_pages, head_paired, h_r, heads
):
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(
        scheduler="clc_dynamic_persistent",
        has_varlen=True,
        use_paged_kv=True,
        num_tokens_per_page=page_size,
        max_num_pages_per_seq_kv=max_pages,
        head_paired=head_paired,
        h_r=h_r,
    )
    tensors = run.prepare_paged_tensors(
        (33, 257),
        (65, 257),
        heads=heads,
        cfg=cfg,
        seed=20260909 + page_size,
    )
    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors, cfg=cfg)


@pytest.mark.parametrize(
    ("head_paired", "h_r", "window_size_left", "heads"),
    (
        pytest.param(False, 1, 0, 2, id="query-paired-causal"),
        pytest.param(True, 4, 0, 8, id="head-paired-causal"),
        pytest.param(True, 4, 129, 8, id="head-paired-left-window"),
    ),
)
@require_available_sm100()
def test_fmha_d128_paged_ragged_causal_accuracy(
    head_paired,
    h_r,
    window_size_left,
    heads,
):
    resources = _module("fmha_resources")
    run = _module("fmha_run")
    cfg = resources.FmhaConfig(
        scheduler="clc_dynamic_persistent",
        has_varlen=True,
        use_paged_kv=True,
        is_causal=True,
        has_q_offset=True,
        head_paired=head_paired,
        h_r=h_r,
        window_size_left=window_size_left,
        num_tokens_per_page=32,
        max_num_pages_per_seq_kv=16,
    )
    tensors = run.prepare_paged_tensors(
        (33, 257),
        (65, 385),
        heads=heads,
        cfg=cfg,
        seed=20261000 + window_size_left,
    )
    run.run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    run.verify_output(tensors, cfg=cfg)
