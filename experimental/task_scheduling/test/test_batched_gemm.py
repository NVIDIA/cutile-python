# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end BF16 expert GEMM and fused SwiGLU correctness tests."""

import importlib
from dataclasses import replace

import pytest
import torch
from task_scheduling_test_requirements import cuda_lang as cl
from task_scheduling_test_utils import require_blackwell_cc100

config = importlib.import_module(
    "experimental.task_scheduling.tutorial.08_batched_gemm.batched_gemm_config"
)
kernel = importlib.import_module(
    "experimental.task_scheduling.tutorial.08_batched_gemm.batched_gemm_kernel"
)
runner = importlib.import_module(
    "experimental.task_scheduling.tutorial.08_batched_gemm.batched_gemm_run"
)


def make_bf16_config(**overrides):
    options = {
        "dtype_a": int(config.DType.BF16),
        "dtype_b": int(config.DType.BF16),
        "dtype_c": int(config.DType.BF16),
        "batch_mode": int(config.BatchMode.BATCH_M),
        "transpose_mma_output": 0,
        "tile_n": 64,
        "mma_n": 64,
        "tile_k": 64,
        "mma_k": 16,
        "num_stages_a": 3,
        "num_stages_b": 3,
        "num_stages_tmem_acc": 2,
    }
    options.update(overrides)
    return config.make_config(**options)


@pytest.mark.parametrize("scheduler", [0, 1])
@pytest.mark.parametrize(
    "tile_n,tile_k,k,stages",
    [
        (8, 64, 64, 1),
        (16, 64, 320, 2),
        (64, 64, 576, 3),
        (64, 128, 512, 3),
        (128, 256, 768, 1),
    ],
)
@require_blackwell_cc100()
def test_batched_gemm_bf16_expert_tiles(scheduler, tile_n, tile_k, k, stages):
    cfg = make_bf16_config(
        tile_n=tile_n,
        mma_n=tile_n,
        tile_k=tile_k,
        num_stages_a=stages,
        num_stages_b=stages,
        tile_scheduler=scheduler,
    )
    tensors = runner.prepare_tensors(
        num_experts=3, num_tokens=768, problem_n=2 * tile_n, problem_k=k, cfg=cfg
    )
    # Revisit experts out of order so persistent tiles cannot reuse stale
    # metadata and B cannot accidentally be treated as a single dense matrix.
    tensors["tile_idx"].copy_(
        torch.tensor([2, 0, 2, 1, 0, 1], device="cuda", dtype=torch.int32)
    )
    for _ in range(2):
        runner.run(tensors)
        torch.cuda.synchronize()
        runner.verify_output(tensors, atol=1.0e-4, rtol=1.0e-2)


@require_blackwell_cc100()
def test_batched_gemm_fp16_output():
    cfg = replace(make_bf16_config(), dtype_c=int(config.DType.FP16))
    tensors = runner.prepare_tensors(cfg=cfg)
    runner.run(tensors)
    torch.cuda.synchronize()
    runner.verify_output(tensors, atol=1.0e-4, rtol=1.0e-2)


@pytest.mark.parametrize("scheduler", [0, 1])
@pytest.mark.parametrize("unroll_mma", [0, 1])
@pytest.mark.parametrize("act_kind", [0, 1])
@require_blackwell_cc100()
def test_batched_gemm_uses_live_k_bound(scheduler, unroll_mma, act_kind):
    cfg = make_bf16_config(
        tile_scheduler=scheduler, use_unroll_loop_2x_for_mma=unroll_mma,
        act_kind=act_kind,
    )
    tensors = runner.prepare_tensors(cfg=cfg, problem_k=256)
    compiled_kernel, pipeline, _ = runner.run(tensors)
    for k in (64, 576):
        tensors = runner.prepare_tensors(cfg=cfg, problem_k=k)
        a, b, c, tile_idx, mn_limit = (
            tensors[name] for name in ("a", "b", "c", "tile_idx", "mn_limit")
        )
        tma_a_desc, tma_b_desc = kernel.make_tensor_maps(a, b, pipeline.cfg)
        cl.launch(
            torch.cuda.current_stream(),
            pipeline.grid,
            (pipeline.cfg.threads_per_cta, 1, 1),
            compiled_kernel,
            (
                tma_a_desc,
                tma_b_desc,
                c,
                tile_idx,
                mn_limit,
                a.shape[1],
                b.shape[1],
                k,
            ),
        )
        torch.cuda.synchronize()
        runner.verify_output(tensors, atol=1.0e-4, rtol=1.0e-2)


@require_blackwell_cc100()
def test_clc_work_queue_revisits_tiles_with_top_k():
    cfg = make_bf16_config(tile_scheduler=1)
    tensors = runner.prepare_tensors(
        cfg=cfg, num_experts=3, num_tokens=8192, top_k=2, problem_n=256, problem_k=64
    )
    _, pipeline, _ = runner.run(tensors)
    assert (
        pipeline.grid[0] * pipeline.grid[1]
        > torch.cuda.get_device_properties(0).multi_processor_count
    )
    torch.cuda.synchronize()
    runner.verify_output(tensors, atol=1.0e-4, rtol=1.0e-2)


@pytest.mark.parametrize("scheduler", [0, 1])
@pytest.mark.parametrize(
    "tile_n,tile_k,k,epi_tile_n,dtype_c,unroll_mma",
    [(8, 64, 64, 8, 3, 0), (16, 128, 384, 16, 3, 0),
     (64, 64, 576, 8, 3, 1), (128, 256, 768, 32, 2, 0)],
)
@require_blackwell_cc100()
def test_fc1_swiglu_expert_tiles(
    scheduler, tile_n, tile_k, k, epi_tile_n, dtype_c, unroll_mma
):
    cfg = make_bf16_config(
        act_kind=int(config.ActKind.SWIGLU), tile_scheduler=scheduler,
        tile_n=tile_n, mma_n=tile_n, tile_k=tile_k, epi_tile_n=epi_tile_n,
        dtype_c=dtype_c, use_unroll_loop_2x_for_mma=unroll_mma,
        num_stages_a=1 if tile_k == 256 else 3,
        num_stages_b=1 if tile_k == 256 else 3,
    )
    tensors = runner.prepare_tensors(
        cfg=cfg, num_experts=3, num_tokens=769, top_k=2,
        problem_n=2 * tile_n, problem_k=k,
    )
    assert tensors["c"].shape == (tensors["a"].shape[1], tile_n)
    # Exercise the nonlinear region and asymmetric pairs, not just sigmoid(0).
    tensors["a"].mul_(8)
    tensors["b"][:, 0::2].mul_(4)
    tensors["b"][:, 1::2].mul_(12)
    # CLC must reload expert metadata across revisited/out-of-order tiles.
    tensors["tile_idx"].copy_(tensors["tile_idx"].roll(1))
    for _ in range(2):
        tensors["c"].fill_(float("nan"))
        runner.run(tensors)
        torch.cuda.synchronize()
        runner.verify_output(tensors, atol=1.0e-4, rtol=1.0e-2)
        padded = tensors["a"][0].abs().sum(-1) == 0
        assert torch.count_nonzero(tensors["c"][padded]).item() == 0


@pytest.mark.parametrize("scheduler", [0, 1])
@pytest.mark.parametrize(
    "tile_n,tile_k,k,epi_tile_n,dtype_c",
    [
        (8, 64, 64, 8, 3),
        (64, 64, 576, 32, 3),
        (128, 128, 384, 64, 2),
        (128, 128, 384, 32, 3),
    ],
)
@require_blackwell_cc100()
def test_fc1_swiglu_swap_ab_expert_tiles(
    scheduler, tile_n, tile_k, k, epi_tile_n, dtype_c
):
    cfg = make_bf16_config(
        batch_mode=int(config.BatchMode.BATCH_N), transpose_mma_output=1,
        act_kind=int(config.ActKind.SWIGLU), tile_scheduler=scheduler,
        tile_n=tile_n, mma_n=tile_n, tile_k=tile_k, epi_tile_n=epi_tile_n,
        dtype_c=dtype_c,
        use_unroll_loop_2x_for_mma=int(
            scheduler == 0 and tile_n == 128 and tile_k == 128 and epi_tile_n == 32
        ),
    )
    tensors = runner.prepare_tensors(
        cfg=cfg, num_experts=3, num_tokens=769, top_k=2,
        problem_n=256, problem_k=k,
    )
    assert tensors["c"].shape == (128, tensors["b"].shape[1])
    assert tensors["c"].transpose(0, 1).is_contiguous()
    tensors["a"].mul_(8)
    tensors["b"].mul_(4)
    tensors["tile_idx"].copy_(tensors["tile_idx"].roll(1))
    for _ in range(2):
        tensors["c"].fill_(float("nan"))
        runner.run(tensors)
        torch.cuda.synchronize()
        runner.verify_output(tensors, atol=1.0e-4, rtol=1.0e-2)
        padded = tensors["b"][0].abs().sum(-1) == 0
        assert torch.count_nonzero(tensors["c"][:, padded]).item() == 0
