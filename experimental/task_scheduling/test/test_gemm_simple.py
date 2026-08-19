# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

from task_scheduling_test_utils import require_blackwell_cc100


def require_available_blackwell():
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="requires a Blackwell CC 10.0 GPU")
    return require_blackwell_cc100()


def test_fp16_gemm_program_infers_resource_roles():
    kernel = importlib.import_module(
        "experimental.task_scheduling.tutorial.02_gemm_simple."
        "01_fp16_bf16_gemm_3"
    )
    program = kernel.make_gemm_program()
    tasks = {task.name: task for task in program.manager.user_tasks}

    expected_roles = {
        "LoadTask": (["GmemAb"], ["SmemAb"]),
        "MmaTask": (["SmemAb"], ["TmemC"]),
        "StoreTask": (["TmemC"], ["GmemD"]),
        "PaddingTask": ([], []),
    }
    for name, (consumers, producers) in expected_roles.items():
        task = tasks[name]
        assert [resource.name for resource in task.src_resources] == consumers
        assert [resource.name for resource in task.dst_resources] == producers

    assert program.manager.resource_dependency_graph == {}


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@require_available_blackwell()
def test_fp16_bf16_gemm_3_prim_ts_1cta(dtype):
    """Minimal single-CTA CUDA Lang task-scheduled GEMM."""
    kernel = importlib.import_module(
        "experimental.task_scheduling.tutorial.02_gemm_simple."
        "01_fp16_bf16_gemm_3"
    )

    tolerance = 1.0e-4 if dtype == "fp16" else 2.0e-2
    kernel.verify(
        (128, 256, 64),
        tolerance=tolerance,
        dtype=dtype,
    )
