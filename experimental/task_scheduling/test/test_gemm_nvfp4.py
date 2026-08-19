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


_NVFP4_TMA_WARP_CONFIGS = [False, True]


@pytest.mark.parametrize(
    "use_two_tma_warps",
    _NVFP4_TMA_WARP_CONFIGS,
    ids=["1tma", "2tma"],
)
def test_nvfp4_program_infers_resource_roles(use_two_tma_warps):
    mod = importlib.import_module(
        "experimental.task_scheduling.tutorial.05_gemm_nvfp4.01_gemm_nvfp4"
    )
    program = mod.make_gemm_program(use_two_tma_warps)
    tasks = {task.name: task for task in program.manager.user_tasks}

    assert [resource.name for resource in tasks["MmaTask"].src_resources] == [
        "SmemA",
        "SmemB",
        "SmemSfA",
        "SmemSfB",
    ]
    assert [resource.name for resource in tasks["MmaTask"].dst_resources] == [
        "TmemSfA",
        "TmemSfB",
        "TmemC",
    ]
    assert [resource.name for resource in tasks["StoreTask"].src_resources] == [
        "TmemC"
    ]
    assert [resource.name for resource in tasks["StoreTask"].dst_resources] == [
        "GmemD"
    ]

    if use_two_tma_warps:
        expected_load_roles = {
            "LoadATask": (["GmemA", "GmemSfA"], ["SmemA", "SmemSfA"]),
            "LoadBTask": (["GmemB", "GmemSfB"], ["SmemB", "SmemSfB"]),
        }
    else:
        expected_load_roles = {
            "LoadTask": (
                ["GmemA", "GmemB", "GmemSfA", "GmemSfB"],
                ["SmemA", "SmemB", "SmemSfA", "SmemSfB"],
            )
        }
    for name, (consumers, producers) in expected_load_roles.items():
        task = tasks[name]
        assert [resource.name for resource in task.src_resources] == consumers
        assert [resource.name for resource in task.dst_resources] == producers
    assert tasks["PaddingTask"].resources == []

    assert program.manager.resource_dependency_graph == {}


@pytest.mark.parametrize(
    "use_two_tma_warps",
    _NVFP4_TMA_WARP_CONFIGS,
    ids=["1tma-fused-static", "2tma-fused-static"],
)
@pytest.mark.parametrize(
    "mnkl",
    [
        (256, 256, 256, 1),
        (8192, 8192, 8192, 8),
    ],
)
@require_available_blackwell()
def test_gemm_nvfp4_ts(
    mnkl,
    use_two_tma_warps,
):
    mod = importlib.import_module(
        "experimental.task_scheduling.tutorial.05_gemm_nvfp4.01_gemm_nvfp4"
    )

    mod.verify(
        mnkl,
        tolerance=1.0e-1,
        use_two_tma_warps=use_two_tma_warps,
    )
