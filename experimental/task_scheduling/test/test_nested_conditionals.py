# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end nested conditional TMA copy tests."""

import importlib

import pytest
import torch


@pytest.mark.parametrize("rows, columns", [(1, 128), (4, 256), (9, 512), (257, 128)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nested_conditional_tma_copy_on_device(rows, columns):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires an SM100-class GPU")
    module = importlib.import_module(
        "experimental.task_scheduling.tutorial.01_copy_basics.04_copy_tma_nested_conditional"
    )
    module.run_tma_copy_nested_conditional_kernel_prim((rows, columns), verbose=False)
