# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest


cuda_lang = pytest.importorskip(
    "cuda.lang", reason="Skipping task-scheduling test: cuda.lang not found"
)
task_scheduling = pytest.importorskip(
    "task_scheduling", reason="Skipping task-scheduling test: package not found"
)


def require_task_scheduling_dependencies():
    return cuda_lang, task_scheduling
