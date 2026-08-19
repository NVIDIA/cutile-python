# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from task_scheduling_test_requirements import cuda_lang, task_scheduling


def test_imports():
    assert cuda_lang is not None
    assert task_scheduling is not None
    assert not hasattr(task_scheduling, "TaskLocalVariable")
