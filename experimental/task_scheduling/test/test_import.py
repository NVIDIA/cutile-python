# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("cuda.lang", reason="Skipping task_scheduling test: module not found")
pytest.importorskip("task_scheduling", reason="Skipping task_scheduling test: module not found")


def test_imports():
    import cuda.lang
    import task_scheduling

    assert cuda.lang is not None
    assert task_scheduling is not None
