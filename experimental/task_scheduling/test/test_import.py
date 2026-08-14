# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def test_imports():
    import cuda.lang
    import task_scheduling

    assert cuda.lang is not None
    assert task_scheduling is not None
