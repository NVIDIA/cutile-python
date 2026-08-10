# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.lang._target import TargetInfo


@pytest.mark.parametrize(
    "arch,expected",
    (
        ("compute_90", TargetInfo(major=9, minor=0)),
        ("compute_100a", TargetInfo(major=10, minor=0, suffix="a")),
        ("compute_100f", TargetInfo(major=10, minor=0, suffix="f")),
    ),
)
def test_target_info_from_arch(arch, expected):
    assert TargetInfo.from_arch(arch) == expected


@pytest.mark.parametrize(
    "arch",
    (
        "invalid",
        "sm_100a",
        "compute_100z",
    ),
)
def test_target_info_from_invalid_arch(arch):
    with pytest.raises(ValueError, match="invalid CUDA target name"):
        TargetInfo.from_arch(arch)
