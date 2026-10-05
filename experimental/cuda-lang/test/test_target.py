# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.lang._target import TargetInfo
from cuda.lang._exception import InternalError
from cuda.lang._ir.ir import (
    Builder,
    IRContext,
    Loc,
    Region,
    TileBuilder,
)


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


def test_host_context_rejects_target_info():
    target = TargetInfo.from_arch("compute_100a")
    with pytest.raises(ValueError, match="host IR context"):
        IRContext(execution_space="host", target_info=target)


def test_target_info_get_current():
    target = TargetInfo.from_arch("compute_100f")
    ctx = IRContext(target_info=target)
    loc = Loc.unknown()
    with Builder(Region(ctx), loc):
        assert TargetInfo.get_current() is target


@pytest.mark.parametrize("execution_space", ("device", "host"))
def test_target_info_get_current_without_target(execution_space):
    ctx = IRContext(execution_space=execution_space)
    with TileBuilder(ctx, Loc.unknown()):
        with pytest.raises(InternalError, match="current IR context has no GPU target"):
            TargetInfo.get_current()
