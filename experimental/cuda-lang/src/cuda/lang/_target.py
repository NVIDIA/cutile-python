# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache


class TargetFeature(Enum):
    PACKED_F32X2 = auto()


@dataclass(frozen=True)
class TargetInfo:
    major: int
    minor: int
    suffix: str | None = None

    @classmethod
    def from_arch(cls, arch: str) -> "TargetInfo":
        return _parse_target_name(arch, "compute")

    def supports(self, feature: TargetFeature) -> bool:
        return feature in _features_for_target(self)


def _parse_target_name(name: str, prefix: str) -> TargetInfo:
    match = re.fullmatch(rf"{prefix}_(\d+)([af]?)", name)
    if match is None:
        raise ValueError(f"invalid CUDA target name: {name!r}")
    digits, suffix = match.groups()
    if len(digits) < 2:
        raise ValueError(f"invalid CUDA target name: {name!r}")
    return TargetInfo(
        major=int(digits[:-1]),
        minor=int(digits[-1]),
        suffix=suffix or None,
    )


@cache
def _features_for_target(target: TargetInfo) -> frozenset[TargetFeature]:
    features = set()
    if (target.major, target.minor) >= (10, 0):
        features.add(TargetFeature.PACKED_F32X2)
    return frozenset(features)


__all__ = ("TargetFeature", "TargetInfo")
