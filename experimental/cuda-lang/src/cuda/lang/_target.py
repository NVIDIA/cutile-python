# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from ._exception import InternalError


@dataclass(frozen=True)
class TargetInfo:
    major: int
    minor: int
    suffix: str | None = None

    @classmethod
    def from_arch(cls, arch: str) -> "TargetInfo":
        return _parse_target_name(arch, "compute")

    @staticmethod
    def get_current() -> "TargetInfo":
        """Return the GPU target of the active IR builder's context."""
        from cuda.lang._ir.ir import TileBuilder
        ctx = TileBuilder.get_current().ir_ctx
        if ctx.target_info is None:
            raise InternalError("The current IR context has no GPU target")
        return ctx.target_info


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


__all__ = ("TargetInfo",)
