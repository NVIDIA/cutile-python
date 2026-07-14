# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from ._exception import TypeCheckingError


@dataclass(frozen=True)
class CompilerOptions:
    opt_level: int = 3
    max_threads_per_block: tuple[int, int, int] | None = None
    max_blocks_per_cluster: int | None = None
    max_registers_per_thread: int | None = None
    min_blocks_per_sm: int | None = None

    def __post_init__(self):
        message = (
            "Expected max_threads_per_block to be an integer tuple of length three"
        )
        match self.max_threads_per_block:
            case tuple() as t if len(t) == 3:
                for element in self.max_threads_per_block:
                    if not isinstance(element, int):
                        raise TypeCheckingError(message)
            case None:
                ...  # ok
            case _:
                raise TypeCheckingError(message)

        for field in (
            "max_blocks_per_cluster",
            "max_registers_per_thread",
            "min_blocks_per_sm",
        ):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, int) or value < 0):
                message = (
                    f"Expected compiler option {field} to be a "
                    f"positive integer but got {value}"
                )
                raise TypeCheckingError(message)
