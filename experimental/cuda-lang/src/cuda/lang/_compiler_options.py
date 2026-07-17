# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass, field, fields
from ._exception import TypeCheckingError


@dataclass(frozen=True)
class CompilerOptions:
    opt_level: int = 3
    max_threads_per_block: tuple[int, int, int] | None = None
    max_blocks_per_cluster: int | None = None
    max_registers_per_thread: int | None = None
    min_blocks_per_sm: int | None = None
    _ptx_compiler_verbose: bool = field(
        default=False,
        metadata={"PTX_FLAG": "--verbose"},
    )
    _ptx_compiler_warn_on_local_memory_usage: bool = field(
        default=False,
        metadata={"PTX_FLAG": "--warn-on-local-memory-usage"},
    )
    _ptx_compiler_warn_on_spills: bool = field(
        default=False,
        metadata={"PTX_FLAG": "--warn-on-spills"},
    )
    _ptx_compiler_make_errors_visible_at_exit: bool = field(
        default=False,
        metadata={"PTX_FLAG": "--make-errors-visible-at-exit"},
    )

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

        for field_ in (
            "max_blocks_per_cluster",
            "max_registers_per_thread",
            "min_blocks_per_sm",
        ):
            value = getattr(self, field_)
            if value is not None and (not isinstance(value, int) or value < 0):
                message = (
                    f"Expected compiler option {field_} to be a "
                    f"positive integer but got {value}"
                )
                raise TypeCheckingError(message)

        for field_ in fields(self):
            if "PTX_FLAG" not in field_.metadata:
                continue

            if not isinstance(getattr(self, field_.name), bool):
                message = (
                    f"Expected compiler option {field_.name} to be bool "
                    f"but got {type(getattr(self, field_.name))}"
                )
                raise TypeCheckingError(message)

    @property
    def _ptx_compiler_options(self) -> tuple[str, ...]:
        return tuple(
            field_.metadata["PTX_FLAG"]
            for field_ in fields(self)
            if "PTX_FLAG" in field_.metadata and getattr(self, field_.name)
        )
