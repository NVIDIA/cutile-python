# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence

from cuda.tile._context import TileContextConfig


def launch(stream,
           grid: tuple[int] | tuple[int, int] | tuple[int, int, int],
           kernel,
           kernel_args: tuple[Any, ...],
           /):
    ...


def launch_with_block(stream,
                      grid: tuple[int] | tuple[int, int] | tuple[int, int, int],
                      block: tuple[int] | tuple[int, int] | tuple[int, int, int],
                      kernel,
                      kernel_args: tuple[Any, ...],
                      /):
    ...


def get_compute_capability():
    ...


def get_driver_version():
    ...


def _get_max_grid_size(device_id, /):
    ...


def get_parameter_constraints_from_pyargs(dispatcher, pyargs, calling_convention, /):
    ...


def dev_features_enabled():
    ...


class TileDispatcher:
    def __init__(self, arg_constant_flags: Sequence[bool]):
        ...


class TileContext:
    def __init__(self, config: TileContextConfig):
        ...

    @property
    def config(self) -> TileContextConfig:
        ...

    @property
    def autotune_cache(self) -> Any | None:
        ...

    @autotune_cache.setter
    def autotune_cache(self, value: Any | None):
        ...


class CallingConvention:
    @staticmethod
    def cutile_python_v1() -> "CallingConvention":
        ...

    @staticmethod
    def from_code(code: str, /) -> "CallingConvention":
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def code(self) -> str:
        ...


default_tile_context: TileContext
