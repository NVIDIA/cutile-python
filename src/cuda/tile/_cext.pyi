# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence, TypeAlias

from cuda.tile._context import TileContextConfig


Dim3: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]


def launch(stream,
           grid: Dim3,
           kernel,
           kernel_args: tuple[Any, ...],
           /, *,
           programmatic_dependent_launch: bool = False):
    ...


def launch_extended(stream,
                    block_count: Dim3,
                    thread_count: Dim3,
                    kernel,
                    kernel_args: tuple[Any, ...],
                    /, *,
                    cooperative: bool = False,
                    block_in_cluster_count: Dim3 | None = None,
                    preferred_block_in_cluster_count: Dim3 | None = None,
                    programmatic_dependent_launch: bool = False
                    ):
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
    def __init__(self, parameter_annotations: Sequence):
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
    def cutile_python_v2() -> "CallingConvention":
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

    @property
    def version(self) -> int:
        ...


default_tile_context: TileContext


def _synchronize_context() -> None: ...
def _create_stream() -> int: ...
def _destroy_stream(stream: int) -> None: ...
def _benchmark(stream: int,
               grid: tuple[int] | tuple[int, int] | tuple[int, int, int],
               kernel,
               pyargs_tuples: tuple[tuple[Any, ...], ...],
               /) -> float: ...


def run_coroutine(coro):
    """
    Run a coroutine using a software stack to bypass the Python's recursion limit.
    Use resume_after() to break the call chain and push a new frame to the software stack.
    """


def _export_ipc_benchmark_payload(stream: int,
                                  grid: tuple[int] | tuple[int, int] | tuple[int, int, int],
                                  kernel,
                                  pyargs_tuples: tuple[Any, ...],
                                  /) -> bytes | None: ...


def _benchmark_with_ipc_payload(payload: bytes, /) -> float: ...


CU_TENSOR_MAP_DATA_TYPE_UINT8: int
CU_TENSOR_MAP_DATA_TYPE_UINT16: int
CU_TENSOR_MAP_DATA_TYPE_UINT32: int
CU_TENSOR_MAP_DATA_TYPE_INT32: int
CU_TENSOR_MAP_DATA_TYPE_UINT64: int
CU_TENSOR_MAP_DATA_TYPE_INT64: int
CU_TENSOR_MAP_DATA_TYPE_FLOAT16: int
CU_TENSOR_MAP_DATA_TYPE_FLOAT32: int
CU_TENSOR_MAP_DATA_TYPE_FLOAT64: int
CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: int
CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ: int
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32: int
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: int
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B: int
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B: int
CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B: int

CU_TENSOR_MAP_SWIZZLE_NONE: int
CU_TENSOR_MAP_SWIZZLE_32B: int
CU_TENSOR_MAP_SWIZZLE_64B: int
CU_TENSOR_MAP_SWIZZLE_128B: int
CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B: int
CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B: int
CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B: int

CU_TENSOR_MAP_L2_PROMOTION_NONE: int
CU_TENSOR_MAP_L2_PROMOTION_L2_64B: int
CU_TENSOR_MAP_L2_PROMOTION_L2_128B: int
CU_TENSOR_MAP_L2_PROMOTION_L2_256B: int
