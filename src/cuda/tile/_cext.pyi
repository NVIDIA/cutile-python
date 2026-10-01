# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum
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


def get_compute_capability(device_id: int = 0) -> tuple[int, int]:
    ...


def get_driver_version():
    ...


def _get_max_grid_size(device_id, /):
    ...


def get_parameter_constraints_from_pyargs(dispatcher, pyargs, calling_convention, /):
    ...


class ConstantKind(enum.Enum):
    ...


def classify_constant(value, kernel_arg, /) -> ConstantKind | None:
    ...


def foreign_dtype_object_register(foreign, native):
    ...


def foreign_dtype_object_to_native(foreign, /):
    ...


def dev_features_enabled():
    ...


def cconv_v3_enabled():
    ...


class TileDispatcher:
    def __init__(self, parameter_annotations: Sequence):
        ...


class HostDispatcher:
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
               flush_l2: bool = True,
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
                                  flush_l2: bool = True,
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


class BitstreamWriter:
    def fixed1(self, value: int) -> None: ...
    def fixed2(self, value: int) -> None: ...
    def fixed3(self, value: int) -> None: ...
    def fixed4(self, value: int) -> None: ...
    def fixed8(self, value: int) -> None: ...
    def fixed32(self, value: int) -> None: ...

    def vbr4(self, value: int) -> None: ...
    def vbr5(self, value: int) -> None: ...
    def vbr6(self, value: int) -> None: ...
    def vbr8(self, value: int) -> None: ...

    def align_to_word(self) -> int: ...
    def aligned_word(self, value: int) -> int: ...
    def patch_word(self, offset: int, value: int) -> None: ...

    def raw_blob(self, data: bytearray) -> None: ...

    def to_bytes(self) -> bytes: ...


class NVVMProgram:
    def add_module(self, contents: bytes | bytearray, name: str, /) -> None: ...
    def compile(self, options: Sequence[str], /) -> bytes: ...


class NVVM:
    def __init__(self, dll_path: str, /): ...
    def ir_version(self) -> tuple[int, int, int, int]: ...
    def version(self) -> tuple[int, int]: ...
    def create_program(self) -> NVVMProgram: ...
