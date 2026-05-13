# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from cuda.lang._ir.ir import LocalArrayContextManagerValue
from cuda.lang._enums import TensorMapSwizzle
from cuda.tile._ir.type import (
    Type,
    TupleTy,
    ArrayTy,
    TileTy,
    StringTy,
    FunctionTy,
    DTypeConstructor,
    NoneType,
    ModuleTy,
    PointerTy,
    TokenTy,
    TypeTy,
    EnumTy,
    make_tile_ty,
    ContextManagerTy,
    ContextManagerState,
    MemorySpace,
    ArrayValue,
    TupleValue
)
import cuda.tile._datatype as datatype
from cuda.tile._datatype import DType
from cuda.tile._ir.ir import Var, AggregateValue
from cuda.lang._exception import TileTypeError


def _is_power_of_2(value: int) -> bool:
    assert isinstance(value, int)
    return value > 0 and value & (value - 1) == 0


def is_vector_ty(ty: Type) -> bool:
    return (
        isinstance(ty, TileTy)
        and len(ty.shape) == 1
        and _is_power_of_2(ty.shape[0])
    )


def make_vector_ty(dtype: DType, length: int) -> TileTy:
    if not isinstance(length, int):
        raise TileTypeError(
            f"Expected vector length to be an int, got {type(length).__name__}"
        )
    if not _is_power_of_2(length):
        raise TileTypeError(
            f"Expected vector length to be a positive power of two, got {length}"
        )
    return make_tile_ty(dtype, (length,))


@dataclass(frozen=True, eq=True)
class OpaquePointerTy(Type):
    memory_space: MemorySpace = MemorySpace.GENERIC

    def __str__(self) -> str:
        return f'OpaquePtr[{self.memory_space}]'

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class LocalArrayContextManagerTy(ContextManagerTy):
    dtype: DType
    shape: tuple[int, ...]
    alignment: int | None
    state: ContextManagerState

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple[Type, ...]:
        return ()

    def make_aggregate_value(self, items: tuple[Var, ...]) -> AggregateValue:
        return LocalArrayContextManagerValue()

    def get_context_manager_state(self) -> ContextManagerState:
        return self.state


def dtype_to_tensor_map_type(dtype: datatype.DType) -> str:
    match dtype:
        case datatype.uint8: return "CU_TENSOR_MAP_DATA_TYPE_UINT8"
        case datatype.uint16: return "CU_TENSOR_MAP_DATA_TYPE_UINT16"
        case datatype.uint32: return "CU_TENSOR_MAP_DATA_TYPE_UINT32"
        case datatype.int32: return "CU_TENSOR_MAP_DATA_TYPE_INT32"
        case datatype.uint64: return "CU_TENSOR_MAP_DATA_TYPE_UINT64"
        case datatype.int64: return "CU_TENSOR_MAP_DATA_TYPE_INT64"
        case datatype.float16: return "CU_TENSOR_MAP_DATA_TYPE_FLOAT16"
        case datatype.float32: return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32"
        case datatype.float64: return "CU_TENSOR_MAP_DATA_TYPE_FLOAT64"
        case datatype.bfloat16: return "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16"
        case datatype.tfloat32: return "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32"
        case _:
            raise TileTypeError(f"Data type {dtype} is not supported by tensor map")


@dataclass(frozen=True)
class TensorMapTy(Type):
    data_type: str  # "CU_TENSOR_MAP_DATA_TYPE_*"
    tile_shape: tuple[int, ...]
    swizzle: TensorMapSwizzle


__all__ = (
    "Type",
    "TupleTy",
    "ArrayTy",
    "TileTy",
    "StringTy",
    "FunctionTy",
    "DTypeConstructor",
    "NoneType",
    "OpaquePointerTy",
    "ModuleTy",
    "PointerTy",
    "TokenTy",
    "TypeTy",
    "EnumTy",
    "make_tile_ty",
    "make_vector_ty",
    "is_vector_ty",
    "MemorySpace",
    "ArrayValue",
    "TupleValue",
)
