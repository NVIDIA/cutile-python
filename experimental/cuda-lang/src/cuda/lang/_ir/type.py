# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Sequence

from typing_extensions import override

from cuda.lang._ir.ir import LocalArrayContextManagerValue
from cuda.lang._enums import SwizzleMode
from cuda.lang._stub.types import Scalar, Pointer, Vector
from cuda.tile._ir.type import (
    Type,
    TupleTy,
    ArrayTy,
    StringTy,
    FunctionTy,
    DTypeConstructor,
    DTypeSpec,
    NoneType,
    ModuleTy,
    TokenTy,
    TypeTy,
    EnumTy,
    ContextManagerTy,
    ContextManagerState,
    MemorySpace,
    ArrayValue,
    TupleValue,
    PointerInfoTy,
    TensorLikeTy,
    Symbol,
    SymbolicArray,
    SymbolicClosure,
)
import cuda.tile._datatype as datatype
from cuda.tile._datatype import DType, PointerInfo
from cuda.tile._ir.ir import Var, AggregateValue, TypingHooks
from cuda.lang._exception import TypeCheckingError


@dataclass(frozen=True)
class ScalarTy(TensorLikeTy):
    dtype: DType

    def __post_init__(self):
        assert isinstance(self.dtype, DType)
        assert not datatype.is_pointer_dtype(self.dtype)

    @override
    def tensor_dtype(self) -> "DType":
        return self.dtype

    @override
    def tensor_shape(self) -> tuple[int, ...]:
        return ()

    def __str__(self):
        return str(self.dtype)

    @override
    def make_symbol(self, var: "Var") -> Symbol:
        return SymbolicScalar(var)


class SymbolicScalar(Symbol, Scalar):
    def __init__(self, var: "Var[ScalarTy]"):
        Symbol.__init__(self, var)

    @property
    def dtype(self):
        return self._var.get_type().dtype

    def __bool__(self):
        raise ValueError(
            "Symbolic scalar has no concrete value and thus cannot be converted"
            " to boolean"
        )

    def __int__(self):
        raise ValueError(
            "Symbolic scalar has no concrete value and thus cannot be converted"
            " to an integer"
        )

    def __float__(self):
        raise ValueError(
            "Symbolic scalar has no concrete value and thus cannot be converted"
            " to a float"
        )

    def __index__(self):
        raise ValueError(
            "Symbolic scalar has no concrete value and thus cannot be converted"
            " to an integer"
        )

    def __repr__(self):
        return f"<scalar[{self.dtype}]>"


@dataclass(frozen=True)
class PointerTy(TensorLikeTy):
    # NOTE: this is the *pointer*, not *pointee* dtype.
    pointer_dtype: DType

    def __post_init__(self):
        assert datatype.is_pointer_dtype(self.pointer_dtype)

    @property
    def opaque(self) -> bool:
        return PointerInfo(self.pointer_dtype).opaque

    @property
    def pointee_dtype(self) -> DType:
        return PointerInfo(self.pointer_dtype).pointee_dtype

    @property
    def memory_space(self) -> MemorySpace:
        return PointerInfo(self.pointer_dtype).memory_space

    @override
    def tensor_dtype(self) -> "DType":
        return self.pointer_dtype

    @override
    def tensor_shape(self) -> tuple[int, ...]:
        return ()

    def __str__(self):
        return str(self.pointer_dtype)

    @override
    def make_symbol(self, var: "Var") -> Symbol:
        return SymbolicPointer(var)


class SymbolicPointer(Symbol, Pointer):
    def __init__(self, var: "Var[PointerTy]"):
        Symbol.__init__(self, var)

    @property
    def pointer_dtype(self):
        return self._var.get_type().pointer_dtype

    @property
    def opaque(self) -> bool:
        return PointerInfo(self.pointer_dtype).opaque

    @property
    def pointee_dtype(self) -> DType:
        return PointerInfo(self.pointer_dtype).pointee_dtype

    @property
    def memory_space(self) -> MemorySpace:
        return PointerInfo(self.pointer_dtype).memory_space

    def __repr__(self):
        return f"<{self.pointer_dtype}>"


@dataclass(frozen=True)
class VectorTy(TensorLikeTy):
    element_dtype: DType
    length: int

    def __post_init__(self):
        if not isinstance(self.length, int):
            raise TypeCheckingError(
                f"Expected vector length to be an int, got {type(self.length).__name__}"
            )
        if self.length <= 0:
            raise TypeCheckingError(f"Expected vector length to be positive, got {self.length}")

    @override
    def tensor_dtype(self) -> "DType":
        return self.element_dtype

    @override
    def tensor_shape(self) -> tuple[int, ...]:
        return (self.length,)

    def __str__(self):
        return f"Vector[{self.element_dtype}, {self.length}]"

    @override
    def make_symbol(self, var: "Var") -> Symbol:
        return SymbolicVector(var)


class SymbolicVector(Symbol, Vector):
    def __init__(self, var: "Var[VectorTy]"):
        Symbol.__init__(self, var)

    @property
    def element_dtype(self) -> "DType":
        return self._var.get_type().element_dtype

    @property
    def element_count(self) -> int:
        return self._var.get_type().length

    def __len__(self) -> int:
        return self.element_count

    def __repr__(self):
        return f"<vector[{self.element_dtype}, count={self.element_count}]>"


def is_vector_ty(ty: Type) -> bool:
    return isinstance(ty, VectorTy)


def make_vector_ty(dtype: DType, length: int) -> VectorTy:
    return VectorTy(dtype, length)


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
    # NOTE: <8b floats will need an explicit encoding argument since they could
    # map to 16U4_ALIGN8B or 16U4_ALIGN16B
    match dtype:
        case (
            datatype.uint8
            | datatype.int8
            | datatype.float8_e4m3fn
            | datatype.float8_e5m2
            | datatype.float8_e8m0fnu
        ):
            return "CU_TENSOR_MAP_DATA_TYPE_UINT8"
        case datatype.uint16:
            return "CU_TENSOR_MAP_DATA_TYPE_UINT16"
        case datatype.uint32:
            return "CU_TENSOR_MAP_DATA_TYPE_UINT32"
        case datatype.int32:
            return "CU_TENSOR_MAP_DATA_TYPE_INT32"
        case datatype.uint64:
            return "CU_TENSOR_MAP_DATA_TYPE_UINT64"
        case datatype.int64:
            return "CU_TENSOR_MAP_DATA_TYPE_INT64"
        case datatype.float16:
            return "CU_TENSOR_MAP_DATA_TYPE_FLOAT16"
        case datatype.float32:
            return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32"
        case datatype.float64:
            return "CU_TENSOR_MAP_DATA_TYPE_FLOAT64"
        case datatype.bfloat16:
            return "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16"
        case datatype.tfloat32:
            return "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32"
        case _:
            raise TypeCheckingError(f"Data type {dtype} is not supported by tensor map")


@dataclass(frozen=True)
class TensorMapTy(Type):
    data_type: str  # "CU_TENSOR_MAP_DATA_TYPE_*"
    element_bitwidth: int
    tile_shape: tuple[int, ...]
    swizzle: SwizzleMode


class LangTypingHooks(TypingHooks):
    @override
    def get_tensor_like_type(self, dtype: DType, shape: Sequence[int]) -> TensorLikeTy:
        match tuple(shape):
            case () if datatype.is_pointer_dtype(dtype):
                return PointerTy(dtype)
            case ():
                return ScalarTy(dtype)
            case (length,):
                return VectorTy(dtype, length)
            case _:
                assert False, "cuda.lang does not support N-dimensional tensors"


def type_bitwidth(x: Type):
    match x:
        case TensorMapTy():
            return 128
        case PointerTy() as pt:
            info = PointerInfo(pt.pointer_dtype)
            return (
                32
                if info.memory_space
                in (MemorySpace.SHARED, MemorySpace.SHARED_CLUSTER, MemorySpace.TENSOR)
                else 64
            )
        case ScalarTy() as st:
            return st.dtype.bitwidth
        case VectorTy() as vt:
            return vt.element_dtype.bitwidth * vt.length
    raise TypeCheckingError(f"Cannot access bitwidth of type '{x}'")


__all__ = (
    "Type",
    "TupleTy",
    "ArrayTy",
    "ScalarTy",
    "PointerTy",
    "VectorTy",
    "StringTy",
    "FunctionTy",
    "DTypeConstructor",
    "DTypeSpec",
    "NoneType",
    "ModuleTy",
    "TokenTy",
    "TypeTy",
    "EnumTy",
    "make_vector_ty",
    "is_vector_ty",
    "MemorySpace",
    "ArrayValue",
    "TupleValue",
    "PointerInfoTy",
    "LangTypingHooks",
    "SymbolicArray",
    "SymbolicClosure",
    "SymbolicVector",
    "SymbolicScalar",
    "SymbolicPointer",
)
