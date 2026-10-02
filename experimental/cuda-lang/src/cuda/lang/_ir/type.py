# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Sequence

from typing_extensions import override

from cuda.lang._ir.ir import LocalArrayContextManagerValue
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
    SymbolicArray,
    SymbolicClosure,
    SliceType,
    StreamTy,
    SliceValue,
    EllipsisType,
)
import cuda.lang._datatype as datatype
from cuda.tile._datatype import DType, PointerInfo
from cuda.tile._cext import _make_symbolic_tensor, _make_constant_0d_tensor
from cuda.tile._ir.ir import Var, AggregateValue, TypingHooks
from cuda.lang._exception import TypeCheckingError


@dataclass(frozen=True)
class ScalarTy(TensorLikeTy):
    dtype: DType

    def __post_init__(self):
        assert isinstance(self.dtype, DType)
        assert datatype.is_numeric(self.dtype)

    @override
    def tensor_dtype(self) -> "DType":
        return self.dtype

    @override
    def tensor_shape(self) -> tuple[int, ...]:
        return ()

    def __str__(self):
        return str(self.dtype)

    @override
    def make_symbol(self, var: "Var") -> Any:
        if var.is_constant():
            return _make_constant_0d_tensor(var.get_constant(), self.dtype, False)
        return _make_symbolic_tensor(var, False)


# A value of an opaque dtype, such as mbarrier or tensor_map_descriptor
@dataclass(frozen=True)
class OpaqueTy(TensorLikeTy):
    dtype: DType

    def __post_init__(self):
        assert isinstance(self.dtype, DType)
        assert not datatype.is_numeric(self.dtype)
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
    def make_symbol(self, var: "Var") -> Any:
        return _make_symbolic_tensor(var, False)


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
    def make_symbol(self, var: "Var") -> Any:
        return _make_symbolic_tensor(var, False)


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
    def make_symbol(self, var: "Var") -> Any:
        return _make_symbolic_tensor(var, False)


def is_vector_ty(ty: Type) -> bool:
    return isinstance(ty, VectorTy)


def make_rank0_ty(dtype: DType) -> ScalarTy | PointerTy | OpaqueTy:
    if datatype.is_pointer_dtype(dtype):
        return PointerTy(dtype)
    if datatype.is_numeric(dtype):
        return ScalarTy(dtype)
    return OpaqueTy(dtype)


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


@dataclass(frozen=True)
class KernelTy(Type):
    def __str__(self):
        return "cuda.lang kernel"


@dataclass(frozen=True)
class HostEntryTy(Type):
    def __str__(self):
        return "cuda.lang host entry"


class LangTypingHooks(TypingHooks):
    @override
    def get_tensor_like_type(self, dtype: DType, shape: Sequence[int]) -> TensorLikeTy:
        match tuple(shape):
            case ():
                return make_rank0_ty(dtype)
            case (length,):
                return VectorTy(dtype, length)
            case _:
                assert False, "cuda.lang does not support N-dimensional tensors"

    @override
    def get_stream_type(self) -> Type:
        return StreamTy()

    @override
    def get_python_constant_type(self, value) -> Type | None:
        from cuda.lang._execution import host_entry, kernel

        if isinstance(value, kernel):
            return KernelTy()
        if isinstance(value, host_entry):
            return HostEntryTy()
        return None


def type_bitwidth(x: Type):
    match x:
        case PointerTy() as pt:
            return pt.pointer_dtype.bitwidth
        case ScalarTy() | OpaqueTy() as ty:
            return ty.dtype.bitwidth
        case VectorTy() as vt:
            return vt.element_dtype.bitwidth * vt.length
    raise TypeCheckingError(f"Cannot access bitwidth of type '{x}'")


__all__ = (
    "Type",
    "TupleTy",
    "ArrayTy",
    "ScalarTy",
    "OpaqueTy",
    "StreamTy",
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
    "make_rank0_ty",
    "make_vector_ty",
    "is_vector_ty",
    "MemorySpace",
    "ArrayValue",
    "TupleValue",
    "PointerInfoTy",
    "KernelTy",
    "HostEntryTy",
    "LangTypingHooks",
    "SymbolicArray",
    "SymbolicClosure",
    "SliceType",
    "SliceValue",
    "EllipsisType",
)
