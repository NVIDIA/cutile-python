# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from enum import EnumMeta
from types import ModuleType, FunctionType
from typing import Any, Callable, Optional, Sequence, Tuple, Iterator
from functools import reduce
import operator

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cuda.tile._datatype import DType


import cuda.tile._bytecode as bc


class Type:

    def __repr__(self):
        return str(self)

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other: "Type"):
        raise NotImplementedError()


@dataclass
class LooselyTypedScalar(Type):
    value: Any


# ============== None Type ===============

class NoneType(Type):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self):
        return "None"

    def __eq__(self, other: Type):
        return isinstance(other, NoneType)

    def __hash__(self):
        return hash("NoneType")


NONE = NoneType()


# ============== Slice Type ===============

class SliceType(Type):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self):
        return "Slice"

    def __eq__(self, other: Type):
        return isinstance(other, NoneType)

    def __hash__(self):
        return hash("SliceType")


SLICE = SliceType()


# ============== Ellipsis Type ===============

class EllipsisType(Type):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self):
        return "Ellipsis"

    def __eq__(self, other: Type):
        return isinstance(other, EllipsisType)

    def __hash__(self):
        return hash("EllipsisType")


ELLIPSIS = EllipsisType()


# ============== Undefined Type ===============

class UndefinedType(Type):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self):
        return "UndefinedType"

    def __eq__(self, other: Type):
        return isinstance(other, UndefinedType)

    def __hash__(self):
        return hash("UndefinedType")


UNDEFINED = UndefinedType()


# ============== String Type ===============

@dataclass(frozen=True, repr=False)
class StringTy(Type):
    value: str

    def __repr__(self):
        return f"<string constant '{self.value}'>"


# ============== Type of DType ===============

@dataclass(frozen=True)
class DTypeSpec(Type):
    dtype: 'DType' = None


# Data type constant that is also callable, e.g. np.float32(1.0)
class DTypeConstructor(DTypeSpec):
    pass


# ============== Tuple ===============

class TupleTy(Type):
    def __init__(self, value_types: Sequence[Type]):
        self._value_types = tuple(value_types)

    def len(self) -> int:
        return len(self._value_types)

    @property
    def value_types(self) -> Tuple[Type, ...]:
        return self._value_types

    def __len__(self) -> int:
        return len(self._value_types)

    def __iter__(self) -> Iterator[Type]:
        return iter(self.value_types)

    def __getitem__(self, index: int) -> Type:
        return self.value_types[index]

    def __eq__(self, other: Type):
        return isinstance(other, TupleTy) and self._value_types == other._value_types

    def __hash__(self):
        return hash(("TupleTy", self._value_types))

    def __str__(self):
        return 'Tuple[' + ','.join(str(x) for x in self._value_types) + ']'

    def map(self, unwrap: Callable[[Type], Any]) -> Tuple[Any, ...]:
        return tuple(unwrap(t) for t in self.value_types)


# ============== SizeTy ===============

class SizeTy(Type):

    def __init__(self, value: Optional[int] = None):
        """Represent a compile time or runtime size"""
        if isinstance(value, int):
            if value < 0:
                raise TypeError(f'SizeTy value must be non negative, got {value}')
        elif value is not None:
            raise TypeError(f'SizeTy value must be int or None, got {value}')
        self._value = value

    @property
    def value(self) -> int:
        if self._value is None:
            raise TypeError('SizeTy value is unknown at compile time')
        return self._value

    @property
    def maybe_value(self) -> Optional[int]:
        return self._value

    @property
    def bytecode_value(self) -> int:
        return bc.DYNAMIC_SHAPE if self._value is None else self._value

    def __eq__(self, other: Type):
        return isinstance(other, SizeTy) and self._value == other._value

    def __hash__(self):
        return hash(("SizeTy", self._value))

    def __str__(self):
        return 'Size(?)' if self._value is None else f'Size({self._value})'


# ============== Tile Type ===============


class TileTy(Type):
    def __init__(self,
                 dtype,
                 shape: TupleTy):
        self.dtype = dtype
        self.shape = shape
        try:
            unwrap: Callable[[SizeTy], int] = lambda t: t.value
            self._unwrapped_shape: Tuple[int, ...] = shape.map(unwrap)
        except (TypeError, AttributeError):
            raise TypeError(f'`shape` must be an Tuple[Size, ...], got: {shape}') from None

    @property
    def shape_value(self) -> Tuple[int, ...]:
        return tuple(x.value for x in self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numel(self):
        # Total number of elements
        return reduce(operator.mul, self._unwrapped_shape, 1)

    def __eq__(self, other: Type):
        if isinstance(other, TileTy):
            return self.dtype == other.dtype and self.shape == other.shape
        return False

    def __hash__(self):
        return hash(("TileTy", self.dtype, self.shape))

    def __str__(self):
        shape_str = "(" + ','.join(str(x) for x in self._unwrapped_shape) + ")"
        return f"Tile[{self.dtype},{shape_str}]"


def make_tile_ty(dtype, shape: Sequence[int]):
    shape = TupleTy(tuple(SizeTy(x) for x in shape))
    return TileTy(dtype, shape)


# ============== Array Type ===============

class ArrayTy(Type):
    def __init__(self,
                 dtype,
                 /,
                 shape: TupleTy,
                 strides: TupleTy,
                 elements_disjoint: bool,
                 base_ptr_div_by: Optional[int],
                 stride_div_by: Tuple[Optional[int], ...],
                 shape_div_by: Tuple[Optional[int], ...]):
        self.dtype = dtype
        self.shape = shape
        self.strides = strides

        unwrap: Callable[[SizeTy], Optional[int]] = lambda t: t.maybe_value
        try:
            self._unwrapped_shape: Tuple[Optional[int], ...] = shape.map(unwrap)
        except (TypeError, AttributeError):
            raise TypeError(f'`shape` must be an Tuple[Size], got: {shape}') from None

        try:
            self._unwrapped_strides: Tuple[Optional[int], ...] = strides.map(unwrap)
        except (TypeError, AttributeError):
            raise TypeError(f'`strides` must be an Tuple[Size], got: {strides}') from None

        self.elements_disjoint = elements_disjoint
        self.base_ptr_div_by = base_ptr_div_by
        self.stride_div_by = stride_div_by
        self.shape_div_by = shape_div_by

    def unify(self, other: "ArrayTy") -> Optional["ArrayTy"]:
        if self.dtype != other.dtype or self.ndim != other.ndim:
            return None

        shape = TupleTy(tuple(s1 if s1 == s2 else SizeTy()
                              for s1, s2 in zip(self.shape, other.shape, strict=True)))
        strides = TupleTy(tuple(s1 if s1 == s2 else SizeTy()
                                for s1, s2 in zip(self.strides, other.strides, strict=True)))

        elements_disjoint = self.elements_disjoint and other.elements_disjoint
        base_ptr_div_by = math.gcd(self.base_ptr_div_by, other.base_ptr_div_by)
        shape_div_by = tuple(
            None if (d1 is None or d2 is None) else math.gcd(d1, d2)
            for d1, d2 in zip(self.shape_div_by, other.shape_div_by, strict=True)
        )
        stride_div_by = tuple(
            None if (d1 is None or d2 is None) else math.gcd(d1, d2)
            for d1, d2 in zip(self.stride_div_by, other.stride_div_by, strict=True)
        )
        return ArrayTy(self.dtype,
                       shape=shape,
                       strides=strides,
                       elements_disjoint=elements_disjoint,
                       base_ptr_div_by=base_ptr_div_by,
                       shape_div_by=shape_div_by,
                       stride_div_by=stride_div_by)

    @property
    def ndim(self):
        return len(self.shape)

    def __eq__(self, other: Type):
        return (isinstance(other, ArrayTy)
                and self.dtype == other.dtype
                and self.shape == other.shape
                and self.strides == other.strides
                and self.base_ptr_div_by == self.base_ptr_div_by
                and self.stride_div_by == self.stride_div_by
                and self.shape_div_by == self.shape_div_by)

    def __hash__(self):
        return hash(("ArrayTy", self.dtype, self.shape, self.strides,
                     self.base_ptr_div_by, self.stride_div_by, self.shape_div_by))

    def __str__(self):
        shape_str = ('?' if x is None else str(x) for x in self._unwrapped_shape)
        shape_str = "(" + ','.join(shape_str) + ")"
        strides_str = ('?' if x is None else str(x) for x in self._unwrapped_strides)
        strides_str = "(" + ','.join(strides_str) + ")"
        return f"Array[{self.dtype},{shape_str}:{strides_str}]"


# ============== List Type ===============


@dataclass(frozen=True)
class ListTy(Type):
    item_type: Type


# ============== Pointer Type ===============


@dataclass(frozen=True)
class PointerTy(Type):
    pointee_type: Type


# ============== Range Iter Type ===============


class RangeIterType(Type):
    def __init__(self, dtype):
        self.dtype = dtype

    def __str__(self):
        return f"Range<{self.dtype}>"

    def __eq__(self, other: Type):
        return isinstance(other, RangeIterType) and other.dtype == self.dtype


# =============== Token Type ================


@dataclass(frozen=True)
class TokenTy(Type):
    def __str__(self):
        return "Token"


@dataclass(frozen=True)
class ModuleTy(Type):
    py_mod: ModuleType

    def __str__(self):
        return str(self.py_mod)


@dataclass(frozen=True)
class TypeTy(Type):
    ty: type


@dataclass(frozen=True)
class FunctionTy(Type):
    func: FunctionType

    def __str__(self):
        return str(self.func)


@dataclass(frozen=True)
class BoundMethodTy(Type):
    self_ty: Type
    func: FunctionType


@dataclass(frozen=True)
class EnumTy(Type):
    enum_ty: EnumMeta

    def __str__(self) -> str:
        return f"Enum[{self.enum_ty.__name__}]"
