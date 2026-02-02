# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import math
from dataclasses import dataclass
from enum import EnumMeta
from types import ModuleType, FunctionType
from typing import Any, Callable, Optional, Sequence, Tuple, Iterator
from functools import reduce
import operator

from typing import TYPE_CHECKING

from cuda.tile._exception import Loc

if TYPE_CHECKING:
    from cuda.tile._datatype import DType
    from cuda.tile._ir.ir import Var, AggregateValue
    from cuda.tile._ir import hir
    from cuda.tile._ir.scope import LocalScope


import cuda.tile._bytecode as bc


class Type:
    def is_aggregate(self) -> bool:
        return False

    def aggregate_item_types(self) -> tuple["Type", ...]:
        raise NotImplementedError()

    def flatten_aggregate(self) -> Iterator["Type"]:
        if self.is_aggregate():
            for ty in self.aggregate_item_types():
                yield from ty.flatten_aggregate()
        else:
            yield self

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        raise NotImplementedError()

    def __repr__(self):
        return str(self)

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other: "Type"):
        raise NotImplementedError()


@dataclass
class LooselyTypedScalar(Type):
    value: Any

    @property
    def shape(self):
        return ()

    @property
    def shape_value(self):
        return ()


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
        return isinstance(other, SliceType)

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


# ============== Invalid Type ===============

# Type that generates an error when used.

@dataclass
class InvalidType(Type):
    error_message: str
    loc: Loc

    def __repr__(self):
        return f"<Invalid type: {self.error_message}>"


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

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return self._value_types

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        from .ir import TupleValue
        return TupleValue(items)

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


# ============== Pointer Type ===============


@dataclass(frozen=True)
class PointerTy(Type):
    pointee_type: "DType"


# ============== Tile Type ===============


class TileTy(Type):
    def __init__(self,
                 dtype: "DType | PointerTy",
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

    def __repr__(self):
        return f"TileTy(dtype={self.dtype}, shape={self.shape_value})"

    def __str__(self):
        shape_str = "(" + ','.join(str(x) for x in self._unwrapped_shape) + ")"
        return f"Tile[{self.dtype},{shape_str}]"


def make_tile_ty(dtype, shape: Sequence[int]) -> TileTy:
    shape = TupleTy(tuple(SizeTy(x) for x in shape))
    return TileTy(dtype, shape)


# ============== Array Type ===============


def array_size_type() -> Type:
    from .._datatype import int32
    return TileTy(int32, TupleTy(()))


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

    def is_aggregate(self) -> bool:
        # Even though arrays are actually represented with TensorViews, they can't be
        # propagated through control flow. So we need to be able to unpack the array
        # into its individual (base_ptr, *shape, *strides) values.
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        base_ptr_ty = PointerTy(self.dtype)
        base_ptr_tile_ty = TileTy(base_ptr_ty, TupleTy(()))
        size_ty = array_size_type()
        return (base_ptr_tile_ty,) + (size_ty,) * (self.ndim * 2)

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        from .ir import ArrayValue
        assert len(items) == 1 + 2 * self.ndim
        return ArrayValue(items[0], items[1:self.ndim + 1], items[self.ndim + 1:])

    def unify(self, other: "ArrayTy") -> Optional["ArrayTy"]:
        if self.dtype != other.dtype or self.ndim != other.ndim:
            return None

        shape = TupleTy(tuple(s1 if s1 == s2 else SizeTy()
                              for s1, s2 in zip(self.shape, other.shape, strict=True)))
        strides = TupleTy(tuple(s1 if s1 == s2 else SizeTy()
                                for s1, s2 in zip(self.strides, other.strides, strict=True)))

        elements_disjoint = self.elements_disjoint and other.elements_disjoint
        base_ptr_div_by = (
            None if (self.base_ptr_div_by is None or other.base_ptr_div_by is None)
            else math.gcd(self.base_ptr_div_by, other.base_ptr_div_by)
        )
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

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        from .._datatype import int32, int64
        ptr_ty = PointerTy(int64)
        ptr_tile_ty = TileTy(ptr_ty, TupleTy(()))
        len_ty = TileTy(int32, TupleTy(()))
        return ptr_tile_ty, len_ty

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        from .ir import ListValue
        base, length = items
        return ListValue(base, length)


# ============== Range Iter Type ===============


# FIXME: rename to RangeTy, this is not really an iterator
class RangeIterType(Type):
    def __init__(self, dtype):
        self.dtype = dtype

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        ty = make_tile_ty(self.dtype, ())
        return ty, ty, ty

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        from .ir import RangeValue
        start, stop, step = items
        return RangeValue(start, stop, step)

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

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return (self.self_ty,)

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        from .ir import BoundMethodValue
        [bound_self] = items
        return BoundMethodValue(bound_self)


@dataclass(frozen=True)
class EnumTy(Type):
    enum_ty: EnumMeta

    def __str__(self) -> str:
        return f"Enum[{self.enum_ty.__name__}]"


# Placeholder object for use as an inspect.Parameter's default value inside
# signatures of closures.
@dataclass(frozen=True)
class ClosureDefaultPlaceholder:
    # Index into `ClosureTy.default_value_types` and `ClosureValue.default_values`.
    default_value_index: int


@dataclass(frozen=True)
class LiveCapturedScope:
    depth: int
    local_scope: "LocalScope"


@dataclass(frozen=True)
class ClosureTy(Type):
    func_hir: "hir.Function"
    default_value_types: tuple[Type, ...]

    # Lists all enclosing functions' scopes that are still live.
    captured_scopes: tuple[LiveCapturedScope, ...]

    frozen_capture_types_by_depth: tuple[tuple[Type, ...] | None]

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return (
            *self.default_value_types,
            *(t for types in self.frozen_capture_types_by_depth
              if types is not None for t in types)
        )

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        from .ir import ClosureValue
        it = iter(items)
        default_values = tuple(next(it) for _ in self.default_value_types)
        frozen_captures_by_depth = tuple(None if types is None else tuple(next(it) for _ in types)
                                         for types in self.frozen_capture_types_by_depth)
        assert next(it, None) is None
        return ClosureValue(default_values=default_values,
                            frozen_captures_by_depth=frozen_captures_by_depth)

    def __str__(self):
        ret = f"Closure[{self.func_hir.desc.short_str()}"
        if len(self.default_value_types) > 0:
            default_strings = []
            for p in self.func_hir.signature.parameters.values():
                if p.default is not inspect.Parameter.empty:
                    assert isinstance(p.default, ClosureDefaultPlaceholder)
                    default_ty = self.default_value_types[p.default.default_value_index]
                    default_strings.append(f"'{p.name}': {default_ty}")
            ret += ", defaults={" + ", ".join(default_strings) + "}"
        if any(x is not None and len(x) > 0 for x in self.frozen_capture_types_by_depth):
            capture_strings = []
            for types, local_indices, parent_func in zip(self.frozen_capture_types_by_depth,
                                                         self.func_hir.captures_by_depth,
                                                         self.func_hir.enclosing_funcs,
                                                         strict=True):
                if types is None:
                    continue
                for ty, idx in zip(types, local_indices, strict=True):
                    name = parent_func.local_names[idx]
                    capture_strings.append(f"'{name}': {ty}")
            ret += ", frozen_captures={" + ", ".join(capture_strings) + "}"

        return ret + "]"
