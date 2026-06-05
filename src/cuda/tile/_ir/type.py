# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum
import inspect
import dataclasses
import os
from dataclasses import dataclass
from enum import EnumMeta
from types import ModuleType, FunctionType, BuiltinFunctionType, MethodType, MappingProxyType
from typing import Any, Callable, Optional, Sequence, Tuple, Iterator, Mapping
from functools import reduce
import operator

from typing import TYPE_CHECKING

from typing_extensions import override

from cuda.tile._exception import Loc, TileTypeError, TileValueError
from cuda.tile._memory_model import MemorySpace
from cuda.tile._stub import Tile, Array
from cuda.tile._numeric_semantics import PaddingMode
from .aggregate_value import AggregateValue
from cuda.tile._datatype import DType, PointerInfo

if TYPE_CHECKING:
    from cuda.tile._ir.ir import Var, TypingHooks
    from cuda.tile._ir import hir
    from cuda.tile._ir.scope import LocalScope


import cuda.tile._bytecode as bc


class Type:
    def make_symbol(self, var: "Var") -> Any:
        raise TileTypeError(f"Objects of type {self} are not supported at compile time")

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


class Symbol:
    def __init__(self, var: "Var"):
        self._var = var


def var2sym(var: "Var") -> Any:
    if var.is_constant():
        return var.get_constant()
    return var.get_type().make_symbol(var)


class TensorLikeTy(Type):
    """
    Base class for all tensor-like types, e.g. tiles, loosely typed scalars, etc.
    """
    def tensor_dtype(self) -> "DType":
        raise NotImplementedError()

    def tensor_shape(self) -> tuple[int, ...]:
        raise NotImplementedError()


@dataclass
class LooselyTypedScalar(TensorLikeTy):
    value: Any

    @override
    def tensor_dtype(self) -> "DType":
        from .typing_support import dtype_of_constant_scalar
        return dtype_of_constant_scalar(self.value)

    @override
    def tensor_shape(self) -> tuple[int, ...]:
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


# ============== Type of PointerInfo ==========

@dataclass(frozen=True)
class PointerInfoTy(Type):
    info: "PointerInfo"

    def __repr__(self):
        return f"<{self.info}>"


# ============== Tuple ===============

class TupleTy(Type):
    def __init__(self, value_types: Sequence[Type]):
        self._value_types = tuple(value_types)

    def make_symbol(self, var: "Var") -> Symbol:
        tup_val = var.get_aggregate()
        assert isinstance(tup_val, TupleValue)
        return tuple(var2sym(x) for x in tup_val.items)

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return self._value_types

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
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


@dataclass
class TupleValue(AggregateValue):
    items: tuple["Var", ...]

    def as_tuple(self) -> tuple["Var", ...]:
        return self.items


# ============== Dictionary (limited support for **kwargs) ==============

@dataclass(frozen=True)
class DictTy(Type):
    keys: tuple[str, ...]
    value_types: tuple[Type, ...]

    def make_symbol(self, var: "Var"):
        dict_val = var.get_aggregate()
        assert isinstance(dict_val, DictValue)
        items = [(k, var2sym(v)) for k, v in zip(self.keys, dict_val.values, strict=True)]
        return MappingProxyType(dict(items))

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return self.value_types

    def make_aggregate_value(self, values: tuple["Var", ...]) -> "AggregateValue":
        assert len(values) == len(self.keys)
        return DictValue(values)

    def __str__(self):
        return (
                "dict["
                + ", ".join(f"{k}: {ty}"
                            for k, ty in zip(self.keys, self.value_types, strict=True))
                + "]"
        )


@dataclass
class DictValue(AggregateValue):
    values: tuple["Var", ...]


# ============== Dataclass ===============


@dataclass(frozen=True)
class DataclassTy(Type):
    cls: type
    field_types: tuple[Type, ...]

    def make_symbol(self, var: "Var") -> Symbol:
        dc_val = var.get_aggregate()
        assert isinstance(dc_val, DataclassValue)
        return self.cls(**{f.name: var2sym(dc_val.get_field(f.name))
                           for f in dataclasses.fields(self.cls)})

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return self.field_types

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        from .typing_support import get_dataclass_info
        return DataclassValue(items, get_dataclass_info(self.cls))

    def __str__(self):
        return (
            self.cls.__name__ + "["
            + ", ".join(f"{f.name}: {ty}"
                        for f, ty in zip(dataclasses.fields(self.cls),
                                         self.field_types, strict=True))
            + "]"
        )


@dataclass(frozen=True)
class DataclassInfo:
    cls: type
    field_names: Sequence[str]
    field_name_to_idx: Mapping[str, int]
    init_signature: inspect.Signature


@dataclass
class DataclassValue(AggregateValue):
    items: tuple["Var", ...]
    info: DataclassInfo

    def as_tuple(self) -> tuple["Var", ...]:
        return self.items

    def get_field(self, name: str):
        return self.items[self.info.field_name_to_idx[name]]


# ============== Formatted String Type ===============

@dataclass(frozen=True)
class FormattedPiece:
    """A single typed placeholder in a formatted string."""
    value_idx: int          # index into FormattedStringTy.value_types
    format_spec: str | None  # None = type-inferred; otherwise Python format spec (e.g. '.2f')


@dataclass(frozen=True)
class StringFormat:
    """Immutable format template for a formatted string value."""
    pieces: tuple[str | FormattedPiece, ...]


@dataclass(frozen=True)
class FormattedStringTy(Type):
    format: "StringFormat"
    value_types: tuple
    has_single_quote: bool = False
    has_double_quote: bool = False

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple:
        return self.value_types

    def make_aggregate_value(self, items: tuple) -> "AggregateValue":
        return FormattedStringValue(self.format, items)

    def __str__(self):
        parts = []
        for piece in self.format.pieces:
            if isinstance(piece, str):
                parts.append(piece)
            else:
                ty = self.value_types[piece.value_idx]
                if piece.format_spec is not None:
                    parts.append(f"{{<{ty}>:{piece.format_spec}}}")
                else:
                    parts.append(f"{{<{ty}>}}")
        return 'FormattedString<"' + "".join(parts) + '">'


def size_to_bytecode(s: Optional[int]) -> int:
    return bc.DYNAMIC_SHAPE if s is None else s


@dataclass
class FormattedStringValue(AggregateValue):
    format: "Any"  # StringFormat from type.py
    values: tuple["Var"]

    def as_tuple(self) -> tuple["Var"]:
        return self.values


# ============== Tile Type ===============


class TileTy(TensorLikeTy):
    def __new__(cls, dtype: "DType", shape: Sequence[int] = ()) -> "TileTy":
        shape = tuple(shape)
        try:
            return _tile_ty_cache[(dtype, shape)]
        except KeyError:
            pass

        assert isinstance(dtype, DType)
        ret = object.__new__(cls)
        ret.dtype = dtype
        ret.shape = shape
        _tile_ty_cache[(dtype, shape)] = ret
        return ret

    @override
    def tensor_dtype(self) -> "DType":
        return self.dtype

    @override
    def tensor_shape(self) -> tuple[int, ...]:
        return self.shape

    @override
    def make_symbol(self, var: "Var") -> Symbol:
        return SymbolicTile(var)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numel(self):
        return reduce(operator.mul, self.shape, 1)

    def __eq__(self, other: Type):
        if isinstance(other, TileTy):
            return self.dtype == other.dtype and self.shape == other.shape
        return False

    def __hash__(self):
        return hash(("TileTy", self.dtype, self.shape))

    def __repr__(self):
        return f"TileTy(dtype={self.dtype}, shape={self.shape})"

    def __str__(self):
        shape_str = "(" + ','.join(str(x) for x in self.shape) + ")"
        return f"Tile[{self.dtype},{shape_str}]"


_tile_ty_cache: dict[tuple["DType", tuple[int, ...]], TileTy] = dict()


class SymbolicTile(Symbol, Tile):
    def __init__(self, var: "Var"):
        Symbol.__init__(self, var)

    @property
    def dtype(self):
        ty = self._var.get_type()
        assert isinstance(ty, TileTy)
        return ty.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        ty = self._var.get_type()
        assert isinstance(ty, TileTy)
        return ty.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __bool__(self):
        raise TileValueError("Symbolic tile has no concrete value and thus cannot be converted"
                             " to boolean")

    def __int__(self):
        raise TileValueError("Symbolic tile has no concrete value and thus cannot be converted"
                             " to an integer")

    def __float__(self):
        raise TileValueError("Symbolic tile has no concrete value and thus cannot be converted"
                             " to a float")

    def __index__(self):
        raise TileValueError("Symbolic tile has no concrete value and thus cannot be converted"
                             " to an integer")

    def __repr__(self):
        return f"<tile[{self.dtype}, {self.shape}]>"


# ============== Array Type ===============


class ArrayTy(Type):
    def __init__(self,
                 dtype: "DType",
                 /,
                 shape: Tuple[Optional[int], ...],
                 strides: Tuple[Optional[int], ...],
                 typing_hooks: "TypingHooks",
                 index_dtype=None,
                 memory_space: MemorySpace = MemorySpace.GENERIC):
        from .._datatype import int32, DType
        assert isinstance(dtype, DType)
        self.dtype = dtype
        self.shape = shape
        self.strides = strides
        self.index_dtype = int32 if index_dtype is None else index_dtype
        self.memory_space = memory_space
        self.typing_hooks = typing_hooks

    def make_symbol(self, var: "Var") -> Any:
        return SymbolicArray(var)

    def is_aggregate(self) -> bool:
        # Even though arrays are actually represented with TensorViews, they can't be
        # propagated through control flow. So we need to be able to unpack the array
        # into its individual (base_ptr, *shape, *strides) values.
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        from .._datatype import pointer_dtype
        base_ptr_ty = pointer_dtype(self.dtype, self.memory_space)
        base_ptr_tile_ty = self.typing_hooks.get_tensor_like_type(base_ptr_ty, ())
        size_ty = self.typing_hooks.get_tensor_like_type(self.index_dtype, ())
        return (base_ptr_tile_ty,) + (size_ty,) * (self.ndim * 2)

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        assert len(items) == 1 + 2 * self.ndim
        return ArrayValue(items[0], items[1:self.ndim + 1], items[self.ndim + 1:])

    @property
    def ndim(self):
        return len(self.shape)

    def __eq__(self, other: Type):
        return (isinstance(other, ArrayTy)
                and self.dtype == other.dtype
                and self.shape == other.shape
                and self.strides == other.strides
                and self.index_dtype == other.index_dtype
                and self.memory_space == other.memory_space)

    def __hash__(self):
        return hash(("ArrayTy", self.dtype, self.shape, self.strides, self.index_dtype,
                     self.memory_space))

    def __str__(self):
        from .._datatype import int32
        shape_str = ('?' if x is None else str(x) for x in self.shape)
        shape_str = "(" + ','.join(shape_str) + ")"
        strides_str = ('?' if x is None else str(x) for x in self.strides)
        strides_str = "(" + ','.join(strides_str) + ")"
        indexty_str = "" if self.index_dtype == int32 else f",index_dtype={self.index_dtype}]"
        memspc = "" if self.memory_space == MemorySpace.GENERIC else f", {self.memory_space}"
        return f"Array[{self.dtype},{shape_str}:{strides_str}{indexty_str}{memspc}]"


@dataclass
class ArrayValue(AggregateValue):
    base_ptr: "Var"
    shape: tuple["Var", ...]
    strides: tuple["Var", ...]

    def as_tuple(self) -> tuple["Var", ...]:
        return self.base_ptr, *self.shape, *self.strides


class SymbolicArray(Symbol, Array):
    def __init__(self, var: "Var"):
        Symbol.__init__(self, var)

    @property
    def dtype(self):
        ty = self._var.get_type()
        assert isinstance(ty, ArrayTy)
        return ty.dtype

    @property
    def shape(self):
        agg = self._var.get_aggregate()
        assert isinstance(agg, ArrayValue)
        return tuple(var2sym(v) for v in agg.shape)

    @property
    def strides(self):
        agg = self._var.get_aggregate()
        assert isinstance(agg, ArrayValue)
        return tuple(var2sym(v) for v in agg.strides)

    @property
    def ndim(self) -> int:
        ty = self._var.get_type()
        assert isinstance(ty, ArrayTy)
        return ty.ndim

    def __repr__(self):
        ty = self._var.get_type()
        assert isinstance(ty, ArrayTy)

        shape_str = ", ".join("?" if s is None else str(s) for s in ty.shape)

        return f"<array[{ty.dtype}, ({shape_str})]>"


# ============== PartitionView Type ===============


@dataclass(frozen=True)
class PartitionViewTy(Type):
    array_ty: ArrayTy
    tile_shape: tuple[int, ...]
    order: tuple[int, ...]
    padding_mode: PaddingMode

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return self.array_ty.aggregate_item_types()

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        return self.array_ty.make_aggregate_value(items)

    @property
    def dtype(self):
        return self.array_ty.dtype

    def __str__(self):
        return (f"PartitionView[{self.array_ty},tile_shape={self.tile_shape},order={self.order},"
                f"padding_mode={self.padding_mode}]")


# ============== StridedView Type ===============


@dataclass(frozen=True)
class StridedViewTy(Type):
    array_ty: ArrayTy
    tile_shape: tuple[int, ...]
    traversal_steps: tuple[int, ...]
    order: tuple[int, ...]
    padding_mode: PaddingMode

    def __str__(self):
        return (f"StridedView[{self.array_ty},tile_shape={self.tile_shape},"
                f"traversal_steps={self.traversal_steps},order={self.order},"
                f"padding_mode={self.padding_mode}]")


# ============== GatherScatterView Type ===============


@dataclass(frozen=True)
class GatherScatterViewTy(Type):
    array_ty: ArrayTy
    tile_shape: tuple[int, ...]
    sparse_dim: int
    padding_mode: PaddingMode

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return self.array_ty.aggregate_item_types()

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        return self.array_ty.make_aggregate_value(items)

    @property
    def dtype(self):
        return self.array_ty.dtype

    def __str__(self):
        return (f"GatherScatterView[{self.array_ty},tile_shape={self.tile_shape},"
                f"sparse_dim={self.sparse_dim},padding_mode={self.padding_mode}]")


# ============== IndexSlice Type ===============


@dataclass(frozen=True)
class IndexSliceTy(Type):
    """Type of a ct.Slice(start, length)."""
    start_ty: "Type"
    length_ty: "Type"

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return (self.start_ty, self.length_ty)

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        assert len(items) == 2
        return IndexSliceValue(items[0], items[1])

    def __str__(self) -> str:
        return f"IndexSlice[start_ty={self.start_ty}, length_ty={self.length_ty}]"


@dataclass
class IndexSliceValue(AggregateValue):
    start: "Var"
    length: "Var"

    def as_tuple(self) -> tuple["Var", ...]:
        return (self.start, self.length)


# ============== TiledView Type ===============

@dataclass(frozen=True)
class TiledViewTy(Type):
    array_ty: ArrayTy
    tile_shape: tuple[int, ...]
    padding_mode: PaddingMode
    traversal_steps: Optional[tuple[int, ...]] = None

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return (self.array_ty,)

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        [array] = items
        return TiledViewValue(array)

    @property
    def ndim(self):
        return self.array_ty.ndim

    @property
    def dtype(self):
        return self.array_ty.dtype

    def __str__(self):
        return (f"TiledView[{self.array_ty},tile_shape={self.tile_shape},"
                f"padding_mode={self.padding_mode},traversal_steps={self.traversal_steps}]")


@dataclass
class TiledViewValue(AggregateValue):
    array: "Var"

    def as_tuple(self) -> tuple["Var", ...]:
        return (self.array,)


# ============== Raw Array Memory Type ===============


@dataclass(frozen=True)
class RawArrayMemoryTy(Type):
    """Type for a RawArrayMemory object that allows load/store by element offset (no index math)."""
    dtype: "DType"

    def __post_init__(self):
        from .._datatype import DType
        assert isinstance(self.dtype, DType)

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        from .._datatype import pointer_dtype
        base_ptr_dtype = pointer_dtype(self.dtype)
        base_ptr_tile_ty = TileTy(base_ptr_dtype)
        return (base_ptr_tile_ty,)

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        assert len(items) == 1
        return RawArrayMemoryValue(items[0])

    def __str__(self):
        return f"RawArrayMemory[{self.dtype}]"


@dataclass
class RawArrayMemoryValue(AggregateValue):
    base_ptr: "Var"

    def as_tuple(self) -> tuple["Var", ...]:
        return (self.base_ptr,)


# ============== List Type ===============


@dataclass(frozen=True)
class ListTy(Type):
    item_type: Type

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        from .._datatype import int32, int64, pointer_dtype
        ptr_dtype = pointer_dtype(int64)
        ptr_tile_ty = TileTy(ptr_dtype)
        len_ty = TileTy(int32)
        return ptr_tile_ty, len_ty

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        base, length = items
        return ListValue(base, length)


@dataclass
class ListValue(AggregateValue):
    base_ptr: "Var"
    length: "Var"

    def as_tuple(self) -> tuple["Var", ...]:
        return self.base_ptr, self.length


# ============== Range Iter Type ===============


# FIXME: rename to RangeTy, this is not really an iterator
class RangeIterType(Type):
    def __init__(self, dtype):
        self.dtype = dtype

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        ty = TileTy(self.dtype)
        return ty, ty, ty

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        start, stop, step = items
        return RangeValue(start, stop, step)

    def __str__(self):
        return f"Range<{self.dtype}>"

    def __eq__(self, other: Type):
        return isinstance(other, RangeIterType) and other.dtype == self.dtype


@dataclass
class RangeValue(AggregateValue):
    start: "Var"
    stop: "Var"
    step: "Var"

    def as_tuple(self) -> tuple["Var", ...]:
        return self.start, self.stop, self.step


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
    func: FunctionType | BuiltinFunctionType

    def make_symbol(self, var: "Var") -> Any:
        self_sym = var2sym(var.get_aggregate().bound_self)
        return MethodType(self.func, self_sym)

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return (self.self_ty,)

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
        [bound_self] = items
        return BoundMethodValue(bound_self)


@dataclass
class BoundMethodValue(AggregateValue):
    bound_self: "Var"

    def as_tuple(self) -> tuple["Var", ...]:
        return (self.bound_self,)


@dataclass(frozen=True)
class EnumTy(Type):
    enum_ty: EnumMeta

    def __str__(self) -> str:
        return f"Enum[{self.enum_ty.__name__}]"


class ContextManagerLifecycle(enum.IntEnum):
    FRESH = 0
    ENTERED = 1
    EXITED = 2


@dataclass(eq=False)
class ContextManagerState:
    exit_callback: Callable[[], None] = lambda: None
    lifecycle: ContextManagerLifecycle = ContextManagerLifecycle.FRESH


class ContextManagerTy(Type):
    def get_context_manager_state(self) -> ContextManagerState:
        raise NotImplementedError()


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

    def make_symbol(self, var: "Var") -> Symbol:
        return SymbolicClosure(var)

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple["Type", ...]:
        return (
            *self.default_value_types,
            *(t for types in self.frozen_capture_types_by_depth
              if types is not None for t in types)
        )

    def make_aggregate_value(self, items: tuple["Var", ...]) -> "AggregateValue":
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


@dataclass
class ClosureValue(AggregateValue):
    # Default values of parameters. These need to be carried by the closure's value
    # because default expressions are evaluated at definition time, not when the closure is called.
    # Should have the same length as the corresponding `ClosureTy.default_value_types`.
    default_values: tuple["Var", ...]

    # Tuple of the same length as `ty.func_hir.enclosing_functions`
    # and `ty.frozen_capture_types_by_depth`, where `ty` is the `ClosureTy` of this closure.
    #
    # For each depth `i`, `frozen_captures_by_depth[i]` is either:
    #   - None: means the enclosing function's LocalScope is still live;
    #   - tuple[Var, ...]: means the enclosing function's LocalScope is no longer live.
    #       The tuple contains the final values of the variables captured from the enclosing
    #       function's scope. Its length should be the same as `ty.func_hir.captures_by_depth`.
    frozen_captures_by_depth: tuple[tuple["Var", ...] | None, ...]

    def as_tuple(self) -> tuple["Var", ...]:
        return (
            *self.default_values,
            *(v for values in self.frozen_captures_by_depth
              if values is not None for v in values)
        )


class SymbolicClosure(Symbol):
    def __repr__(self):
        ty = self._var.get_type()
        assert isinstance(ty, ClosureTy)
        desc = ty.func_hir.desc
        what = "lambda" if desc.name is None else f"function '{desc.name}'"
        filename = os.path.basename(desc.filename)
        return f"<{what} @{filename}:{desc.line}>"

    def __call__(self, *args, **kwargs):
        from cuda.tile._dispatch_mode import DispatchMode
        return DispatchMode().get_current().call_tile_function_from_host(self, args, kwargs)
