# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import cache
from typing import Optional, Tuple
from enum import IntEnum

from cuda.tile._exception import TileTypeError
from cuda.tile._execution import function, stub
from cuda.tile._memory_model import MemorySpace
import cuda.tile._bytecode as bc


__all__ = ["bool_", "uint8", "uint16", "uint32", "uint64",
           "int8", "int16", "int32", "int64",
           "float16", "float32", "float64",
           "bfloat16", "tfloat32", "float8_e4m3fn", "float8_e5m2",
           "float8_e8m0fnu", "float4_e2m1fn", "DType"]


class DType:
    """A *data type* (or *dtype*) describes the type of the objects of an |array|, |tile|, or
    operation.

    |Dtypes| determine how values are stored in memory and how operations on those values are
    performed.
    |Dtypes| are immutable.

    |Dtypes| can be used in |host code| and |tile code|.
    They can be |kernel| parameters.
    """

    def __new__(self):
        raise TypeError("DType objects cannot be created")

    def __reduce__(self):
        return _define_dtype, (self.__name__, _dtype_defs[self])

    @property
    @function(host=True, tile=False)
    def bitwidth(self):
        """The number of bits in an element of the |data type|."""
        return _dtype_defs[self].bitwidth

    @property
    @function(host=True, tile=False)
    def name(self):
        """The name of the |data type|."""
        return self.__name__

    @function(host=True, tile=False)
    def __repr__(self):
        return f"<DType '{self.__name__}'>"

    @function(host=True, tile=False)
    def __str__(self):
        return self.__name__

    @stub
    def __call__(self, value, /):
        """Construct a Scalar of this |data type| from a value."""


class NumericDTypeCategory(IntEnum):
    Boolean = 0
    Integral = 1
    Float = 2
    RestrictedFloat = 3

    @property
    def pytype(self) -> type:
        match self:
            case NumericDTypeCategory.Boolean: return bool
            case NumericDTypeCategory.Integral: return int
            case NumericDTypeCategory.Float: return float
            case NumericDTypeCategory.RestrictedFloat: return float
            case _: assert False, self

    @property
    def arithmetic(self) -> bool:
        match self:
            case NumericDTypeCategory.Boolean: return True
            case NumericDTypeCategory.Integral: return True
            case NumericDTypeCategory.Float: return True
            case NumericDTypeCategory.RestrictedFloat: return False
            case _: assert False, self


class IntegerInfo:
    """
    Machine information for integer data types, similar to numpy.iinfo.
    """
    @stub(host=True)
    def __init__(self, dtype: DType):
        definition = _dtype_defs[dtype]
        if not isinstance(definition, _IntegerDTypeDefinition):
            raise TypeError(f"'{dtype}' is not an integer dtype")
        self._dtype = dtype
        self._definition = definition

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def bits(self) -> int:
        return self._definition.bitwidth

    @property
    def min(self) -> int:
        return self._definition.get_min_value()

    @property
    def max(self) -> int:
        return self._definition.get_max_value()

    def __eq__(self, other):
        return isinstance(other, IntegerInfo) and self._dtype == other._dtype

    def __hash__(self):
        return hash(self._dtype)


@dataclass(frozen=True, kw_only=True)
class _DTypeDefinition:
    bitwidth: int
    numeric_category: NumericDTypeCategory | None = None
    simple_bytecode_type: bc.SimpleType | None = None


@dataclass(frozen=True, kw_only=True)
class _IntegerDTypeDefinition(_DTypeDefinition):
    signed: bool

    def get_min_value(self) -> int:
        return -(1 << (self.bitwidth - 1)) if self.signed else 0

    def get_max_value(self) -> int:
        return (1 << (self.bitwidth - 1)) - 1 if self.signed else (1 << self.bitwidth) - 1


@dataclass(frozen=True, kw_only=True)
class _PointerDTypeDefinition(_DTypeDefinition):
    pointee_dtype: DType | None  # None for opaque pointers
    memory_space: MemorySpace


_dtype_defs: dict[DType, _DTypeDefinition] = dict()
_dtype_by_name: dict[str, DType] = dict()
_dtype_lock = threading.Lock()


def _define_dtype(name: str, definition: _DTypeDefinition) -> DType:
    assert isinstance(definition, _DTypeDefinition)
    with _dtype_lock:
        if name in _dtype_by_name:
            existing = _dtype_by_name[name]
            assert _dtype_defs[existing] == definition
            return existing

        dtype = object.__new__(DType)
        dtype.__name__ = name
        _dtype_defs[dtype] = definition
        _dtype_by_name[name] = dtype
        return dtype


def _numeric_dtype(name: str,
                   bitwidth: int,
                   category: NumericDTypeCategory,
                   bc_type: bc.SimpleType) -> DType:
    definition = _DTypeDefinition(bitwidth=bitwidth,
                                  numeric_category=category,
                                  simple_bytecode_type=bc_type)
    return _define_dtype(name, definition)


def _integer_dtype(name: str, bitwidth: int, signed: bool, bc_type: bc.SimpleType) -> DType:
    definition = _IntegerDTypeDefinition(bitwidth=bitwidth,
                                         numeric_category=NumericDTypeCategory.Integral,
                                         simple_bytecode_type=bc_type,
                                         signed=signed)
    dtype = _define_dtype(name, definition)
    signedness = "signed" if signed else "unsigned"
    dtype.__doc__ = (f"{bitwidth}-bit {signedness} integer |arithmetic dtype| with values"
                     f" on the interval"
                     f" [{definition.get_min_value()}, +{definition.get_max_value()}]")
    return dtype


bool_ = _numeric_dtype('bool_', 8, NumericDTypeCategory.Boolean, bc.SimpleType.I1)
bool_.__doc__ = """A 8-bit |arithmetic dtype| (``True`` or ``False``)."""

uint8 = _integer_dtype('uint8', 8, False, bc.SimpleType.I8)
uint16 = _integer_dtype('uint16', 16, False, bc.SimpleType.I16)
uint32 = _integer_dtype('uint32', 32, False, bc.SimpleType.I32)
uint64 = _integer_dtype('uint64', 64, False, bc.SimpleType.I64)
int8 = _integer_dtype('int8', 8, True, bc.SimpleType.I8)
int16 = _integer_dtype('int16', 16, True, bc.SimpleType.I16)
int32 = _integer_dtype('int32', 32, True, bc.SimpleType.I32)
int64 = _integer_dtype('int64', 64, True, bc.SimpleType.I64)

float16 = _numeric_dtype('float16', 16, NumericDTypeCategory.Float, bc.SimpleType.F16)
float16.__doc__ = """A IEEE 754 half-precision (16-bit) binary floating-point |arithmetic dtype| \
(see |IEEE 754-2019|)."""

float32 = _numeric_dtype('float32', 32, NumericDTypeCategory.Float, bc.SimpleType.F32)
float32.__doc__ = """A IEEE 754 single-precision (32-bit) binary floating-point |arithmetic dtype| \
(see |IEEE 754-2019|)."""

float64 = _numeric_dtype('float64', 64, NumericDTypeCategory.Float, bc.SimpleType.F64)
float64.__doc__ = """A IEEE 754 double-precision (64-bit) binary floating-point |arithmetic dtype| \
(see |IEEE 754-2019|)."""

bfloat16 = _numeric_dtype('bfloat16', 16, NumericDTypeCategory.Float, bc.SimpleType.BF16)
bfloat16.__doc__ = """A 16-bit floating-point |arithmetic dtype| with 1 sign bit, 8 exponent bits, \
and 7 mantissa bits."""

tfloat32 = _numeric_dtype("tfloat32", 32, NumericDTypeCategory.RestrictedFloat, bc.SimpleType.TF32)
tfloat32.__doc__ = """A 32-bit tensor floating-point |numeric dtype| with 1 sign \
bit, 8 exponent bits, and 10 mantissa bits (19-bit representation stored in 32-bit container)."""

float8_e4m3fn = _numeric_dtype("float8_e4m3fn", 8, NumericDTypeCategory.RestrictedFloat,
                               bc.SimpleType.F8E4M3FN)
float8_e4m3fn.__doc__ = """An 8-bit floating-point |numeric dtype| with 1 sign bit, \
4 exponent bits, and 3 mantissa bits."""

float8_e5m2 = _numeric_dtype("float8_e5m2", 8, NumericDTypeCategory.RestrictedFloat,
                             bc.SimpleType.F8E5M2)
float8_e5m2.__doc__ = """An 8-bit floating-point |numeric dtype| with 1 sign bit, \
5 exponent bits, and 2 mantissa bits."""

float8_e8m0fnu = _numeric_dtype("float8_e8m0fnu", 8, NumericDTypeCategory.RestrictedFloat,
                                bc.SimpleType.F8E8M0FNU)
float8_e8m0fnu.__doc__ = """An 8-bit floating-point |numeric dtype| with no sign bit, \
8 exponent bits, and 0 mantissa bits."""

float4_e2m1fn = _numeric_dtype("float4_e2m1fn", 4, NumericDTypeCategory.RestrictedFloat,
                               bc.SimpleType.F4E2M1FN)
float4_e2m1fn.__doc__ = """A 4-bit floating-point |numeric dtype| with 1 sign bit, \
2 exponent bits, and 1 mantissa bit."""


default_int_type = int32
default_float_type = float32


#: Unsigned integral |dtypes|. These |dtypes| are arithmetic.
unsigned_integral_dtypes = [uint64, uint32, uint16, uint8]

#: Signed integral |dtypes|. These |dtypes| are arithmetic.
signed_integral_dtypes = [int64, int32, int16, int8]


def is_numeric(t: DType) -> bool:
    return _dtype_defs[t].numeric_category is not None


def numeric_dtype_category(t: DType) -> NumericDTypeCategory:
    cat = _dtype_defs[t].numeric_category
    if cat is None:
        raise ValueError(f"{t} is not a numeric dtype")
    return cat


def dtype_simple_bytecode_type(t: DType) -> bc.SimpleType:
    ret = _dtype_defs[t].simple_bytecode_type
    assert ret is not None
    return ret


def is_boolean(t: DType) -> bool:
    return _dtype_defs[t].numeric_category == NumericDTypeCategory.Boolean


def is_integral(t: DType) -> bool:
    return _dtype_defs[t].numeric_category == NumericDTypeCategory.Integral


def is_signed(t: DType) -> bool:
    """Returns True if the |dtype| is a signed numeric type, such as a signed integer or
    a floating-point type."""
    info = _dtype_defs[t]
    match info.numeric_category:
        case None: return False
        case NumericDTypeCategory.Boolean: return False
        case NumericDTypeCategory.Integral:
            assert isinstance(info, _IntegerDTypeDefinition)
            return info.signed
        case NumericDTypeCategory.Float: return True
        case NumericDTypeCategory.RestrictedFloat: return True
        case _: assert False, info.numeric_category


def integer_dtype(bitwidth: int, *, signed: bool) -> DType:
    match bitwidth, signed:
        case 8, False: return uint8
        case 16, False: return uint16
        case 32, False: return uint32
        case 64, False: return uint64
        case 8, True: return int8
        case 16, True: return int16
        case 32, True: return int32
        case 64, True: return int64
        case _: raise ValueError(f"No such {'signed' if signed else 'unsigned'}"
                                 f" integer dtype of bitwidth {bitwidth}")


_signedness = (bc.Signedness.Unsigned, bc.Signedness.Signed)


def get_signedness(t: DType) -> bc.Signedness:
    return _signedness[is_signed(t)]


def is_float(t: DType) -> bool:
    return _dtype_defs[t].numeric_category in (NumericDTypeCategory.Float,
                                               NumericDTypeCategory.RestrictedFloat)


def is_unrestricted_float(t: DType) -> bool:
    return _dtype_defs[t].numeric_category == NumericDTypeCategory.Float


def is_restricted_float(t: DType) -> bool:
    return _dtype_defs[t].numeric_category == NumericDTypeCategory.RestrictedFloat


def is_arithmetic(t: DType) -> bool:
    """Returns True if the |dtype| supports general arithmetic operations such as
    addition, subtraction, multiplication, and division."""
    cat = _dtype_defs[t].numeric_category
    return cat is not None and cat.arithmetic


def broadcast_shapes(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    s1 = [1] * (len(s2) - len(s1)) + list(s1)

    result_shape = []
    for d1, d2 in zip(s1, s2):
        if d1 != d2:
            if d1 == 1:
                result_shape.append(d2)
            elif d2 == 1:
                result_shape.append(d1)
            else:
                raise TypeError(f"Broadcast shapes mismatch: {s1}, {s2}")
        else:
            result_shape.append(d1)
    return tuple(result_shape)


# ============= Arithmetic Promotion ==============

class _DTypePromotionImpl:

    # shorter alias to make the table
    b1 = bool_
    u8 = uint8
    u16 = uint16
    u32 = uint32
    u64 = uint64
    i8 = int8
    i16 = int16
    i32 = int32
    i64 = int64
    f16 = float16
    f32 = float32
    f64 = float64
    bf = bfloat16
    tf32 = tfloat32
    f8e4m3fn = float8_e4m3fn
    f8e5m2 = float8_e5m2
    f8e8m0fnu = float8_e8m0fnu
    f4e2m1fn = float4_e2m1fn
    na = None

    # Entries for restricted arithmetic dtypes will never be reached, but we need to keep them
    # for the table to be valid.

    # General rules
    # Cross categories: Bool -> Integral -> Float
    # Within categories: small bitwidth -> large bitwidth

    # Exceptions
    # Signed and unsigned requires explicit type cast
    # Restricted floats requires explicit type cast
    # Float16 and BFloat 16 requires explicit type cast

    _order = [
      b1,  u8,  u16, u32, u64, i8,  i16, i32, i64, f16, f32, f64, bf,  tf32, f8e4m3fn, f8e5m2,    f8e8m0fnu, f4e2m1fn   # noqa
    ]
    _common_dtype_table = [
     [b1,  u8,  u16, u32, u64, i8,  i16, i32, i64, f16, f32, f64, bf,  na,   na,       na,        na,        na],        # b1  # noqa
     [u8,  u8,  u16, u32, u64, na,  na,  na,  na,  f16, f32, f64, bf,  na,   na,       na,        na,        na],        # u8  # noqa
     [u16, u16, u16, u32, u64, na,  na,  na,  na,  f16, f32, f64, bf,  na,   na,       na,        na,        na],        # u16  # noqa
     [u32, u32, u32, u32, u64, na,  na,  na,  na,  f16, f32, f64, bf,  na,   na,       na,        na,        na],        # u32  # noqa
     [u64, u64, u64, u64, u64, na,  na,  na,  na,  f16, f32, f64, bf,  na,   na,       na,        na,        na],        # u64  # noqa
     [i8,  na,  na,  na,  na,  i8,  i16, i32, i64, f16, f32, f64, bf,  na,   na,       na,        na,        na],        # i8  # noqa
     [i16, na,  na,  na,  na,  i16, i16, i32, i64, f16, f32, f64, bf,  na,   na,       na,        na,        na],        # i16  # noqa
     [i32, na,  na,  na,  na,  i32, i32, i32, i64, f16, f32, f64, bf,  na,   na,       na,        na,        na],        # i32  # noqa
     [i64, na,  na,  na,  na,  i64, i64, i64, i64, f16, f32, f64, bf,  na,   na,       na,        na,        na],        # i64  # noqa
     [f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f32, f64, na,  na,   na,       na,        na,        na],        # f16  # noqa
     [f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f64, f32, na,   na,       na,        na,        na],        # f32  # noqa
     [f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, na,   na,       na,        na,        na],        # f64  # noqa
     [bf,  bf,  bf,  bf,  bf,  bf,  bf,  bf,  bf,  na,  f32, f64, bf,  na,   na,       na,        na,        na],        # bf  # noqa
     [na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  tf32, na,       na,        na,        na],        # tf32  # noqa
     [na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,   f8e4m3fn, na,        na,        na],        # f8e4m3fn  # noqa
     [na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,   na,       f8e5m2,    na,        na],        # f8e5m2  # noqa
     [na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,   na,       na,        f8e8m0fnu, na],        # f8e8m0fnu  # noqa
     [na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,   na,       na,        na,        f4e2m1fn],  # f4e2m1fn  # noqa
    ]

    @classmethod
    def promote_dtypes(cls, t1: DType, t2: DType, force_float: bool = False) -> DType:
        if t1 == t2 and (not force_float or is_float(t1)):
            return t1
        if is_restricted_float(t1) or is_restricted_float(t2):
            raise TileTypeError(
                f"Implicit promotion of {t1} and {t2} is not supported as it involves restricted "
                f"float dtypes. Please perform an explicit cast instead."
            )
        if is_pointer_dtype(t1) or is_pointer_dtype(t2):
            raise TileTypeError("Implicit promotion of pointer dtypes is not supported")
        idx1, idx2 = cls._order.index(t1), cls._order.index(t2)
        if idx1 >= len(cls._common_dtype_table) or idx2 >= len(cls._common_dtype_table[idx1]):
            raise IndexError(f"Invalid dtypes in common dtype table: {t1}, {t2}")
        ret = cls._common_dtype_table[idx1][idx2]
        if ret is None:
            msg = (f'Implicit promotion of {t1} and {t2} is not supported. '
                   'Please perform an explict cast instead.')
            raise TileTypeError(msg)
        return ret if not force_float or is_float(ret) else default_float_type


_mma_supported_dtypes = {
    float8_e4m3fn: (float16, float32),
    float8_e5m2: (float16, float32),
    float16: (float16, float32),
    bfloat16: (float32,),
    float32: (float32,),
    tfloat32: (float32,),
    float64: (float64,),
    int8: (int32,),
    uint8: (int32,),
}


def _resolve_mma_supported_dtype(x_dtype: DType,
                                 y_dtype: DType,
                                 acc_dtype: Optional[DType] = None) -> DType:
    if x_dtype != y_dtype and (x_dtype not in (int8, uint8) or y_dtype not in (int8, uint8)):
        raise TileTypeError(f"x and y must have the same dtype unless they are int8/uint8, "
                            f"got {x_dtype} {y_dtype}")
    if x_dtype not in _mma_supported_dtypes:
        candidates = ",".join(str(x) for x in _mma_supported_dtypes.keys())
        raise TileTypeError(f"Unsupported input dtype {x_dtype}, "
                            f"supported dtypes are {candidates}")
    if acc_dtype is not None:
        candidates = _mma_supported_dtypes[x_dtype]
        if acc_dtype not in candidates:
            raise TileTypeError(f"Unsupported acc dtype {acc_dtype}, "
                                f"supported dtypes are {candidates}")
    else:
        acc_dtype = _mma_supported_dtypes[x_dtype][0]
    return acc_dtype


_mma_scaled_supported_dtypes = {
    # operand dtype -> {scale dtype: (result dtype, scaling block sizes)}
    float8_e4m3fn: {float8_e8m0fnu: (float32, (32,))},
    float8_e5m2:   {float8_e8m0fnu: (float32, (32,))},
    float4_e2m1fn: {float8_e8m0fnu: (float32, (16, 32)),
                    float8_e4m3fn:  (float32, (16,))},
}


def _resolve_mma_scaled_supported_dtype(x_dtype: DType,
                                        x_scale_dtype: DType,
                                        y_dtype: DType,
                                        y_scale_dtype: DType,
                                        acc_dtype: DType):
    if x_dtype != y_dtype:
        raise TileTypeError(
            f"x and y must have the same dtype, got {x_dtype} and {y_dtype}")
    if x_scale_dtype != y_scale_dtype:
        raise TileTypeError(
            f"x_scale and y_scale must have the same dtype, "
            f"got {x_scale_dtype} and {y_scale_dtype}")
    if x_dtype not in _mma_scaled_supported_dtypes:
        candidates = ", ".join(str(d) for d in _mma_scaled_supported_dtypes.keys())
        raise TileTypeError(
            f"Unsupported input dtype {x_dtype} for mma_scaled, "
            f"supported input dtypes are {candidates}")
    scale_candidates = _mma_scaled_supported_dtypes[x_dtype]
    if x_scale_dtype not in scale_candidates:
        candidate_names = ", ".join(str(s) for s in scale_candidates.keys())
        raise TileTypeError(
            f"Unsupported scale dtype {x_scale_dtype} for input dtype {x_dtype}, "
            f"supported scale dtypes are {candidate_names}")
    expected_acc, _ = scale_candidates[x_scale_dtype]
    if acc_dtype != expected_acc:
        raise TileTypeError(
            f"Unsupported acc dtype {acc_dtype} for mma_scaled, "
            f"expected {expected_acc}")


def _get_mma_scaled_scaling_block_sizes(data_dtype, scale_dtype) -> Tuple[int, ...]:
    assert data_dtype in _mma_scaled_supported_dtypes
    scale_candidates = _mma_scaled_supported_dtypes[data_dtype]
    assert scale_dtype in scale_candidates
    _, scaling_block_sizes = scale_candidates[scale_dtype]
    return scaling_block_sizes


# =============== Documentation Generator ================

def _is_public(dtype_name: str) -> bool:
    import cuda.tile
    return hasattr(cuda.tile, dtype_name)


def _generate_rst_dtype_promotion_table() -> str:
    """Generate an RST table representation of the dtype promotion rules."""
    # Skip dtypes not exposed in cuda.tile yet. Promomotion table is append only.
    return _generate_rst_table()


def _get_all_public_numeric_dtypes_for_docs():
    return [dtype for dtype in _DTypePromotionImpl._order if _is_public(dtype.name)]


def _generate_rst_numeric_dtypes() -> str:
    """Generate RST documentation for numeric datatypes."""
    import cuda.tile
    content = []

    for dtype in _get_all_public_numeric_dtypes_for_docs():
        # Skip dtypes not exposed in cuda.tile yet
        if not hasattr(cuda.tile, dtype.name):
            continue
        content.append(f".. autodata:: cuda.tile.{dtype.name}")
        content.append("   :annotation:")
        content.append("")  # Empty line between types

    return '\n'.join(content)


def _generate_rst_table() -> str:
    short_names = {dtype: short_name.lower()
                   for short_name, dtype in _DTypePromotionImpl.__dict__.items()
                   if isinstance(dtype, DType)}

    # Get data type names based on table order
    public_dtypes = _get_all_public_numeric_dtypes_for_docs()
    size = len(public_dtypes)

    # Determine maximum width for all columns based on dtype names
    max_name_width = max(len(short_names[dtype]) for dtype in public_dtypes)
    max_name_width = max(max_name_width, len("ERR"))  # Account for "ERR" cells

    # Build all column widths with padding
    padding = 2  # space on each side
    col_width = max_name_width + padding  # Same width for all columns including row header

    lines = []

    # Generate separator line with same width for all columns
    sep_line = "+" + "+".join(["-" * col_width] * (size + 1)) + "+"
    header_sep_line = "+" + "+".join(["=" * col_width] * (size + 1)) + "+"

    # Table header
    lines.append(sep_line)
    header_cells = [f" {'':<{col_width-2}} "]
    for dtype in public_dtypes:
        col_name = short_names[dtype]
        header_cells.append(f" {col_name:<{col_width-2}} ")
    lines.append("|" + "|".join(header_cells) + "|")
    lines.append(header_sep_line)

    # Table rows
    public_indices = [_DTypePromotionImpl._order.index(dtype) for dtype in public_dtypes]
    for row_dtype, row_index in zip(public_dtypes, public_indices, strict=True):
        row_name = short_names[row_dtype]
        row_cells = [f" {row_name:<{col_width-2}} "]

        for col_dtype, col_index in zip(public_dtypes, public_indices, strict=True):
            cell = _DTypePromotionImpl._common_dtype_table[row_index][col_index]
            if cell is None:
                cell_str = "ERR"
            else:
                cell_str = short_names[cell]
            row_cells.append(f" {cell_str:<{col_width-2}} ")

        lines.append("|" + "|".join(row_cells) + "|")
        lines.append(sep_line)

    # Add a legend for the table
    lines.append("")  # Empty line after table
    lines.append("Legend:")
    lines.append("")  # Empty line before bullet points

    # Create bullet points for each enum and its corresponding dtype
    for dtype in public_dtypes:
        lines.append(f"* {short_names[dtype]}: ``{dtype.name}``")

    # Add an entry for the error case
    lines.append("* ERR: Implicit promotion between these types is not supported")

    return "\n".join(lines)


# ============== Pointer DType ===============


class PointerInfo:
    @stub(host=True)
    def __init__(self, dtype: DType):
        definition = _dtype_defs[dtype]
        if not isinstance(definition, _PointerDTypeDefinition):
            raise TypeError(f"'{dtype}' is not a pointer dtype")
        self._dtype = dtype
        self._definition = definition

    @property
    @stub(host=True)
    def opaque(self) -> bool:
        return self._definition.pointee_dtype is None

    @property
    @stub(host=True)
    def pointee_dtype(self) -> DType:
        if self._definition.pointee_dtype is None:
            raise ValueError("Opaque pointer has no pointee dtype")
        return self._definition.pointee_dtype

    @property
    @stub(host=True)
    def memory_space(self) -> MemorySpace:
        return self._definition.memory_space

    def __repr__(self):
        if self.opaque:
            type_str = "opaque"
        else:
            type_str = f"pointee_dtype={self.pointee_dtype}"

        if self.memory_space is MemorySpace.GENERIC:
            memspc_str = ""
        else:
            memspc_str = f", MemorySpace.{self.memory_space._name_}"

        return f"PointerInfo({type_str}{memspc_str})"

    def __eq__(self, other):
        return isinstance(other, PointerInfo) and self._dtype == other._dtype

    def __hash__(self):
        return hash(self._dtype)


@stub(host=True)
def is_pointer_dtype(dtype: DType) -> bool:
    return isinstance(_dtype_defs[dtype], _PointerDTypeDefinition)


@stub(host=True)
def pointer_dtype(pointee_dtype: DType,
                  memory_space: MemorySpace = MemorySpace.GENERIC) -> DType:
    assert pointee_dtype is not None
    return _get_pointer_dtype(pointee_dtype, memory_space)


@stub(host=True)
def opaque_pointer_dtype(memory_space: MemorySpace = MemorySpace.GENERIC) -> DType:
    return _get_pointer_dtype(None, memory_space)


@cache
def _get_pointer_dtype(pointee_dtype: DType | None, memory_space: MemorySpace) -> DType:
    match memory_space:
        case MemorySpace.SHARED | MemorySpace.TENSOR | MemorySpace.SHARED_CLUSTER:
            bitwidth = 32
        case _:
            bitwidth = 64

    params = []
    if pointee_dtype is None:
        name = "opaque_pointer"
    else:
        assert isinstance(pointee_dtype, DType)
        name = "pointer"
        params.append(str(pointee_dtype))

    if memory_space != MemorySpace.GENERIC:
        params.append(f"MemorySpace.{memory_space._name_}")

    if len(params) > 0:
        name += "[" + ", ".join(params) + "]"

    return _define_dtype(name,
                         _PointerDTypeDefinition(bitwidth=bitwidth,
                                                 pointee_dtype=pointee_dtype,
                                                 memory_space=memory_space))
