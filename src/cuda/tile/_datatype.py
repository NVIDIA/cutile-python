# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Any, Callable, Tuple
from enum import IntEnum

from cuda.tile._exception import TileTypeError
from cuda.tile._ir.type import Type, TupleTy, SizeTy
from cuda.tile._execution import function
import cuda.tile._bytecode as bc


__all__ = ["bool_", "uint8", "uint16", "uint32", "uint64",
           "int8", "int16", "int32", "int64",
           "float16", "float32", "float64",
           "bfloat16", "tfloat32", "float8_e4m3fn", "float8_e5m2",
           "DType", "NumericDType", "ArithmeticDType",
           "NumericDTypeCategories"]


class DType(Type):
    """A *data type* (or *dtype*) describes the type of the objects of an |array|, |tile|, or
    operation.

    |Dtypes| determine how values are stored in memory and how operations on those values are
    performed.
    |Dtypes| are immutable.

    |Dtypes| can be used in |host code| and |tile code|.
    They can be |kernel| parameters.
    """

    def __init__(self, name: str, bitwidth: int, py_type: Callable[[Any], Any],
                 bytecode_type: bc.SimpleType):
        self._name = name
        self._bitwidth = bitwidth
        self._py_type = py_type
        self._bytecode_type = bytecode_type

    @property
    @function(host=True, tile=False)
    def bitwidth(self):
        """The number of bits in an element of the |data type|."""
        return self._bitwidth

    @property
    @function(host=True, tile=False)
    def name(self):
        """The name of the |data type|."""
        return self._name

    @property
    @function(host=True, tile=False)
    def __name__(self) -> str:
        return self._name

    @function(host=True, tile=False)
    def __str__(self):
        return self._name

    @function
    def __call__(self, value):
        """Construct a Scalar of this |data type| from a value."""

    @function(host=True, tile=True)
    def __eq__(self, other: Type):
        return isinstance(other, DType) and self._name == other._name

    @function(host=True, tile=False)
    def __hash__(self):
        return hash(self._name)


class NumericDType(DType):
    """A *numeric* |dtype| represents numbers."""


class ArithmeticDType(NumericDType):
    """An *arithmetic* |dtype| is a |numeric dtype| that supports general arithmetic operations
    such as addition, subtraction, multiplication, and division."""


bool_ = ArithmeticDType('bool_', 8, bool, bc.SimpleType.I1)
bool_.__doc__ = """A 8-bit |arithmetic dtype| (``True`` or ``False``)."""

uint8 = ArithmeticDType('uint8', 8, int, bc.SimpleType.I8)
uint8.__doc__ = """A 8-bit unsigned integer |arithmetic dtype| whose values exist on the interval \
[0, +256]."""

uint16 = ArithmeticDType('uint16', 16, int, bc.SimpleType.I16)
uint16.__doc__ = """A 16-bit unsigned integer |arithmetic dtype| whose values exist on the \
interval [0, +65,536]."""

uint32 = ArithmeticDType('uint32', 32, int, bc.SimpleType.I32)
uint32.__doc__ = """A 32-bit unsigned integer |arithmetic dtype| whose values exist on the \
interval [0, +4,294,967,295]."""

uint64 = ArithmeticDType('uint64', 64, int, bc.SimpleType.I64)
uint64.__doc__ = """A 64-bit unsigned integer |arithmetic dtype| whose values exist on the \
interval [0, +18,446,744,073,709,551,615]."""

int8 = ArithmeticDType('int8', 8, int, bc.SimpleType.I8)
int8.__doc__ = """A 8-bit signed integer |arithmetic dtype| whose values exist on the interval \
[−128, +127]."""

int16 = ArithmeticDType('int16', 16, int, bc.SimpleType.I16)
int16.__doc__ = """A 16-bit signed integer |arithmetic dtype| whose values exist on the interval \
[−32,768, +32,767]."""

int32 = ArithmeticDType('int32', 32, int, bc.SimpleType.I32)
int32.__doc__ = """A 32-bit signed integer |arithmetic dtype| whose values exist on the interval \
[−2,147,483,648, +2,147,483,647]."""

int64 = ArithmeticDType('int64', 64, int, bc.SimpleType.I64)
int64.__doc__ = """A 64-bit signed integer |arithmetic dtype| whose values exist on the interval \
[−9,223,372,036,854,775,808, +9,223,372,036,854,775,807]."""

float16 = ArithmeticDType('float16', 16, float, bc.SimpleType.F16)
float16.__doc__ = """A IEEE 754 half-precision (16-bit) binary floating-point |arithmetic dtype| \
(see |IEEE 754-2019|)."""

float32 = ArithmeticDType('float32', 32, float, bc.SimpleType.F32)
float32.__doc__ = """A IEEE 754 single-precision (32-bit) binary floating-point |arithmetic dtype| \
(see |IEEE 754-2019|)."""

float64 = ArithmeticDType('float64', 64, float, bc.SimpleType.F64)
float64.__doc__ = """A IEEE 754 double-precision (64-bit) binary floating-point |arithmetic dtype| \
(see |IEEE 754-2019|)."""

bfloat16 = ArithmeticDType('bfloat16', 16, float, bc.SimpleType.BF16)
bfloat16.__doc__ = """A 16-bit floating-point |arithmetic dtype| with 1 sign bit, 8 exponent bits, \
and 7 mantissa bits."""

tfloat32 = NumericDType("tfloat32", 32, float, bc.SimpleType.TF32)
tfloat32.__doc__ = """A 32-bit tensor floating-point |arithmetic dtype| with 1 sign \
bit, 8 exponent bits, and 10 mantissa bits (19-bit representation stored in 32-bit container)."""

float8_e4m3fn = NumericDType("float8_e4m3fn", 8, float, bc.SimpleType.F8E4M3FN)
float8_e4m3fn.__doc__ = """A 8-bit floating-point |arithmetic dtype| with 1 sign bit, \
4 exponent bits, and 3 mantissa bits."""

float8_e5m2 = NumericDType("float8_e5m2", 8, float, bc.SimpleType.F8E5M2)
float8_e5m2.__doc__ = """A 8-bit floating-point |arithmetic dtype| with 1 sign bit, \
5 exponent bits, and 2 mantissa bits."""


class DTypeEnum(IntEnum):
    B1 = 0
    U8 = 1
    U16 = 2
    U32 = 3
    U64 = 4
    I8 = 5
    I16 = 6
    I32 = 7
    I64 = 8
    F16 = 9
    F32 = 10
    F64 = 11
    BF = 12
    TF32 = 13
    F8E4M3FN = 14
    F8E5M2 = 15


dtype_to_enum = {
    bool_: DTypeEnum.B1,
    uint8: DTypeEnum.U8,
    uint16: DTypeEnum.U16,
    uint32: DTypeEnum.U32,
    uint64: DTypeEnum.U64,
    int8: DTypeEnum.I8,
    int16: DTypeEnum.I16,
    int32: DTypeEnum.I32,
    int64: DTypeEnum.I64,
    float16: DTypeEnum.F16,
    float32: DTypeEnum.F32,
    float64: DTypeEnum.F64,
    bfloat16: DTypeEnum.BF,
    tfloat32: DTypeEnum.TF32,
    float8_e4m3fn: DTypeEnum.F8E4M3FN,
    float8_e5m2: DTypeEnum.F8E5M2,
}
_enum_to_dtype = dict((i, t) for t, i in dtype_to_enum.items())


default_int_type = int32
default_float_type = float32


class NumericDTypeCategory(IntEnum):
    Boolean = 0
    Integral = 1
    Float = 2
    RestrictedFloat = 3


class NumericDTypeCategories:
    """|Numeric dtypes| are grouped into categories, which dictate what
    promotions and conversions are allowed.
    """
    Boolean = [bool_]
    Integral = [uint8, uint16, uint32, uint64, int8, int16, int32, int64]
    Float = [float16, float32, float64, bfloat16]
    RestrictedFloat = [tfloat32, float8_e4m3fn, float8_e5m2]

    @classmethod
    def get_category(cls, t: DType) -> NumericDTypeCategory:
        """Return the category for a given |dtype|."""
        if t in cls.Boolean:
            return NumericDTypeCategory.Boolean
        elif t in cls.Integral:
            return NumericDTypeCategory.Integral
        elif t in cls.Float:
            return NumericDTypeCategory.Float
        elif t in cls.RestrictedFloat:
            return NumericDTypeCategory.RestrictedFloat
        raise RuntimeError(f'Unknown dtype category for {t}')


#: All numeric |dtypes|.
numeric_dtypes = (NumericDTypeCategories.Boolean +
                  NumericDTypeCategories.Integral +
                  NumericDTypeCategories.Float +
                  NumericDTypeCategories.RestrictedFloat)

#: All arithmetic |dtypes|.
arithmetic_dtypes = (NumericDTypeCategories.Boolean +
                     NumericDTypeCategories.Integral +
                     NumericDTypeCategories.Float)

#: Unsigned integral |dtypes|. These |dtypes| are arithmetic.
unsigned_integral_dtypes = [uint64, uint32, uint16, uint8]

#: Signed integral |dtypes|. These |dtypes| are arithmetic.
signed_integral_dtypes = [int64, int32, int16, int8]


def is_boolean(t: DType) -> bool:
    return t in NumericDTypeCategories.Boolean


def is_integral(t: DType) -> bool:
    return t in NumericDTypeCategories.Integral


def is_signed(t: DType) -> bool:
    """Returns True if the |dtype| is a signed numeric type, such as a signed integer or
    a floating-point type."""
    return t in numeric_dtypes and t not in unsigned_integral_dtypes and t != bool_


_signedness = (bc.Signedness.Unsigned, bc.Signedness.Signed)


def get_signedness(t: DType) -> bc.Signedness:
    return _signedness[is_signed(t)]


def is_float(t: DType) -> bool:
    return t in NumericDTypeCategories.Float


def is_restricted_float(t: DType) -> bool:
    return t in NumericDTypeCategories.RestrictedFloat


def is_arithmetic(t: DType) -> bool:
    """Returns True if the |dtype| supports general arithmetic operations such as
    addition, subtraction, multiplication, and division."""
    return t in arithmetic_dtypes


def is_restricted_arithmetic(t: DType) -> bool:
    return t in NumericDTypeCategories.RestrictedFloat


def broadcast_shapes(s1: TupleTy, s2: TupleTy) -> TupleTy:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    s1 = [SizeTy(1)] * (len(s2) - len(s1)) + list(s1)

    result_shape = []
    for d1, d2 in zip(s1, s2):
        if d1 != d2:
            if d1.value == 1:
                result_shape.append(d2)
            elif d2.value == 1:
                result_shape.append(d1)
            else:
                raise TypeError(f"Broadcast shapes mismatch: {s1}, {s2}")
        else:
            result_shape.append(d1)
    return TupleTy(tuple(result_shape))


# ============= Arithmetic Promotion ==============


class _DTypePromotionImpl:

    # shorter alias to make the table
    b1 = DTypeEnum.B1
    u8 = DTypeEnum.U8
    u16 = DTypeEnum.U16
    u32 = DTypeEnum.U32
    u64 = DTypeEnum.U64
    i8 = DTypeEnum.I8
    i16 = DTypeEnum.I16
    i32 = DTypeEnum.I32
    i64 = DTypeEnum.I64
    f16 = DTypeEnum.F16
    f32 = DTypeEnum.F32
    f64 = DTypeEnum.F64
    bf = DTypeEnum.BF
    tf = DTypeEnum.TF32
    f8e4m3fn = DTypeEnum.F8E4M3FN
    f8e5m2 = DTypeEnum.F8E5M2
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
    _common_dtype_table = [
     # b1, u8, u16, u32, u64, i8,  i16, i32, i64, f16, f32, f64, bf,  tf, f8e4m3fn, f8e5m2
     [b1,  u8,  u16, u32, u64, i8,  i16, i32, i64, f16, f32, f64, bf,  na,  na,       na],    # b1
     [u8,  u8,  u16, u32, u64, na,  na,  na,  na,  f16, f32, f64, bf,  na,  na,       na],    # u8
     [u16, u16, u16, u32, u32, na,  na,  na,  na,  f16, f32, f64, bf,  na,  na,       na],    # u16
     [u32, u32, u32, u32, u64, na,  na,  na,  na,  f16, f32, f64, bf,  na,  na,       na],    # u32
     [u64, u64, u64, u64, u64, na,  na,  na,  na,  f16, f32, f64, bf,  na,  na,       na],    # u64
     [i8,  na,  na,  na,  na,  i8,  i16, i32, i64, f16, f32, f64, bf,  na,  na,       na],    # i8
     [i16, na,  na,  na,  na,  i16, i16, i32, i64, f16, f32, f64, bf,  na,  na,       na],    # i16
     [i32, na,  na,  na,  na,  i32, i32, i32, i64, f16, f32, f64, bf,  na,  na,       na],    # i32
     [i64, na,  na,  na,  na,  i64, i64, i64, i64, f16, f32, f64, bf,  na,  na,       na],    # i64
     [f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f32, f64, na,  na,  na,       na],    # f16
     [f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f64, f32, na,  na,       na],    # f32
     [f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, na,  na,       na],    # f64
     [bf,  bf,  bf,  bf,  bf,  bf,  bf,  bf,  bf,  na,  f32, f64, bf,  na,  na,       na],    # bf
     [na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  tf,  na,       na],    # tf
     [na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  f8e4m3fn, na],    # f8e4m3fn # noqa
     [na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,  na,   f8e5m2],    # f8e5m2  # noqa
    ]

    @classmethod
    def promote_dtypes(cls, t1: DType, t2: DType, force_float: bool = False) -> DType:
        if is_restricted_arithmetic(t1) or is_restricted_arithmetic(t2):
            if t1 == t2:
                return t1
            raise TileTypeError(
                f"Implicit promotion of {t1} and {t2} is not supported as it involves restricted "
                f"arithmetic dtypes in an unsupported way. Please perform an explicit cast instead."
            )
        idx1, idx2 = dtype_to_enum[t1], dtype_to_enum[t2]
        if idx1 >= len(cls._common_dtype_table) or idx2 >= len(cls._common_dtype_table[idx1]):
            raise IndexError(f"Invalid dtypes in common dtype table: {t1}, {t2}")
        ret = cls._common_dtype_table[idx1][idx2]
        if ret is None:
            msg = (f'Implicit promotion of {t1} and {t2} is not supported. '
                   'Please perform an explict cast instead.')
            raise TileTypeError(msg)
        res_type = _enum_to_dtype[ret]
        return res_type if not force_float or is_float(res_type) else default_float_type


def get_int_min_max(t: DType) -> Tuple[int, int]:
    assert is_integral(t)
    if is_signed(t):
        n = 1 << (t.bitwidth - 1)
        return -n, n - 1
    else:
        return 0, (1 << t.bitwidth) - 1


# TODO: bitwidth now is presenting the storage size, bool is 8-bit.
# When we compare bool and subbyte dtype like int4 later, we need to change this.
def can_autocast_dtypes(from_d: DType, to_d: DType) -> bool:
    if from_d == to_d:
        return True

    if is_restricted_arithmetic(from_d) or is_restricted_arithmetic(to_d):
        return False

    k1, k2 = NumericDTypeCategories.get_category(from_d), NumericDTypeCategories.get_category(to_d)
    if k1 == k2:
        # same category allowed if signedness is the same and not narrowing, lose range or precision
        return (is_signed(from_d) == is_signed(to_d)) and (from_d.bitwidth < to_d.bitwidth)
    else:
        return k1 < k2


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


# =============== Documentation Generator ================

def _generate_rst_dtype_promotion_table() -> str:
    """Generate an RST table representation of the dtype promotion rules."""
    return _generate_rst_table(_DTypePromotionImpl._common_dtype_table)


def _generate_rst_numeric_dtypes() -> str:
    """Generate RST documentation for numeric datatypes."""
    content = []

    for dtype in numeric_dtypes:
        content.append(f".. autodata:: cuda.tile.{dtype.name}")
        content.append("   :annotation:")
        content.append("")  # Empty line between types

    return '\n'.join(content)


def _generate_rst_table(common_dtype_table) -> str:
    # Get table dimensions
    rows = len(common_dtype_table)
    if rows == 0:
        return "Empty promotion table"
    cols = len(common_dtype_table[0])

    # Get data type names based on table order
    dtype_names = []
    for i, enum_val in enumerate(DTypeEnum):
        if i < rows:
            dtype_names.append(enum_val.name.lower())

    # Determine maximum width for all columns based on dtype names
    max_name_width = max(len(name) for name in dtype_names) if dtype_names else 1
    max_name_width = max(max_name_width, len("ERR"))  # Account for "ERR" cells

    # Build all column widths with padding
    padding = 2  # space on each side
    col_width = max_name_width + padding  # Same width for all columns including row header

    lines = []

    # Generate separator line with same width for all columns
    sep_line = "+" + "+".join(["-" * col_width] * (cols + 1)) + "+"
    header_sep_line = "+" + "+".join(["=" * col_width] * (cols + 1)) + "+"

    # Table header
    lines.append(sep_line)
    header_cells = [f" {'':<{col_width-2}} "]
    for i in range(cols):
        col_name = dtype_names[i] if i < len(dtype_names) else "?"
        header_cells.append(f" {col_name:<{col_width-2}} ")
    lines.append("|" + "|".join(header_cells) + "|")
    lines.append(header_sep_line)

    # Table rows
    for i, row in enumerate(common_dtype_table):
        row_name = dtype_names[i] if i < len(dtype_names) else "?"
        row_cells = [f" {row_name:<{col_width-2}} "]

        for j, cell in enumerate(row):
            if j < cols:
                if cell is None:
                    cell_str = "ERR"
                elif isinstance(cell, DTypeEnum):
                    cell_str = cell.name.lower()
                else:
                    cell_str = str(cell)
                row_cells.append(f" {cell_str:<{col_width-2}} ")

        lines.append("|" + "|".join(row_cells) + "|")
        lines.append(sep_line)

    # Add a legend for the table
    lines.append("")  # Empty line after table
    lines.append("Legend:")
    lines.append("")  # Empty line before bullet points

    # Create bullet points for each enum and its corresponding dtype
    for enum_val, dtype_obj in _enum_to_dtype.items():
        enum_name = enum_val.name.lower()
        dtype_name = dtype_obj.name
        lines.append(f"* {enum_name}: ``{dtype_name}``")

    # Add an entry for the error case
    lines.append("* ERR: Implicit promotion between these types is not supported")

    return "\n".join(lines)
