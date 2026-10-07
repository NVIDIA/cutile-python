# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple
from enum import IntEnum

from cuda.tile._exception import TileTypeError
from cuda.tile._execution import stub
from cuda.tile._memory_model import MemorySpace
import cuda.tile._bytecode as bc
from cuda.tile._cext import (
    DType, is_numeric, is_boolean, is_integral, is_signed, is_float, is_unrestricted_float,
    is_restricted_float, is_arithmetic, _is_pointer_dtype, _is_foreign_pointer_dtype,
    _pointer_pointee_dtype, _pointer_memory_space, _foreign_pointer_pointee_dtype,
    _get_pointer_dtype, _get_foreign_pointer_dtype,
    integer_dtype_min, integer_dtype_max,
    bool_, uint8, uint16, uint32, uint64, int8, int16, int32, int64,
    float16, bfloat16, float32, float64, tfloat32, float8_e4m3fn, float8_e5m2, float8_e8m0fnu,
    float8_e5m3fnu, float4_e2m1fn
)


__all__ = ["bool_", "uint8", "uint16", "uint32", "uint64",
           "int8", "int16", "int32", "int64",
           "float16", "float32", "float64",
           "bfloat16", "tfloat32", "float8_e4m3fn", "float8_e5m2",
           "float8_e8m0fnu", "float8_e5m3fnu", "float4_e2m1fn", "DType",
           "foreign_pointer_dtype", "is_numeric", "is_boolean", "is_integral", "is_signed",
           "is_arithmetic", "is_float", "is_unrestricted_float", "is_restricted_float"]


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


class IntegerInfo:
    """
    Machine information for integer data types, similar to numpy.iinfo.
    """
    @stub(host=True)
    def __init__(self, dtype: DType):
        if not is_integral(dtype):
            raise TypeError(f"'{dtype}' is not an integer dtype")
        self._dtype = dtype

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def bits(self) -> int:
        return self._dtype.bitwidth

    @property
    def min(self) -> int:
        return integer_dtype_min(self._dtype)

    @property
    def max(self) -> int:
        return integer_dtype_max(self._dtype)

    def __eq__(self, other):
        return isinstance(other, IntegerInfo) and self._dtype == other._dtype

    def __hash__(self):
        return hash(self._dtype)


default_int_type = int32
default_float_type = float32


#: Unsigned integral |dtypes|. These |dtypes| are arithmetic.
unsigned_integral_dtypes = [uint64, uint32, uint16, uint8]

#: Signed integral |dtypes|. These |dtypes| are arithmetic.
signed_integral_dtypes = [int64, int32, int16, int8]


def numeric_dtype_category(t: DType) -> NumericDTypeCategory:
    if is_boolean(t):
        return NumericDTypeCategory.Boolean
    elif is_integral(t):
        return NumericDTypeCategory.Integral
    elif is_unrestricted_float(t):
        return NumericDTypeCategory.Float
    elif is_restricted_float(t):
        return NumericDTypeCategory.RestrictedFloat
    else:
        assert not is_numeric(t)
        raise ValueError(f"{t} is not a numeric dtype")


_dtype_to_simple_bytecode_type = {
    bool_: bc.SimpleType.I1,
    uint8: bc.SimpleType.I8,
    uint16: bc.SimpleType.I16,
    uint32: bc.SimpleType.I32,
    uint64: bc.SimpleType.I64,
    int8: bc.SimpleType.I8,
    int16: bc.SimpleType.I16,
    int32: bc.SimpleType.I32,
    int64: bc.SimpleType.I64,
    float16: bc.SimpleType.F16,
    bfloat16: bc.SimpleType.BF16,
    float32: bc.SimpleType.F32,
    tfloat32: bc.SimpleType.TF32,
    float64: bc.SimpleType.F64,
    float8_e4m3fn: bc.SimpleType.F8E4M3FN,
    float8_e5m2: bc.SimpleType.F8E5M2,
    float8_e8m0fnu: bc.SimpleType.F8E8M0FNU,
    float4_e2m1fn: bc.SimpleType.F4E2M1FN,
    float8_e5m3fnu: bc.SimpleType.FNV8E5M3FNU,
}


def dtype_simple_bytecode_type(t: DType) -> bc.SimpleType:
    return _dtype_to_simple_bytecode_type[t]


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
    f8e5m3fnu = float8_e5m3fnu
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
                    float8_e4m3fn:  (float32, (16,)),
                    float8_e5m3fnu: (float32, (16, 32))},
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
    """Information encoded in a pointer dtype."""

    @stub(host=True)
    def __init__(self, dtype: DType):
        if not is_pointer_dtype(dtype):
            raise TypeError(f"'{dtype}' is not a pointer dtype")
        self._dtype = dtype

    @property
    @stub(host=True)
    def opaque(self) -> bool:
        """Whether the pointer dtype is opaque."""
        return _pointer_pointee_dtype(self._dtype) is None

    @property
    @stub(host=True)
    def pointee_dtype(self) -> DType:
        """Data type pointed to by this pointer dtype."""
        ret = _pointer_pointee_dtype(self._dtype)
        if ret is None:
            raise ValueError("Opaque pointer has no pointee dtype")
        return ret

    @property
    @stub(host=True)
    def memory_space(self) -> MemorySpace:
        """CUDA memory space encoded in this pointer dtype."""
        return _pointer_memory_space(self._dtype)

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


@stub(host=True, compiled_host=True)
def is_pointer_dtype(dtype: DType) -> bool:
    """Return whether ``dtype`` is a pointer dtype."""
    return _is_pointer_dtype(dtype)


@stub(host=True, compiled_host=True)
def pointer_dtype(pointee_dtype: DType,
                  memory_space: MemorySpace = MemorySpace.GENERIC) -> DType:
    """Return the dtype for a pointer to ``pointee_dtype`` in ``memory_space``.

    Args:
        pointee_dtype (DType): Type of the data pointed to by a pointer
            described by this dtype.
        memory_space (MemorySpace): Memory space where the pointer described by
            this dtype resides.
    """
    assert pointee_dtype is not None
    return _get_pointer_dtype(pointee_dtype, memory_space)


@stub(host=True, compiled_host=True)
def opaque_pointer_dtype(memory_space: MemorySpace = MemorySpace.GENERIC) -> DType:
    """Return the dtype for an opaque pointer in ``memory_space``.

    Args:
        memory_space (MemorySpace): Memory space where the pointer described by
            this dtype resides.
    """
    return _get_pointer_dtype(None, memory_space)


# ============== Foreign Pointer DType ===============

def is_foreign_pointer_dtype(dtype: DType) -> bool:
    return _is_foreign_pointer_dtype(dtype)


def foreign_pointer_pointee_dtype(dtype: DType) -> DType:
    """Return the pointee dtype encoded in a foreign pointer dtype."""
    return _foreign_pointer_pointee_dtype(dtype)


@stub(host=True, static_eval_ok=True)
def foreign_pointer_dtype(pointee_dtype: DType) -> DType:
    if not isinstance(pointee_dtype, DType):
        raise TypeError("pointee_dtype must be a cuda.tile dtype")
    if is_foreign_pointer_dtype(pointee_dtype):
        raise TypeError("nested foreign pointer dtypes are not supported")
    return _get_foreign_pointer_dtype(pointee_dtype)
