# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

from cuda.tile._memory_model import MemorySpace
from cuda.tile._datatype import (
    DType,
    NumericDTypeCategory,
    bfloat16,
    bool_,
    float8_e8m0fnu,
    float4_e2m1fn,
    float8_e4m3fn,
    float8_e5m2,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    tfloat32,
    uint8,
    uint16,
    uint32,
    uint64,
    is_float,
    is_restricted_float,
    is_unrestricted_float,
    is_arithmetic,
    is_boolean,
    is_integral,
    is_signed,
    is_numeric,
    unsigned_integral_dtypes,
    signed_integral_dtypes,
    get_signedness,
    default_int_type,
    integer_dtype,
    is_pointer_dtype,
    pointer_dtype,
    opaque_pointer_dtype,
    _define_dtype,
    _DTypeDefinition,
    PointerInfo,
    numeric_dtype_category,
)


# Lang-specific types.
float6_e2m3fn = _define_dtype(
    "float6_e2m3fn",
    _DTypeDefinition(bitwidth=6, numeric_category=NumericDTypeCategory.RestrictedFloat),
)
float6_e2m3fn.__doc__ = """A 6-bit floating-point numeric dtype with 1 sign bit,
2 exponent bits, and 3 mantissa bits."""

float6_e3m2fn = _define_dtype(
    "float6_e3m2fn",
    _DTypeDefinition(bitwidth=6, numeric_category=NumericDTypeCategory.RestrictedFloat),
)
float6_e3m2fn.__doc__ = """A 6-bit floating-point numeric dtype with 1 sign bit,
3 exponent bits, and 2 mantissa bits."""

arithmetic_float_dtypes = (
    float16,
    bfloat16,
    float32,
    float64,
)

non_arithmetic_float_dtypes = (
    tfloat32,
    float8_e4m3fn,
    float8_e5m2,
    float8_e8m0fnu,
    float4_e2m1fn,
    float6_e2m3fn,
    float6_e3m2fn,
)

mbarrier = _define_dtype('mbarrier', _DTypeDefinition(bitwidth=64))
cluster_launch_control_token = _define_dtype(
    "cluster_launch_control_token", _DTypeDefinition(bitwidth=128)
)


def to_torch_dtype(dtype: DType, /):
    if not isinstance(dtype, DType):
        raise TypeError("Expected a DType object")

    import torch

    dtype_map = {
        bool_: torch.bool,
        uint8: torch.uint8,
        uint16: torch.uint16,
        uint32: torch.uint32,
        uint64: torch.uint64,
        int8: torch.int8,
        int16: torch.int16,
        int32: torch.int32,
        int64: torch.int64,
        float16: torch.float16,
        bfloat16: torch.bfloat16,
        float32: torch.float32,
        float64: torch.float64,
    }
    if dtype in dtype_map:
        return dtype_map[dtype]

    optional_dtype_names = {
        float8_e4m3fn: "float8_e4m3fn",
        float8_e5m2: "float8_e5m2",
        float8_e8m0fnu: "float8_e8m0fnu",
    }
    if dtype in optional_dtype_names and hasattr(torch, optional_dtype_names[dtype]):
        return getattr(torch, optional_dtype_names[dtype])

    raise NotImplementedError(f"No torch dtype mapping for {dtype}")


TypeSpec: TypeAlias = DType


__all__ = [
    "is_float",
    "is_restricted_float",
    "is_unrestricted_float",
    "is_arithmetic",
    "is_boolean",
    "is_integral",
    "is_signed",
    "is_pointer_dtype",
    "is_numeric",
    "pointer_dtype",
    "opaque_pointer_dtype",
    "MemorySpace",
    "get_signedness",
    "integer_dtype",
    "unsigned_integral_dtypes",
    "signed_integral_dtypes",
    "arithmetic_float_dtypes",
    "non_arithmetic_float_dtypes",
    "bool_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "bfloat16",
    "tfloat32",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e8m0fnu",
    "float4_e2m1fn",
    "mbarrier",
    "cluster_launch_control_token",
    "DType",
    "to_torch_dtype",
    "default_int_type",
    "PointerInfo",
    "numeric_dtype_category"
]
