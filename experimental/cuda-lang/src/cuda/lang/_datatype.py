# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from typing import TypeAlias, Union, Any

from cuda.lang._ir.type import (
    MemorySpace,
    OpaquePointerTy,
    PointerTy,
    TileTy,
    is_vector_ty,
    make_vector_ty,
)
from cuda.tile._ir.type import PointerTy as TilePointerTy, SymbolicTile
from cuda.tile._ir.typing_support import is_dtype, register_dtypes, to_dtype
from cuda.tile._datatype import (
    DType,
    ArithmeticDType,
    NumericDType,
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
    is_restricted_arithmetic,
    is_boolean,
    is_integral,
    is_signed,
    get_signedness,
    default_int_type,
)


def vector_ty(dtype: DType, length: int) -> TileTy:
    # Return a rank-1 TileTy(dtype, (length,)), creating it on first request and
    # registering the instance in cutile's dtype_registry so it can be
    # used as a compile-time constant inside kernels. Subsequent calls
    # with the same (dtype, length) return the already-registered instance.
    vt = make_vector_ty(dtype, length)
    if is_dtype(vt):
        return to_dtype(vt)
    register_dtypes({vt: vt}, usable_as_constructor=True)
    return vt


def to_torch_dtype(dtype: DType | TileTy):
    import torch

    if isinstance(dtype, TileTy):
        dtype = dtype.dtype

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


class OpaquePointerSpec(enum.Enum):
    GENERIC = MemorySpace.GENERIC.value
    GLOBAL = MemorySpace.GLOBAL.value
    SHARED = MemorySpace.SHARED.value
    CONSTANT = MemorySpace.CONSTANT.value
    LOCAL = MemorySpace.LOCAL.value
    SHARED_CLUSTER = MemorySpace.SHARED_CLUSTER.value
    TENSOR = MemorySpace.TENSOR.value

    ANY = -1


any_opaque_ptr = OpaquePointerSpec.ANY
opaque_generic_ptr = opaque_ptr = OpaquePointerSpec.GENERIC
opaque_global_ptr = OpaquePointerSpec.GLOBAL
opaque_shared_ptr = OpaquePointerSpec.SHARED
opaque_constant_ptr = OpaquePointerSpec.CONSTANT
opaque_local_ptr = OpaquePointerSpec.LOCAL
opaque_shared_cluster_ptr = OpaquePointerSpec.SHARED_CLUSTER
opaque_tensor_ptr = OpaquePointerSpec.TENSOR


TypeSpec: TypeAlias = Union[OpaquePointerSpec | DType | TileTy]


def is_literal_or_exact_dtype(value: Any, dtype: DType):
    match value:
        case int() if is_integral(dtype):
            return True
        case bool() if is_boolean(dtype):
            return True
        case float() if is_float(dtype):
            return True
        case SymbolicTile() if is_vector_ty(dtype):
            return value.dtype == dtype.dtype and value.shape == dtype.shape
        case SymbolicTile() if value.dtype == dtype:
            return True
        case _:
            return False


def is_any_pointer(value):
    return (
        isinstance(value, SymbolicTile)
        and value.ndim == 0
        and isinstance(value.dtype, (PointerTy, OpaquePointerTy, TilePointerTy))
    )


def satisfies_pointer_constraint(value, constraint: OpaquePointerSpec):
    if not is_any_pointer(value):
        return False

    if constraint == any_opaque_ptr:
        return True

    pointer_ty = value.dtype
    if isinstance(pointer_ty, TilePointerTy):
        if constraint == opaque_ptr:
            return True
        return pointer_ty.memory_space.value == constraint.value

    return pointer_ty.memory_space.value == constraint.value


__all__ = [
    "is_float",
    "is_restricted_float",
    "is_unrestricted_float",
    "is_arithmetic",
    "is_restricted_arithmetic",
    "is_boolean",
    "is_integral",
    "is_signed",
    "is_any_pointer",
    "satisfies_pointer_constraint",
    "is_literal_or_exact_dtype",
    "get_signedness",
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
    "bfloat16",
    "tfloat32",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e8m0fnu",
    "float4_e2m1fn",
    "DType",
    "NumericDType",
    "ArithmeticDType",
    "vector_ty",
    "to_torch_dtype",
    "default_int_type",
    "any_opaque_ptr",
    "opaque_ptr",
    "opaque_generic_ptr",
    "opaque_global_ptr",
    "opaque_shared_ptr",
    "opaque_constant_ptr",
    "opaque_local_ptr",
    "opaque_shared_cluster_ptr",
    "opaque_tensor_ptr",
    "MemorySpace",
]
