# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Callable
from cuda.tile._ir.ir import Block
from cuda.tile._ir.core_ops import TypedConst
from cuda.tile._ir.ops import TileAtomicRMW, TileAtomicRedView, AtomicRMWMode
from cuda.tile._ir.type import TileTy, Type
from cuda.tile._datatype import (
    DType, float4_e2m1fn, float8_e4m3fn, float8_e5m2, float8_e8m0fnu, float8_e5m3fnu,
    bfloat16, is_pointer_dtype, PointerInfo
)
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._exception import TileUnsupportedFeatureError, TileValueError

# Minimum SM architecture number (e.g. 90 for sm_90) per dtype.
# Technically sm_89 (Ada Lovelace) supports FP8, but tileiras doesn't have support for it yet.
_DTYPE_MIN_SM: dict[DType, int] = {
    float8_e4m3fn: 90,
    float8_e5m2: 90,
    float8_e8m0fnu: 100,
    float4_e2m1fn: 100,
    float8_e5m3fnu: 107,
}

# Minimum bytecode version per dtype.
_DTYPE_MIN_BC_VERSION: dict[DType, BytecodeVersion] = {
    float8_e8m0fnu: BytecodeVersion.V_13_2,
    float4_e2m1fn: BytecodeVersion.V_13_3,
    float8_e5m3fnu: BytecodeVersion.V_13_4,
}

# dtype: (predicate, error message).
_DTYPE_INVALID_VALUE: dict[DType, tuple[Callable[[float | int], bool], str]] = {
    float8_e8m0fnu: (
        lambda v: math.copysign(1.0, v) < 0,
        f"negative values cannot be represented in {float8_e8m0fnu}",
    ),
    float4_e2m1fn: (
        math.isnan,
        f"NaN cannot be represented in {float4_e2m1fn}",
    ),
    float8_e5m3fnu: (
        lambda v: math.copysign(1.0, v) < 0,
        f"negative values cannot be represented in {float8_e5m3fnu}",
    )
}


def _extract_dtypes(ty: Type | None) -> set[DType]:
    if ty is None:
        return set()

    if ty.is_aggregate():
        result = set()
        for item_ty in ty.aggregate_item_types():
            result |= _extract_dtypes(item_ty)
        return result

    if isinstance(ty, TileTy):
        dtype = ty.dtype
        while is_pointer_dtype(dtype):
            info = PointerInfo(dtype)
            if info.opaque:
                return set()
            dtype = info.pointee_dtype
        return {dtype}

    return set()


def _check_const_value(op: TypedConst):

    def get_values(v):
        if not isinstance(v, tuple):
            return (v,)
        return sum((get_values(c) for c in v), start=())

    dtype = next(iter(_extract_dtypes(op.result_vars[0].try_get_type())), None)
    entry = _DTYPE_INVALID_VALUE.get(dtype)
    if entry is not None:
        predicate, msg = entry
        if any(predicate(v) for v in get_values(op.value)):
            raise TileValueError(msg, loc=op.loc)


def _check_atomic_rmw_dtype(op: TileAtomicRedView | TileAtomicRMW,
                            sm_arch: str | None,
                            sm_number: int | None,
                            version: BytecodeVersion):
    dtypes = (_extract_dtypes(op.view.try_get_type())
              if isinstance(op, TileAtomicRedView) else
              _extract_dtypes(op.result_vars[0].try_get_type()))
    if not (op.mode == AtomicRMWMode.ADD_FLOAT and bfloat16 in dtypes):
        return

    if sm_number is not None and sm_number < 90:
        raise TileUnsupportedFeatureError(
            f"{bfloat16} is not supported by atomic add on {sm_arch}",
            loc=op.loc
        )

    min_version = BytecodeVersion.V_13_3
    if version < min_version:
        raise TileUnsupportedFeatureError(
            f"{bfloat16} on atomic add requires tileiras"
            f" {min_version.as_string()} or later."
            f" Current version is {version.as_string()}.",
            loc=op.loc
        )


def _check_dtype(dtype: DType, sm_arch: str | None, sm_number: int | None,
                 version: BytecodeVersion, loc):
    min_sm = _DTYPE_MIN_SM.get(dtype)
    if sm_number is not None and min_sm is not None and sm_number < min_sm:
        raise TileUnsupportedFeatureError(
            f"{dtype} is not supported on {sm_arch}",
            loc=loc,
        )

    min_version = _DTYPE_MIN_BC_VERSION.get(dtype)
    if min_version is not None and version < min_version:
        raise TileUnsupportedFeatureError(
            f"{dtype} requires tileiras"
            f" {min_version.as_string()} or later."
            f" Current version is {version.as_string()}.",
            loc=loc,
        )


def check_dtype_support(root_block: Block, sm_arch: str | None,
                        version: BytecodeVersion) -> None:
    # Skip arch check by setting sm_number to None when sm_arch is not provided
    sm_number = int(sm_arch.removeprefix("sm_")) if sm_arch is not None else None
    for op in root_block.traverse():
        if isinstance(op, TypedConst):
            _check_const_value(op)

        if isinstance(op, (TileAtomicRedView, TileAtomicRMW)):
            _check_atomic_rmw_dtype(op, sm_arch, sm_number, version)

        all_dtypes = set().union(*(_extract_dtypes(v.try_get_type()) for v in op.all_inputs()))
        for dtype in all_dtypes:
            _check_dtype(dtype, sm_arch, sm_number, version, op.loc)
