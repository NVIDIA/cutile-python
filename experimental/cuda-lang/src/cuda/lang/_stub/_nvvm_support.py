# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from cuda.lang import _datatype as datatype
from cuda.lang._execution import stub
from cuda.lang._ir.type import TileTy, is_vector_ty
from cuda.tile._ir.type import SymbolicTile


def satisfies_scalar_integral_constraint(value):
    return (
        type(value) is int
        or (
            isinstance(value, SymbolicTile)
            and value.ndim == 0
            and datatype.is_integral(value.dtype)
        )
    )


def satisfies_scalar_or_vector_constraint(value):
    return isinstance(value, SymbolicTile) and (
        value.ndim == 0 or is_vector_ty(TileTy(value.dtype, value.shape))
    )


def satisfies_vector_constraint(value):
    return isinstance(value, SymbolicTile) and is_vector_ty(
        TileTy(value.dtype, value.shape)
    )


@stub
def _raw_nvvm_intrinsic(
    intrinsic: str,
    result_dtypes: tuple[datatype.TypeSpec, ...] = (),
    operands: tuple[Any, ...] = (),
):
    '''
    Call an NVVM intrinsic directly.
    '''


__all__ = (
    "_raw_nvvm_intrinsic",
    "satisfies_vector_constraint",
    "satisfies_scalar_or_vector_constraint",
    "satisfies_scalar_integral_constraint",
)
