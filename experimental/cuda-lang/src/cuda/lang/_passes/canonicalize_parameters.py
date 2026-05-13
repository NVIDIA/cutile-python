# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable

from cuda.lang._ir.type import ArrayTy, MemorySpace, PointerTy, make_tile_ty, ArrayValue
from cuda.lang._ir.ops import MakeTensorView, ReinterpretPointer


def _rewrite_make_tensor_view(builder, op, array_parameter_names) -> Iterable:
    needs_rewrite = (
        isinstance(op, MakeTensorView)
        and op.result_var.name in array_parameter_names
        and not isinstance(op.result_var.get_type(), ArrayTy)
    )
    if not needs_rewrite:
        return [op]

    array_ty = op.result_var.get_type()
    new_array_ty = ArrayTy(
        dtype=array_ty.dtype,
        shape=array_ty.shape,
        strides=array_ty.strides,
        memory_space=MemorySpace.GENERIC,
    )
    base_ptr_ty = make_tile_ty(
        PointerTy(
            pointee=make_tile_ty(array_ty.dtype, ()),
            memory_space=MemorySpace.GENERIC,
        ),
        (),
    )
    base_ptr = op.base_ptr
    if base_ptr.get_type() != base_ptr_ty:
        reinterpret_result = builder.ir_ctx.make_temp(op.loc)
        reinterpret_result.set_type(base_ptr_ty)
        reinterpret = ReinterpretPointer(
            pointer=base_ptr,
            loc=op.loc,
            result_vars=(reinterpret_result,),
        )
        base_ptr = reinterpret_result
    op.base_ptr = base_ptr
    op.result_var.set_type(new_array_ty, force=True)
    op.result_var.set_aggregate(
        ArrayValue(
            base_ptr,
            op.result_var.get_aggregate().shape,
            op.result_var.get_aggregate().strides,
        )
    )
    return [reinterpret, op]


def canonicalize_parameters(params, builder) -> None:
    '''
    We use Tile's flatten_block_parameters and ast2hir lowering, which means
    we don't have control over the result type of make_tensor_view, but we would
    like to use our own array and pointer types to represent data that is not
    useful to Tile.
    Override the array construction to use our own types.
    Arrays created after the kernel/function arguments are only ever created
    by hir2ir lowerings that live in cuda.lang.
    '''
    array_param_names = {var.name for var in params.aggregate_vars}
    new_ops = []
    for op in builder.ops:
        rewritten_ops = _rewrite_make_tensor_view(
            builder,
            op,
            array_param_names,
        )
        new_ops.extend(rewritten_ops)

    builder._ops = new_ops


__all__ = ("canonicalize_parameters",)
