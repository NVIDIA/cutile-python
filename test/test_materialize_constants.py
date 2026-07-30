# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct

from cuda.tile._cext import CallingConvention
from cuda.tile._compile import compile_tile
from cuda.tile._ir.core_ops import TypedConst
from cuda.tile._ir.ops import MakeTensorView
from cuda.tile._ir.ops_utils import get_dtype
from cuda.tile.compilation import (
    ArrayConstraint,
    ConstantConstraint,
    KernelSignature,
    ListConstraint,
)


def _get_defined_op(body, name: str):
    definitions = {
        var.name: op
        for op in body.traverse()
        for var in op.result_vars
    }
    return definitions[name]


def test_unannotated_unit_stride_is_materialized():
    def kernel(x):
        ct.store(x, (0, 0), 0)

    constraint = ArrayConstraint(
        ct.float32,
        2,
        index_dtype=ct.int32,
        base_addr_divisible_by=1,
        stride_lower_bound_incl=0,
        stride_constant=(None, 1),
        stride_divisible_by=(1, 1),
        shape_divisible_by=(1, 1),
        alias_groups=[],
        may_alias_internally=False,
    )
    signature = KernelSignature((constraint,), CallingConvention.cutile_python_v2())
    [body] = compile_tile(
        kernel, [signature], return_final_ir=True, return_cubin=False
    ).final_ir

    [view] = [op for op in body.traverse() if isinstance(op, MakeTensorView)]

    # No ArrayAnnotation is present, so the array strides stays dynamic.
    assert view.result_var.get_type().strides == (None, None)
    inner_stride_defined_op = _get_defined_op(body, view.strides[1].name)
    assert isinstance(inner_stride_defined_op, TypedConst)
    assert inner_stride_defined_op.value == 1


def test_unannotated_unit_stride_is_materialized_in_nested_block():
    def kernel(xs, count: ct.Constant[int]):
        for i in range(count):
            item = xs[i]
            ct.store(item, (0,), 0)

    element_constraint = ArrayConstraint(
        ct.float32,
        1,
        index_dtype=ct.int32,
        base_addr_divisible_by=1,
        stride_lower_bound_incl=0,
        stride_constant=(1,),
        stride_divisible_by=(1,),
        shape_divisible_by=(1,),
        alias_groups=[],
        may_alias_internally=False,
    )
    list_constraint = ListConstraint(
        element_constraint, alias_groups=[], elements_may_alias=False
    )
    signature = KernelSignature(
        (list_constraint, ConstantConstraint(2)), CallingConvention.cutile_python_v2()
    )
    [body] = compile_tile(
        kernel, [signature], return_final_ir=True, return_cubin=False
    ).final_ir

    [view] = [op for op in body.traverse() if isinstance(op, MakeTensorView)]
    assert view not in body.operations
    assert view.result_var.get_type().strides == (None,)

    stride_defined_op = _get_defined_op(body, view.strides[0].name)
    assert isinstance(stride_defined_op, TypedConst)
    assert stride_defined_op.value == 1


def test_wrapping_integral_cast_is_materialized():
    def kernel(x):
        index = ct.astype(x.shape[0], ct.int8)
        ct.store(x, (index,), 0)

    signature = KernelSignature(
        (ArrayConstraint(
            ct.float32,
            1,
            index_dtype=ct.int32,
            base_addr_divisible_by=1,
            stride_lower_bound_incl=0,
            stride_constant=(1,),
            shape_constant=(260,),
            stride_divisible_by=(1,),
            shape_divisible_by=(1,),
            alias_groups=[],
            may_alias_internally=False,
        ),),
        CallingConvention.cutile_python_v2(),
    )
    [body] = compile_tile(
        kernel, [signature], return_final_ir=True, return_cubin=False
    ).final_ir

    assert any(
        isinstance(op, TypedConst)
        and op.value == 4
        and get_dtype(op.result_var.get_type()) == ct.int8
        for op in body.traverse()
    )
