# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from cuda.tile._ir.ir import Block, Operation, Var
from cuda.tile._ir.ops import AssumeDivBy, MakeTensorView, LoadPointer, StorePointer
from cuda.tile._passes.dataflow_analysis import DataflowResult, DataPredicate

_OPS_NEED_ASSUME = (MakeTensorView, LoadPointer, StorePointer)


def add_divby_pass(root_block: Block, df_result: DataflowResult):
    candidates = set()
    _scan_block(root_block, candidates)
    var_map = {}
    _rewrite_block(root_block, df_result, var_map, candidates)


def _scan_block(block: Block, candidates: set[str]):
    """Add var that needs divby to candidates"""
    for op in block:
        if isinstance(op, _OPS_NEED_ASSUME):
            for var in op.all_inputs():
                candidates.add(var.name)
        for b in op.nested_blocks:
            _scan_block(b, candidates)


def _rewrite_block(block: Block,
                   df_result: DataflowResult,
                   var_map: dict[str, Var],
                   candidates: set[str]):
    new_ops = []
    for param in block.params:
        _add_assume_divby(param, df_result, new_ops, var_map)

    for op in block:
        to_assume = tuple(var for var in op.result_vars if var.name in candidates)
        new_ops.append(_remap_operands(op, var_map))
        for var in to_assume:
            _add_assume_divby(var, df_result, new_ops, var_map)
        for b in op.nested_blocks:
            _rewrite_block(b, df_result, var_map, candidates)

    block[:] = new_ops


def _remap_operands(op: Operation, var_map: dict[str, Var]):
    new_fields = {}
    for field_name, var in op.operands.items():
        if isinstance(var, tuple):
            new_var = tuple(var_map.get(vi.name, vi) for vi in var)
            new_fields[field_name] = new_var
        elif isinstance(var, Var):
            new_var = var_map.get(var.name, var)
            new_fields[field_name] = new_var
    return dataclasses.replace(op, **new_fields)


def _add_assume_divby(x: Var,
                      df_result: DataflowResult,
                      op_list: list[Operation],
                      var_map: dict[str, Var]) -> Var:
    if x.name in var_map:
        return x
    MAX_DIVBY = 1024
    divisor = df_result[x.name].div_by
    power_of_2_d = min(divisor & -divisor, MAX_DIVBY)
    if power_of_2_d > 1:
        result_var = x.ctx.make_var_like(x)
        result_var.set_type(x.get_type())
        op = AssumeDivBy(divisor=power_of_2_d,
                         x=x,
                         result_vars=(result_var,), loc=x.loc)
        op_list.append(op)
        var_map[x.name] = result_var
        old_pred = df_result[x.name]
        df_result.predicates[result_var.name] = DataPredicate(
            alias_set=old_pred.alias_set,
            div_by=power_of_2_d,
            may_alias_internally=old_pred.may_alias_internally
            )
        return result_var
    return x
