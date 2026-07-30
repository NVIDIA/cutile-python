# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.ir import Block, Mapper, Operation, Var
from cuda.tile._ir.ops import AssumeDivBy, MakeTensorView, LoadPointer, StorePointer
from cuda.tile._passes.dataflow_analysis import DataflowResult

_OPS_NEED_ASSUME = (MakeTensorView, LoadPointer, StorePointer)


def add_divby_pass(root_block: Block, df_result: DataflowResult):
    candidates = set()
    for op in root_block.traverse():
        if isinstance(op, _OPS_NEED_ASSUME):
            candidates.update(var.name for var in op.all_inputs())
    mapper = Mapper(root_block.ctx, preserve_vars=True)
    _rewrite_block(root_block, df_result, mapper, candidates)


def _rewrite_block(block: Block,
                   df_result: DataflowResult,
                   mapper: Mapper,
                   candidates: set[str]):
    new_ops = []
    for param in block.params:
        _add_assume_divby(param, df_result, new_ops, mapper)

    for op in block:
        to_assume = tuple(var for var in op.result_vars if var.name in candidates)
        new_op = op.clone(mapper)
        new_ops.append(new_op)
        for var in to_assume:
            _add_assume_divby(var, df_result, new_ops, mapper)
        for b in new_op.nested_blocks:
            _rewrite_block(b, df_result, mapper, candidates)

    block[:] = new_ops


def _add_assume_divby(x: Var,
                      df_result: DataflowResult,
                      op_list: list[Operation],
                      mapper: Mapper) -> Var:
    if mapper.get_var(x) is not x:
        return x
    # The constant-materialization pass runs before this pass. An exact constant is stronger than
    # a divisibility assumption, and wrapping it in AssumeDivBy would hide that value from later
    # consumers. Keep the constant as-is rather than materializing the correlated div_by fact.
    if df_result[x.name].const_value is not None:
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
        mapper.set_var(x, result_var)
        df_result.predicates[result_var.name] = df_result[x.name].replace(
                div_by=power_of_2_d)
        return result_var
    return x
