# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.core_ops import TypedConst
from cuda.tile._ir.ir import Block, Mapper, Operation
from cuda.tile._passes.dataflow_analysis import DataflowResult


def materialize_constants_pass(root_block: Block, dataflow_result: DataflowResult):
    """Replace uses of dataflow-proven scalar values with IR constants."""
    mapper = Mapper(root_block.ctx, preserve_vars=True, remap_uses_only=True)
    new_ops: list[Operation] = []

    for var in root_block.all_defined_vars():
        if var.is_constant():
            continue
        value = dataflow_result.constant_value(var)
        if value is None:
            continue

        constant_var = var.ctx.make_var_like(var)
        var.ctx.copy_type_information(var, constant_var)
        mapper.set_var(var, constant_var)
        dataflow_result.predicates[constant_var.name] = dataflow_result[var.name]
        new_ops.append(TypedConst(value=value, result_vars=(constant_var,), loc=var.loc))

    new_ops.extend(op.clone(mapper) for op in root_block)
    root_block[:] = new_ops
