# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.ir import Block
from cuda.tile._ir.control_flow_ops import Loop, IfElse, Break, Continue, EndBranch, Return
from cuda.tile._ir.typing_support import type_of_constant_python_value
from cuda.tile._ir.core_ops import TypedConst


def _has_return(block: Block):
    for op in block:
        if isinstance(op, Return):
            return True
        elif isinstance(op, IfElse):
            if _has_return(op.then_block) or _has_return(op.else_block):
                return True
    return False


def _rewrite(block: Block, returned_values, returned_body, returned_flag):
    new_block = block.empty_like_self()
    new_block.params = block.params

    for op in block:
        if isinstance(op, Return):
            new_block.append(Break(values=(*returned_values, returned_flag), result_vars=(),
                                   loc=op.loc))
        else:
            if isinstance(op, (Break, Continue)):
                op.values = (*op.values, returned_body)
            elif isinstance(op, IfElse):
                _rewrite(op.then_block, returned_values, returned_body, returned_flag)
                _rewrite(op.else_block, returned_values, returned_body, returned_flag)
            new_block.append(op)
    block[:] = new_block.detach_all()


def lower_for_with_return(block: Block, is_in_for_loop: bool = False) -> None:
    new_block = block.empty_like_self()

    for op in block:
        inner_in_for_loop = is_in_for_loop or (isinstance(op, Loop) and op.is_for_loop)
        for inner in op.nested_blocks:
            lower_for_with_return(inner, inner_in_for_loop)

        if isinstance(op, Loop) and (op.is_for_loop or is_in_for_loop) and _has_return(op.body):
            returned_ty = type_of_constant_python_value(False, block.ctx.typing_hooks)

            returned_init = new_block.make_temp_var(op.loc)
            returned_init.set_type(returned_ty)
            new_block.append(TypedConst(value=False, result_vars=(returned_init,), loc=op.loc))

            returned_true = new_block.make_temp_var(op.loc)
            returned_true.set_type(type_of_constant_python_value(True, block.ctx.typing_hooks))
            new_block.append(TypedConst(value=True, result_vars=(returned_true,), loc=op.loc))

            returned_body = op.body.make_temp_var(op.loc)
            returned_body.set_type(returned_ty)

            returned_result = new_block.make_temp_var(op.loc)
            returned_result.set_type(returned_ty)

            returned_values = op.body_vars

            op.body.params = (*op.body.params, returned_body)
            op.initial_values = (*op.initial_values, returned_init)
            op.result_vars = (*op.result_vars, returned_result)

            _rewrite(op.body, returned_values, returned_body, returned_true)
            new_block.append(op)

            then_block = Block(op.body.ctx, op.loc)
            then_block.append(Return(result_vars=(), loc=op.loc))

            else_block = Block(op.body.ctx, op.loc)
            else_block.append(EndBranch(outputs=(), result_vars=(), loc=op.loc))

            ifelse_ret = IfElse(cond=returned_result, then_block=then_block, else_block=else_block,
                                result_vars=(), loc=op.loc)
            new_block.append(ifelse_ret)

        else:
            new_block.append(op)

    block[:] = new_block.detach_all()
