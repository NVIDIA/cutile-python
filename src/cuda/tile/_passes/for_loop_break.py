# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.ir import Block
from cuda.tile._ir.control_flow_ops import Loop, IfElse, Break, Continue, EndBranch
from cuda.tile._ir.typing_support import type_of_constant_python_value
from cuda.tile._ir.core_ops import TypedConst


def _has_break(block: Block):
    for op in block:
        if isinstance(op, Break):
            return True
        elif isinstance(op, IfElse):
            if _has_break(op.then_block) or _has_break(op.else_block):
                return True
    return False


def _append_continue(loop: Loop, done_body):

    then_block = Block(loop.body.ctx, loop.loc)
    then_block.append(Continue(values=tuple(loop.body.params[1:]), result_vars=(), loc=loop.loc))

    else_block = Block(loop.body.ctx, loop.loc)
    else_block.append(EndBranch(outputs=(), result_vars=(), loc=loop.loc))

    top_continue = IfElse(cond=done_body, then_block=then_block, else_block=else_block,
                          result_vars=(), loc=loop.loc)

    new_body = loop.body.empty_like_self()
    new_body.params = loop.body.params
    new_body.append(top_continue)
    new_body.extend(loop.body)
    loop.body[:] = new_body.detach_all()


def _rewrite(block: Block, done_body, done_flag):
    new_block = block.empty_like_self()
    new_block.params = block.params

    for op in block:
        if isinstance(op, Break):
            new_block.append(Continue(values=(*op.values, done_flag), result_vars=(), loc=op.loc))
        else:
            if isinstance(op, Continue):
                op.values = (*op.values, done_body)
            elif isinstance(op, IfElse):
                _rewrite(op.then_block, done_body, done_flag)
                _rewrite(op.else_block, done_body, done_flag)
            new_block.append(op)
    block[:] = new_block.detach_all()


def lower_for_with_break(block: Block) -> None:
    new_block = block.empty_like_self()

    for op in block:
        for inner in op.nested_blocks:
            lower_for_with_break(inner)

        if isinstance(op, Loop) and op.is_for_loop and _has_break(op.body):
            done_ty = type_of_constant_python_value(False, block.ctx.typing_hooks)

            done_init = new_block.make_temp_var(op.loc)
            done_init.set_type(done_ty)
            new_block.append(TypedConst(value=False, result_vars=(done_init,), loc=op.loc))

            done_true = new_block.make_temp_var(op.loc)
            done_true.set_type(type_of_constant_python_value(True, block.ctx.typing_hooks))
            new_block.append(TypedConst(value=True, result_vars=(done_true,), loc=op.loc))

            done_body = op.body.make_temp_var(op.loc)
            done_body.set_type(done_ty)

            done_result = new_block.make_temp_var(op.loc)
            done_result.set_type(done_ty)

            op.body.params = (*op.body.params, done_body)
            op.initial_values = (*op.initial_values, done_init)
            op.result_vars = (*op.result_vars, done_result)

            _rewrite(op.body, done_body, done_true)
            _append_continue(op, done_body)

        new_block.append(op)

    block[:] = new_block.detach_all()
