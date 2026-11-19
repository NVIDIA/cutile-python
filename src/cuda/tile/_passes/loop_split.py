# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Optional, Set, Dict, DefaultDict, Mapping, NamedTuple

from cuda.tile._datatype import is_signed
from cuda.tile._ir.ir import Block, Var, Mapper, IRContext
from cuda.tile._ir.control_flow_ops import Loop, IfElse, EndBranch, Continue
from cuda.tile._ir.core_ops import TypedConst, Assign
from cuda.tile._ir.arithmetic_ops import RawBinaryArithmeticOperation, RawComparisonOperation
from cuda.tile._ir.type import TileTy


class _Condition(NamedTuple):
    cmp: str
    rhs: Var


_FLIP = {"ge": "le", "gt": "lt", "le": "ge", "lt": "gt"}


def _find_splittable_loops(block: Block,
                           def_depth: Dict[str, int],
                           depth: int,
                           for_loop: Optional[Loop],
                           induction_var: Optional[str],
                           comparisons: Dict[str, _Condition],
                           equiv_map: Dict[str, Var],
                           result: DefaultDict[Loop, Dict[IfElse, _Condition]]):
    for op in block:
        if isinstance(op, RawComparisonOperation):
            if op.fn in ("ge", "gt", "le", "lt"):
                lhs = equiv_map.get(op.lhs.name, op.lhs)
                rhs = equiv_map.get(op.rhs.name, op.rhs)
                if lhs.name == induction_var and def_depth[rhs.name] < depth:
                    comparisons[op.result_var.name] = _Condition(op.fn, rhs)
                elif rhs.name == induction_var and def_depth[lhs.name] < depth:
                    comparisons[op.result_var.name] = _Condition(_FLIP[op.fn], lhs)
        elif isinstance(op, IfElse):
            cond = equiv_map.get(op.cond.name, op.cond)
            if cond.name in comparisons:
                assert for_loop is not None
                result[for_loop][op] = comparisons[cond.name]
            _find_splittable_loops(op.then_block, def_depth, depth + 1, None, None, dict(),
                                   equiv_map, result)
            _find_splittable_loops(op.else_block, def_depth, depth + 1, None, None, dict(),
                                   equiv_map, result)
        elif isinstance(op, Assign):
            equiv_map[op.result_var.name] = equiv_map.get(op.value.name, op.value)
        elif isinstance(op, Loop):
            good_loop = (op.is_for_loop
                         and op.step.is_constant()
                         and op.step.get_constant() == 1)
            for var in op.body.params:
                def_depth[var.name] = depth + 1
            _find_splittable_loops(
                op.body,
                def_depth,
                depth + 1,
                op if good_loop else None,
                op.body.params[0].name if good_loop else None,
                dict(),
                equiv_map,
                result
            )

        for v in op.result_vars:
            def_depth[v.name] = depth


_NEED_TO_ADJUST_RANGE = {"ge": False, "gt": True, "le": True, "lt": False}
_BRANCH_TO_KEEP = {"ge": ("else_block", "then_block"),
                   "gt": ("else_block", "then_block"),
                   "le": ("then_block", "else_block"),
                   "lt": ("then_block", "else_block")}


def _apply_splits(block: Block,
                  loops_to_split: Mapping[Loop, _Condition],
                  if_ops_to_flatten: Set[IfElse]):
    new_block = block.empty_like_self()
    for op in block:
        for nested in op.nested_blocks:
            _apply_splits(nested, loops_to_split, if_ops_to_flatten)

        if isinstance(op, Loop) and op in loops_to_split:
            _split_loop(op, loops_to_split[op], if_ops_to_flatten, new_block)
        else:
            new_block.append(op)

    block[:] = new_block.detach_all()


# This is horrible
def _split_loop(loop: Loop, cond: _Condition, if_ops_to_flatten: Set[IfElse], new_block: Block):
    range_dtype = loop.start.get_type()
    split_value = cond.rhs
    loc = loop.loc
    if _NEED_TO_ADJUST_RANGE[cond.cmp]:
        # For `i > c` and `i <= c` the split point is `c + 1`, which wraps around
        # when `c` is the maximum value of the induction variable's type. Saturate
        # it to that maximum instead. Because `stop <= max`, the first loop then
        # covers the whole range and the second loop is empty, which is exactly
        # right since `i <= max` always holds and `i > max` never does.
        split_value = _saturating_add_one(split_value, range_dtype, new_block, loc)

    first_loop_stop = new_block.make_temp_var()
    new_block.append(RawBinaryArithmeticOperation(fn="min", lhs=loop.stop, rhs=split_value,
                                                  result_vars=(first_loop_stop,), loc=loc))

    second_loop_start = new_block.make_temp_var()
    new_block.append(RawBinaryArithmeticOperation(fn="max", lhs=loop.start, rhs=split_value,
                                                  result_vars=(second_loop_start,), loc=loc))

    for var in first_loop_stop, second_loop_start:
        var.set_type(range_dtype)

    first_branch, second_branch = _BRANCH_TO_KEEP[cond.cmp]

    intermediate_vars = tuple(new_block.ctx.make_var_like(v) for v in loop.result_vars)
    for old_var, new_var in zip(loop.result_vars, intermediate_vars, strict=True):
        new_var.set_type(old_var.get_type())

    new_block.append(_clone_loop(loop, loop.start, first_loop_stop, loop.step,
                                 loop.initial_values, intermediate_vars,
                                 if_ops_to_flatten, first_branch, new_block.ctx))

    second_loop = _clone_loop(loop, second_loop_start, loop.stop, loop.step,
                              intermediate_vars, loop.result_vars,
                              if_ops_to_flatten, second_branch, new_block.ctx)
    new_block.append(second_loop)


def _saturating_add_one(value: Var, ty: TileTy, block: Block, loc) -> Var:
    # min(value, max - 1) + 1 is value + 1 saturated at the type's maximum. The add
    # can never overflow, so the result does not depend on wraparound semantics.
    dtype = ty.dtype
    max_value = (1 << (dtype.bitwidth - 1)) - 1 if is_signed(dtype) else (1 << dtype.bitwidth) - 1

    def const(v):
        var = block.make_temp_var()
        block.append(TypedConst(value=v, result_vars=(var,), loc=loc))
        var.set_type(ty)
        return var

    def binop(fn, lhs, rhs):
        var = block.make_temp_var()
        block.append(RawBinaryArithmeticOperation(fn=fn, lhs=lhs, rhs=rhs,
                                                  result_vars=(var,), loc=loc))
        var.set_type(ty)
        return var

    clamped = binop("min", value, const(max_value - 1))
    return binop("add", clamped, const(1))


def _clone_loop(loop: Loop, new_start: Var, new_stop: Var, new_step: Var,
                initial_vars: tuple[Var, ...], result_vars: tuple[Var, ...],
                if_ops_to_flatten: Set[IfElse], branch_to_keep: str, ctx: IRContext) -> Loop:
    mapper = Mapper(ctx)
    new_body = Block(ctx, loop.body.loc)
    new_body.params = mapper.clone_vars(loop.body.params)
    for body_op in loop.body:
        if isinstance(body_op, IfElse) and body_op in if_ops_to_flatten:
            early_continue = False
            branch = getattr(body_op, branch_to_keep)
            for branch_op in branch:
                if isinstance(branch_op, EndBranch):
                    for old_res, branch_res in zip(body_op.result_vars, branch_op.outputs,
                                                   strict=True):
                        new_res = mapper.get_var(branch_res)
                        mapper.set_var(old_res, new_res)
                    break

                new_body.append(branch_op.clone(mapper))
                if isinstance(branch_op, Continue):
                    early_continue = True
                    break
            if early_continue:
                break
        else:
            new_body.append(body_op.clone(mapper))

    return Loop(start=new_start, stop=new_stop, step=new_step, initial_values=initial_vars,
                result_vars=result_vars, body=new_body, loc=loop.loc)


def split_loops(block: Block):
    splittable_loops = defaultdict(dict)
    def_depth = {param.name: 0 for param in block.params}
    _find_splittable_loops(block, def_depth, 0, None, None, dict(), dict(), splittable_loops)

    loops_to_split = dict()
    if_ops_to_flatten = set()
    for loop, if_ops in splittable_loops.items():
        # For now, only split if there is exactly one splittable `if`
        if len(if_ops) != 1:
            continue
        if_op, condition = next(iter(if_ops.items()))
        loops_to_split[loop] = condition
        if_ops_to_flatten.add(if_op)

    if len(loops_to_split) > 0:
        _apply_splits(block, loops_to_split, if_ops_to_flatten)
