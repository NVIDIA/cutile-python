# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Set, Tuple, Dict, Any, Optional, List

from cuda.tile._exception import Loc
from cuda.tile._ir.ir import Block, Operation, Var, IRContext, MemoryEffect
from cuda.tile._ir.ops import Loop, Continue, Break, EndBranch, IfElse, Return, TileReduce, TileScan


def dead_code_elimination_pass(root_block: Block) -> None:
    # Build a data-flow `graph` as a dictionary.
    # Each key is a name of a "consumer" variable, and its value is a sequence of its dependencies.
    # Additionally, we build a set named `used` that initially includes all variables that
    # are used as immediate inputs of operations with side effects.
    graph: Dict[str, List[str] | Tuple[str, ...]] = dict()
    used: Set[str] = set()
    op_to_cf_name: Dict[Operation, str] = dict()
    _build_dataflow_graph(graph, used, op_to_cf_name, root_block,
                          None, None, None, None)

    # Traverse the data-flow graph to grow the set of `used` variables.
    _find_used_variables(graph, used)

    # Finally, walk the IR tree to remove unused operations and variables.
    _prune_block(root_block, used, op_to_cf_name, loop_mask=(), end_branch_mask=())


# Each Loop and IfElse is assigned a unique pseudovariable name of the form "$cf.<NUMBER>",
# so that it can participate in the data-flow graph.  The mapping from Operation to
# the pseudovariable name is maintained in the `op_to_cf_name` dictionary.
#
# This enables us to decide whether Loop and IfElse, as well as the corresponding Break/Continue
# statements, need to stay.
#
# Here are the rules for creating edges to/from the "$cf" nodes (identifiers in parentheses are
# rule names):
#
#   (CF_COND)
#       IfElse depends on its `cond` variable; a for Loop depends on its `start/stop/step`.
#       For example, `$cf.1` will depend on `$1` here:
#
#           if $1:    [$cf.1]
#               ...then-block...
#           else:
#               ...else-block...
#
#   (CF_NESTED)
#       Nested operation depends on its parent Loop/IfElse.
#       In this example, `$1` and `$cf.2` will both depend on `$cf.1`:
#
#           loop:  [$cf.1]
#               $1 = tile_load(...)
#               if $1:    [$cf.2]
#                   ...
#
#   (CF_DEFINED_VARS)
#       Loop body and result variables depend on the loop;
#       IfElse result variables depend on the IfElse operation.
#
#       In this example, `a.2` and `a.3` depend on `$cf.1`:
#
#           a.3 = loop (with a.2 = ...):   [$cf.1]
#               ...body...
#
#   (CF_BREAK_CONTINUE)
#       Break or Continue operation makes its loop depend on the innermost control op
#       that contains the Break/Continue operation.
#
#       This, together with the CF_NESTED rule, ensures that if a loop stays in the program,
#       so do all its Break/Continue ops, no matter how deeply nested inside IfElse statements.
#
#       In the following example, `$cf.1` depends on `$cf.3` because of the `continue` statement:
#
#           loop:  [$cf.1]
#              if foo:   [$cf.2]
#                  if bar:   [$cf.3]
#                      continue
#
def _make_control_flow_name(ctx: IRContext) -> str:
    return ctx.make_var("$cf", Loc.unknown()).name


def _build_dataflow_graph(graph: Dict[str, List[str] | Tuple[str, ...]],
                          used: Set[str],
                          op_to_cf_name: Dict[Operation, str],
                          block: Block,
                          innermost_loop: Optional[Loop],
                          innermost_loop_name: Optional[str],
                          innermost_end_branch_target: Optional[IfElse | TileReduce | TileScan],
                          innermost_cf_name: Optional[str]):
    for op in block:
        if isinstance(op, Loop):
            cf_name = _make_control_flow_name(block.ctx)
            op_to_cf_name[op] = cf_name
            graph[cf_name] = []

            # See rule `CF_DEFINED_VARS`
            for init_var, body_var, res_var in zip(op.initial_values, op.body_vars, op.result_vars,
                                                   strict=True):
                graph[body_var.name] = [init_var.name, cf_name]
                graph[res_var.name] = [cf_name]

            # See rule `CF_NESTED`
            if innermost_cf_name is not None:
                graph[cf_name].append(innermost_cf_name)

            if op.is_for_loop:
                # See rule `CF_COND`
                graph[cf_name].append(op.start.name)
                graph[cf_name].append(op.stop.name)
                graph[cf_name].append(op.step.name)

                # `For` loop can run for zero iterations, which means that initial values
                # of loop variables may flow directly into the loop's result variables.
                for res_var, init_var in zip(op.result_vars, op.initial_values, strict=True):
                    graph[res_var.name].append(init_var.name)

            _build_dataflow_graph(graph, used, op_to_cf_name, op.body,
                                  op, cf_name, None, cf_name)
        elif isinstance(op, Continue):
            assert innermost_loop_name is not None

            # See rule `CF_BREAK_CONTINUE`.
            graph[innermost_loop_name].append(innermost_cf_name)

            # "Next" values feed into the body variables of the next iteration
            for body_var, next_var in zip(innermost_loop.body_vars, op.values, strict=True):
                graph[body_var.name].append(next_var.name)

            # In a `for` loop, "next" values can also feed into the loop's results.
            # That's because the loop can immediately exit if the iterator has been exhausted.
            if innermost_loop.is_for_loop:
                for res_var, next_var in zip(innermost_loop.result_vars, op.values, strict=True):
                    graph[res_var.name].append(next_var.name)
        elif isinstance(op, Break):
            assert innermost_loop_name is not None

            # See rule `CF_BREAK_CONTINUE`.
            graph[innermost_loop_name].append(innermost_cf_name)

            # "Output" values feed into the loop's result variables
            for res_var, out_var in zip(innermost_loop.result_vars, op.values, strict=True):
                graph[res_var.name].append(out_var.name)
        elif isinstance(op, IfElse):
            cf_name = _make_control_flow_name(block.ctx)
            op_to_cf_name[op] = cf_name

            # See rule `CF_COND`
            graph[cf_name] = [op.cond.name]

            # See rule `CF_NESTED`
            if innermost_cf_name is not None:
                graph[cf_name].append(innermost_cf_name)

            # See rule `CF_DEFINED_VARS`
            for res_var in op.result_vars:
                graph[res_var.name] = [cf_name]

            _build_dataflow_graph(graph, used, op_to_cf_name, op.then_block,
                                  innermost_loop, innermost_loop_name, op, cf_name)
            _build_dataflow_graph(graph, used, op_to_cf_name, op.else_block,
                                  innermost_loop, innermost_loop_name, op, cf_name)
        elif isinstance(op, TileReduce | TileScan):
            cf_name = _make_control_flow_name(block.ctx)
            op_to_cf_name[op] = cf_name

            graph[cf_name] = []

            # See rule `CF_NESTED`
            if innermost_cf_name is not None:
                graph[cf_name].append(innermost_cf_name)

            for x, res_var, a, b in zip(op.xs, op.result_vars, op.lhs, op.rhs, strict=True):
                graph[a.name] = [x.name, res_var.name]
                graph[b.name] = [x.name, res_var.name]
                # See rule `CF_DEFINED_VARS`
                graph[res_var.name] = [cf_name, x.name, a.name, b.name]

            _build_dataflow_graph(graph, used, op_to_cf_name, op.body,
                                  None, None, op, cf_name)

        elif isinstance(op, EndBranch):
            # Yielded values flow into the IfElse/TileReduce's result variables.
            for res_var, out_var in zip(innermost_end_branch_target.result_vars, op.outputs,
                                        strict=True):
                graph[res_var.name].append(out_var.name)
        else:
            deps = tuple(v.name for v in op.all_inputs())

            # See rule `CF_NESTED`.
            if innermost_cf_name is not None:
                deps += (innermost_cf_name,)

            if _must_keep(op):
                used.update(deps)

            for dst_var in op.result_vars:
                graph[dst_var.name] = deps


def _must_keep(op: Operation) -> bool:
    return op.memory_effect == MemoryEffect.STORE or isinstance(op, Return)


def _find_used_variables(dataflow_graph: Dict[str, Sequence[str]], used: Set[str]):
    pending = list(used)
    while pending:
        dst = pending.pop()
        for src in dataflow_graph.get(dst, ()):
            if src not in used:
                used.add(src)
                pending.append(src)


def _prune_block(block: Block,
                 used_vars: Set[str],
                 op_to_cf_name: Dict[Operation, str],
                 loop_mask: Tuple[bool, ...],
                 end_branch_mask: Tuple[bool, ...]):
    new_ops = []
    for op in block.operations:
        if isinstance(op, Loop):
            if op_to_cf_name[op] in used_vars:
                mask = tuple(body_var.name in used_vars or res_var.name in used_vars
                             for body_var, res_var in zip(op.body_vars, op.result_vars,
                                                          strict=True))
                _mark_unused_vars_as_undefined(op.initial_values, mask, used_vars)
                new_initial_values = _select_by_mask(op.initial_values, mask)
                new_body_vars = _select_by_mask(op.body_vars, mask)
                op.body.params = ((op.body.params[0], *new_body_vars)
                                  if op.is_for_loop else new_body_vars)
                new_result_vars = _select_by_mask(op.result_vars, mask)
                _prune_block(op.body, used_vars, op_to_cf_name, mask, ())
                new_ops.append(Loop(op.start, op.stop, op.step, new_initial_values, new_result_vars,
                                    op.body, op.loc))
        elif isinstance(op, Continue):
            _mark_unused_vars_as_undefined(op.values, loop_mask, used_vars)
            next_vars = _select_by_mask(op.values, loop_mask)
            new_ops.append(Continue(op.loc, next_vars))
        elif isinstance(op, Break):
            _mark_unused_vars_as_undefined(op.values, loop_mask, used_vars)
            output_vars = _select_by_mask(op.values, loop_mask)
            new_ops.append(Break(op.loc, output_vars))
        elif isinstance(op, IfElse):
            if op_to_cf_name[op] in used_vars:
                mask = tuple(v.name in used_vars for v in op.result_vars)
                _prune_block(op.then_block, used_vars, op_to_cf_name, loop_mask, mask)
                _prune_block(op.else_block, used_vars, op_to_cf_name, loop_mask, mask)
                new_result_vars = _select_by_mask(op.result_vars, mask)
                new_ops.append(IfElse(op.cond, op.then_block, op.else_block, new_result_vars,
                                      op.loc))
        elif isinstance(op, TileReduce):
            if op_to_cf_name[op] in used_vars:
                mask = tuple(v.name in used_vars for v in op.result_vars)
                _prune_block(op.body, used_vars, op_to_cf_name, (), mask)
                new_xs = _select_by_mask(op.xs, mask)
                new_identities = _select_by_mask(op.identities, mask)
                new_lhs = _select_by_mask(op.lhs, mask)
                new_rhs = _select_by_mask(op.rhs, mask)
                op.body.params = new_lhs + new_rhs
                new_result_vars = _select_by_mask(op.result_vars, mask)
                new_ops.append(TileReduce(xs=new_xs, identities=new_identities, axis=op.axis,
                                          body=op.body, result_vars=new_result_vars, loc=op.loc))
        elif isinstance(op, TileScan):
            if op_to_cf_name[op] in used_vars:
                mask = tuple(v.name in used_vars for v in op.result_vars)
                _prune_block(op.body, used_vars, op_to_cf_name, (), mask)
                new_xs = _select_by_mask(op.xs, mask)
                new_identities = _select_by_mask(op.identities, mask)
                new_lhs = _select_by_mask(op.lhs, mask)
                new_rhs = _select_by_mask(op.rhs, mask)
                op.body.params = new_lhs + new_rhs
                new_result_vars = _select_by_mask(op.result_vars, mask)
                new_ops.append(TileScan(xs=new_xs, identities=new_identities, axis=op.axis,
                                        reverse=op.reverse, body=op.body,
                                        result_vars=new_result_vars, loc=op.loc))
        elif isinstance(op, EndBranch):
            output_vars = _select_by_mask(op.outputs, end_branch_mask)
            new_ops.append(EndBranch(op.loc, output_vars))
        elif any(r.name in used_vars for r in op.result_vars) or _must_keep(op):
            new_ops.append(op)
    block.operations = new_ops


# We keep a carried variable as long as its body variable or its result variable is used.
#
# This may create some undefined variables, which is OK. To explicitly say that is OK,
# we mark the Var objects as undefined.
#
# Example 1:
# ==========
#   In the following case, the result variable `x.2` is used,
#   therefore `x` must be kept as a carried variable:
#
#       Source code                       IR
#       ===========                       ===============
#       x = ct.ones(...)                  x = tile_ones(...)
#       while True:                       x.2 = loop (with x.1 = x):
#           x = ct.load(...)                  x.3 = tile_load(...)
#           ct.store(..., x)                  tile_store(..., x.3)
#           if ...:                           if ...:
#               break                             break x.3
#           x = x + 1                         x.4 = x.3 + 1
#                                             continue x.4
#       ct.store(..., x)                  tile_store(..., x.2)
#
#   However, the body variable `x.1` is unused. This also makes the initial variable `x`,
#   as well as the continuation value `x.4`, unused. Thus, dead code elimination will prune
#   the program as such:
#
#       x.2 = loop (with x.1 = <undefined x>):
#           x.3 = tile_load(...)
#           tile_store(..., x.3)
#           if ...:
#               break x.3
#           continue <undefined x.4>
#       tile_store(..., x.2)
#
#
# Example 2:
# ==========
#   Consider the opposite case, where the body variable is used but the result variable isn't:
#
#       Source code                       IR
#       ===========                       ===============
#       x = ct.ones(...)                  x = tile_ones(...)
#       while True:                       x.2 = loop (with x.1 = x):
#           ct.store(..., x)                  tile_store(..., x.1)
#           x = x + 1                         x.3 = x.1 + 1
#           if ...:                           x.4 = if ...:
#               x = x + 2                         x.5 = x.3 + 2
#               break                             break x.5
#                                             else:
#                                                 yield x.3
#                                             continue x.4
#       # x is never used afterward
#
#   Since `x.2` is unused, `x.5` is also unused. Thus, the pruned IR looks like so:
#
#       x = tile_ones(...)
#       x.2 = loop (with x.1 = x):
#           tile_store(..., x.1)
#           x.3 = x.1 + 1
#           x.4 = if ...:
#               break <undefined x.5>
#           else:
#               yield x.3
#         continue x.4
#
def _mark_unused_vars_as_undefined(vars: Sequence[Var], mask: Sequence[bool], used_vars: Set[str]):
    for v, keep in zip(vars, mask, strict=True):
        if keep and v.name not in used_vars:
            v.set_undefined()


def _select_by_mask(seq: Sequence[Any], mask: Sequence[bool]) -> Tuple[Any, ...]:
    return tuple(x for x, keep in zip(seq, mask, strict=True) if keep)
