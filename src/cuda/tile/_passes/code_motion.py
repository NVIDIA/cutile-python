# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from cuda.tile._ir.ir import Block, Var, MemoryEffect
from cuda.tile._ir.ops import Loop, IfElse, Continue, Break, EndBranch, Return, TileReduce, TileScan

from dataclasses import dataclass
import enum


def hoist_loop_invariants(root_block: Block):
    def_depth = {p.name: 0 for p in root_block.params}
    _hoist(root_block, [], def_depth, False)


# Indicates whether a block could in theory be moved, based purely on the classes of operations
# contained within (i.e. whether it contains any side effects or jumps).
# It does not take actual data dependencies into account.
class _BlockMobility(enum.IntEnum):
    # Under no circumstances the block (or any of its ancestors) could be moved.
    # For example, this happens when the block contains an operation with a side effect.
    IMMOVABLE = 0

    # The block itself can't be hoisted by itself, but the loop that contains still may.
    # This happens when the block contains a Continue or a Break statement.
    CAN_MOVE_WITH_LOOP = 1

    # The block can move. Note that it may still not be possible to hoist it,
    # depending on whether it depends on values defined inside its parent loop.
    CAN_MOVE = 2


@dataclass
class _BlockResult:
    mobility: _BlockMobility = _BlockMobility.CAN_MOVE

    # Minimum depth to which the block could be hoisted, based on its data dependencies
    min_depth: int = 0


# Helper class for accumulating data dependency information per operation.
@dataclass
class _DependencyInfo:
    # True if this operation can't be hoisted
    must_stay: bool

    # Maximum depth of this operation's data dependencies' definitions, excluding
    # the block where the operation is originally defined.
    # I.e., this is the minimum depth to which we could hoist the operation,
    # based solely on its data dependencies.
    max_outside_depth: int = 0

    def update(self, dependency_depth: int, cur_block_depth: int):
        if dependency_depth >= cur_block_depth:
            # If the operation depends on another operation in its block, it can't be hoisted.
            # But also it doesn't affect max_outside_depth in case the whole block is moved at once.
            self.must_stay = True
        else:
            # Otherwise, we update the maximum data dependency depth.
            self.max_outside_depth = max(self.max_outside_depth, dependency_depth)


@dataclass
class _StackItem:
    new_block: Block
    is_loop_body: bool


# This function does too many things at once. However, this allows us to do everything
# in a single linear-time pass, no matter how many nested loops we may have.
def _hoist(block: Block, stack: list[_StackItem], def_depth: dict[str, int], is_loop_body: bool) \
        -> _BlockResult:
    depth = len(stack)
    new_block = block.empty_like_self()
    stack.append(_StackItem(new_block, is_loop_body))
    ret = _BlockResult()

    for op in block:
        # We can only hoist operations out of loops, not other nested blocks like IfElse branches.
        depinfo = _DependencyInfo(must_stay=not is_loop_body)

        if isinstance(op, Loop | TileReduce | TileScan):
            for var in op.body.params:
                def_depth[var.name] = depth + 1

            body_res = _hoist(op.body, stack, def_depth, True)
            if body_res.mobility == _BlockMobility.IMMOVABLE:
                # Propagate IMMOVABLE to all ancestors.
                ret.mobility = _BlockMobility.IMMOVABLE
                depinfo.must_stay = True

            inputs = op.initial_values if isinstance(op, Loop) else op.xs
            for v in inputs:
                depinfo.update(_get_def_depth(def_depth, v), depth)
            depinfo.update(body_res.min_depth, depth)
        elif isinstance(op, IfElse):
            depinfo.update(_get_def_depth(def_depth, op.cond), depth)
            for branch in (op.then_block, op.else_block):
                branch_res = _hoist(branch, stack, def_depth, False)
                depinfo.update(branch_res.min_depth, depth)
                if branch_res.mobility != _BlockMobility.CAN_MOVE:
                    # Propagate CAN_MOVE_WITH_LOOP and IMMOVABLE
                    ret.mobility = min(ret.mobility, branch_res.mobility)
                    depinfo.must_stay = True
        elif op.memory_effect == MemoryEffect.STORE or isinstance(op, Return):
            ret.mobility = _BlockMobility.IMMOVABLE
            depinfo.must_stay = True
        elif isinstance(op, (Continue, Break)):
            # Can't move the block that contains a Continue/Break, unless it is moved
            # together with its containing loop.
            ret.mobility = min(ret.mobility, _BlockMobility.CAN_MOVE_WITH_LOOP)
            depinfo.must_stay = True
        elif isinstance(op, EndBranch):
            depinfo.must_stay = True
            for v in op.outputs:
                depinfo.update(_get_def_depth(def_depth, v), depth)
        else:
            # "Pure" operation without any nested blocks, side effects and jumps.
            assert len(op.nested_blocks) == 0
            for v in op.all_inputs():
                depinfo.update(_get_def_depth(def_depth, v), depth)

        target_depth = depth
        if depinfo.must_stay:
            ret.min_depth = max(ret.min_depth, depinfo.max_outside_depth)
        else:
            while target_depth > depinfo.max_outside_depth and stack[target_depth].is_loop_body:
                target_depth -= 1

        stack[target_depth].new_block.append(op)

        # Record the definition depth of the results variables. Note that we do this
        # after hoisting, so that for any subsequent operations, we compute max_outside_depth
        # based on the target depth, not on the original depth.
        for v in op.result_vars:
            def_depth[v.name] = target_depth

    stack.pop()
    block[:] = new_block.detach_all()
    return ret


def _get_def_depth(def_depth: dict[str, int], var: Var) -> int:
    try:
        return def_depth[var.name]
    except KeyError:
        pass
    assert var.is_undefined(), var.name
    return 0
