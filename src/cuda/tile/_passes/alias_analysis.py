# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import FrozenSet, Dict

from cuda.tile._ir.ir import Var, Block
from cuda.tile._ir.ops import Assign, GetArrayListItem, \
    Loop, IfElse, Continue, Break, EndBranch, PointerOffset, \
    TileBroadcast, TileReshape, MakeTensorView, MakeListView, AssumeDivBy, TileReduce, TileScan


class AliasUniverseClass:
    # Union with other set always gives the universe
    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    # Intersection with other set always gives the other set
    def __and__(self, other):
        return other

    def __rand__(self, other):
        return other

    def __bool__(self):
        return True

    def __repr__(self):
        return "UNIVERSE"


ALIAS_UNIVERSE = AliasUniverseClass()

AliasSet = FrozenSet[str] | AliasUniverseClass


@dataclass
class AliasResult:
    aliases: Dict[str, AliasSet]

    def __getitem__(self, var_name: str) -> AliasSet:
        return self.aliases.get(var_name, ALIAS_UNIVERSE)


def alias_analysis_pass(root_block: Block) -> AliasResult:
    alias_tracker = _AliasTracker()
    for p in root_block.params:
        alias_tracker[p.name] = frozenset([p.name])

    _analyze_aliases_in_block(root_block, alias_tracker, None, None)

    while alias_tracker.dirty:
        alias_tracker.dirty = False
        _analyze_aliases_in_block(root_block, alias_tracker, None, None)

    return AliasResult(alias_tracker.finalize())


class _AliasTracker:
    def __init__(self):
        self.dirty = False
        self._aliases: Dict[str, AliasSet] = dict()

    def __getitem__(self, var_name: str) -> AliasSet:
        return self._aliases[var_name]

    def __setitem__(self, var_name: str, alias_set: AliasSet):
        if var_name not in self._aliases or self._aliases[var_name] != alias_set:
            self.dirty = True
        self._aliases[var_name] = alias_set

    def get(self, var_name: str, default: AliasSet) -> AliasSet:
        return self._aliases.get(var_name, default)

    def finalize(self):
        return self._aliases


def _propagate(alias_tracker: _AliasTracker,
               src: Var,
               dst: Var):
    if src.is_undefined():
        alias_tracker[src.name] = frozenset()

    src_aliases = alias_tracker[src.name]
    dst_aliases = alias_tracker.get(dst.name, frozenset())
    alias_tracker[dst.name] = dst_aliases | src_aliases


def _analyze_aliases_in_block(block: Block,
                              alias_tracker: _AliasTracker,
                              innermost_loop: Loop | None,
                              innermost_branch: IfElse | TileReduce | TileScan | None):
    for op in block.operations:
        if isinstance(op, Assign):
            _propagate(alias_tracker, op.value, op.result_var)
        elif isinstance(op, AssumeDivBy):
            _propagate(alias_tracker, op.x, op.result_var)
        elif isinstance(op, GetArrayListItem):
            # TODO: more granular array list get item alias analysis
            # Propagate to the base pointer of the array
            _propagate(alias_tracker, op.x, op.result_vars[0])
            for v in op.result_vars[1:]:
                alias_tracker[v.name] = ALIAS_UNIVERSE
        elif isinstance(op, MakeTensorView):
            _propagate(alias_tracker, op.base_ptr, op.result_var)
        elif isinstance(op, MakeListView):
            _propagate(alias_tracker, op.base_ptr, op.result_var)
        elif isinstance(op, PointerOffset):
            _propagate(alias_tracker, op.pointer, op.result_var)
        elif isinstance(op, TileBroadcast | TileReshape):
            # Needed for tiles of pointers produced by gather/scatter
            _propagate(alias_tracker, op.x, op.result_var)
        elif isinstance(op, Loop):
            if op.is_for_loop:
                alias_tracker[op.induction_var.name] = ALIAS_UNIVERSE

            for init, body, result in zip(op.initial_values, op.body_vars, op.result_vars,
                                          strict=True):
                # Loop initial values flow into body values.
                _propagate(alias_tracker, init, body)

                # `For` loop initial values can flow into result values if
                # loop runs for 0 iteration.
                if op.is_for_loop:
                    _propagate(alias_tracker, init, result)

            _analyze_aliases_in_block(op.body, alias_tracker, op, None)

        elif isinstance(op, Continue):
            for next, body, result in zip(op.values, innermost_loop.body_vars,
                                          innermost_loop.result_vars, strict=True):
                # Loop next values can flow into body values
                _propagate(alias_tracker, next, body)

                # `For` loop next values can flow into result values when
                # the iterator is exhausted.
                if innermost_loop.is_for_loop:
                    _propagate(alias_tracker, next, result)

        elif isinstance(op, Break):
            for output, result in zip(op.values, innermost_loop.result_vars, strict=True):
                _propagate(alias_tracker, output, result)

        elif isinstance(op, IfElse):
            _analyze_aliases_in_block(op.then_block, alias_tracker, innermost_loop, op)

            _analyze_aliases_in_block(op.else_block, alias_tracker, innermost_loop, op)

        elif isinstance(op, EndBranch):
            for output, result in zip(op.outputs, innermost_branch.result_vars, strict=True):
                _propagate(alias_tracker, output, result)

        elif isinstance(op, TileReduce | TileScan):
            for v in op.body.params:
                alias_tracker[v.name] = ALIAS_UNIVERSE
            _analyze_aliases_in_block(op.body, alias_tracker, None, op)

        else:
            assert len(op.nested_blocks) == 0
            for v in op.result_vars:
                alias_tracker[v.name] = ALIAS_UNIVERSE
