# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import itertools
from dataclasses import dataclass
from math import gcd
from typing import Dict, Sequence, TypeVar, Generic, Any

from cuda.tile._ir.ops_utils import get_dtype
from cuda.tile._ir.type import PointerTy, ListValue
from cuda.tile._datatype import is_integral
from cuda.tile._ir.ir import Var, Block
from cuda.tile._ir.ops import Assign, GetArrayListItem, \
    Loop, IfElse, Continue, Break, EndBranch, PointerOffset, \
    TileBroadcast, TileReshape, AssumeDivBy, TileReduce, TileScan, \
    AssumeBounded, TileAsType, TypedConst, RawBinaryArithmeticOperation, Unary
from cuda.tile.compilation._signature import ParameterConstraint, \
    ArrayConstraint, ListConstraint, ScalarConstraint


ALIAS_UNIVERSE = -1
ALIAS_EMPTY = 0
AliasSet = int


@dataclass(frozen=True)
class DataPredicate:
    alias_set: AliasSet
    div_by: int
    may_alias_internally: bool

    def unify(self, other: "DataPredicate") -> "DataPredicate":
        return DataPredicate(
                alias_set=self.alias_set | other.alias_set,
                div_by=gcd(self.div_by, other.div_by),
                may_alias_internally=self.may_alias_internally | other.may_alias_internally
                )

    def replace(self, **key_value_pairs) -> "DataPredicate":
        return dataclasses.replace(self, **key_value_pairs)


@dataclass
class DataflowResult:
    predicates: Dict[str, DataPredicate]

    def __getitem__(self, var_name: str) -> DataPredicate:
        return self.predicates[var_name]


def dataflow_analysis(root_block: Block,
                      parameter_constraints: Sequence[tuple[tuple[Var, ...], ParameterConstraint]]
                      ) -> DataflowResult:
    state = _State(_Tracker(), _Tracker())
    alias_set_mapper = _AliasSetMapper()
    for flat_params, constraint in parameter_constraints:
        if isinstance(constraint, ArrayConstraint):
            predicates = _get_array_predicates(constraint, alias_set_mapper)
            for var, pred in zip(flat_params, predicates, strict=True):
                state.tracker.update(var, pred)
                state.list_array_tracker.update(var, ALWAYS_TRUE_AGG_PREDICATE)
        elif isinstance(constraint, ListConstraint):
            assert isinstance(constraint.element, ArrayConstraint)
            assert len(flat_params) == 2
            base_ptr, size_var = flat_params
            state.tracker.update(base_ptr,
                                 DataPredicate(alias_set=alias_set_mapper(constraint.alias_groups),
                                               div_by=1,
                                               may_alias_internally=constraint.elements_may_alias))
            elt_predicates = _get_array_predicates(constraint.element, alias_set_mapper)
            agg_pred = _AggregatePredicate(dict(enumerate(elt_predicates)))
            state.list_array_tracker.update(base_ptr, agg_pred)
            state.set_always_true(size_var)
        elif isinstance(constraint, ScalarConstraint):
            [only_param] = flat_params
            state.set_always_true(only_param)
        else:
            assert False

    _analyze_aliases_in_block(root_block, state, None, None)

    while state.dirty:
        state.reset_dirty()
        _analyze_aliases_in_block(root_block, state, None, None)

    return DataflowResult(state.tracker.finalize())


def _get_array_predicates(constraint: ArrayConstraint, alias_set_mapper: "_AliasSetMapper"):
    # Base pointer predicate
    ret = [DataPredicate(alias_set=alias_set_mapper(constraint.alias_groups),
                         div_by=constraint.base_addr_divisible_by,
                         may_alias_internally=constraint.may_alias_internally)]

    # Shape predicates
    for div_by in constraint.shape_divisible_by:
        ret.append(DataPredicate(alias_set=ALIAS_UNIVERSE, div_by=div_by,
                                 may_alias_internally=True))

    # Stride predicates
    for static, div_by in zip(constraint.stride_constant,
                              constraint.stride_divisible_by, strict=True):
        if static is not None:
            div_by = static
        ret.append(DataPredicate(alias_set=ALIAS_UNIVERSE, div_by=div_by,
                                 may_alias_internally=True))
    return ret


class _AliasSetMapper:
    def __init__(self):
        self._bit_seq = map(lambda x: 1 << x, itertools.count())
        self._mapping: dict[str, int] = dict()

    def __call__(self, alias_groups: Sequence[str] | None) -> AliasSet:
        if alias_groups is None:
            return ALIAS_UNIVERSE
        elif len(alias_groups) == 0:
            return next(self._bit_seq)
        else:
            ret = 0
            for ap in alias_groups:
                bit = self._mapping.get(ap)
                if bit is None:
                    bit = next(self._bit_seq)
                    self._mapping[ap] = bit
                ret |= bit
            return ret


ALWAYS_TRUE_PREDICATE = DataPredicate(alias_set=ALIAS_UNIVERSE,
                                      div_by=1,
                                      may_alias_internally=True)


class _AggregatePredicate:
    def __init__(self, items: Dict[Any, DataPredicate]):
        self._items = items

    def __getitem__(self, key):
        return self._items.get(key, ALWAYS_TRUE_PREDICATE)

    def unify(self, other: "_AggregatePredicate") -> "_AggregatePredicate":
        all_keys = set(self._items.keys()) | other._items.keys()
        unified_items = {k: self[k].unify(other[k]) for k in all_keys}
        return _AggregatePredicate(unified_items)

    def __eq__(self, other: "_AggregatePredicate") -> bool:
        all_keys = set(self._items.keys()) | other._items.keys()
        return all(self[k] == other[k] for k in all_keys)


ALWAYS_TRUE_AGG_PREDICATE = _AggregatePredicate({})


P = TypeVar("P")


class _Tracker(Generic[P]):
    def __init__(self):
        self.dirty = False
        self._predicates: dict[str, P] = dict()

    def __getitem__(self, var: Var) -> P:
        return self._predicates[var.name]

    def update(self, var: Var, pred: P):
        old_pred = self._predicates.get(var.name)
        if old_pred is None:
            new_pred = pred
        else:
            new_pred = old_pred.unify(pred)
            if new_pred == old_pred:
                return
        self.dirty = True
        self._predicates[var.name] = new_pred

    def propagate(self, src: Var, dst: Var):
        self.update(dst, self[src])

    def finalize(self) -> dict[str, P]:
        return self._predicates


@dataclass
class _State:
    tracker: _Tracker[DataPredicate]
    list_array_tracker: _Tracker[_AggregatePredicate]

    def propagate(self, src: Var, dst: Var):
        self.tracker.propagate(src, dst)
        self.list_array_tracker.propagate(src, dst)

    def set_always_true(self, var: Var):
        self.tracker.update(var, ALWAYS_TRUE_PREDICATE)
        self.list_array_tracker.update(var, ALWAYS_TRUE_AGG_PREDICATE)

    @property
    def dirty(self):
        return self.tracker.dirty or self.list_array_tracker.dirty

    def reset_dirty(self):
        self.tracker.dirty = False
        self.list_array_tracker.dirty = False


def _get_divisibility_for_binary_op(op: RawBinaryArithmeticOperation,
                                    tracker: _Tracker) -> int:
    result_dtype = get_dtype(op.result_var.get_type())
    if is_integral(result_dtype):
        x_div = tracker[op.lhs].div_by
        y_div = tracker[op.rhs].div_by
        if op.fn in ('add', 'sub'):
            return gcd(x_div, y_div)
        elif op.fn == 'mul':
            return x_div * y_div
    return 1


def _get_divisibility_for_unary_op(op: Unary, tracker: _Tracker) -> int:
    result_dtype = get_dtype(op.result_var.get_type())
    if is_integral(result_dtype):
        x_div = tracker[op.operand].div_by
        if op.fn in ('abs', 'neg'):
            return x_div
    return 1


def _analyze_aliases_in_block(block: Block,
                              state: _State,
                              innermost_loop: Loop | None,
                              innermost_branch: IfElse | TileReduce | TileScan | None):
    for op in block.operations:
        if isinstance(op, Assign):
            state.propagate(op.value, op.result_var)
        elif isinstance(op, AssumeDivBy):
            new_pred = state.tracker[op.x].replace(div_by=op.divisor)
            state.tracker.update(op.result_var, new_pred)
            state.list_array_tracker.propagate(op.x, op.result_var)
        elif isinstance(op, GetArrayListItem):
            # TODO: more granular array list get item alias analysis
            # Propagate to the base pointer of the array
            list_val = op.x.get_aggregate()
            assert isinstance(list_val, ListValue)
            arr_predicates = state.list_array_tracker[list_val.base_ptr]
            for i, var in enumerate(op.result_vars):
                state.tracker.update(var, arr_predicates[i])
                state.list_array_tracker.update(var, ALWAYS_TRUE_AGG_PREDICATE)
        elif isinstance(op, PointerOffset):
            # Update divby on pointer We only update divby on a pointer of 0d,
            # (Array.slice) Divby of a block of pointers (like ptr + arange for
            # ct.gather) are handled by tileiras.
            ptr_ty = op.pointer.get_type()
            assert isinstance(ptr_ty.dtype, PointerTy)
            if all(x == 1 for x in ptr_ty.shape):
                bitwidth = ptr_ty.dtype.pointee_type.dtype.bitwidth
                BYTE_BITWIDTH = 8
                if bitwidth % BYTE_BITWIDTH == 0:
                    offset_divby = state.tracker[op.offset].div_by * (bitwidth // BYTE_BITWIDTH)
                else:
                    offset_divby = 1
                ptr_divby = state.tracker[op.pointer].div_by
                new_divby = gcd(ptr_divby, offset_divby)
            else:
                new_divby = 1
            pred = state.tracker[op.pointer].replace(div_by=new_divby)
            state.tracker.update(op.result_var, pred)
            state.list_array_tracker.update(op.result_var, ALWAYS_TRUE_AGG_PREDICATE)
        elif isinstance(op, TileBroadcast | TileReshape | AssumeBounded):
            # Needed for tiles of pointers produced by gather/scatter
            state.propagate(op.x, op.result_var)
        elif isinstance(op, TileAsType):
            orig_dtype = get_dtype(op.x.get_type())
            result_dtype = get_dtype(op.result_var.get_type())
            div_by = state.tracker[op.x].div_by
            if is_integral(orig_dtype) and is_integral(result_dtype):
                # Truncate to int dtype is the same as mod 2^bitwidth
                div_by = gcd(div_by, 1 << result_dtype.bitwidth)
            else:
                div_by = 1
            pred = state.tracker[op.x].replace(div_by=div_by)
            state.tracker.update(op.result_var, pred)
            state.list_array_tracker.update(op.result_var, ALWAYS_TRUE_AGG_PREDICATE)
        elif isinstance(op, TypedConst):
            if isinstance(op.value, int):
                div_by = abs(op.value)
                state.tracker.update(op.result_var, DataPredicate(alias_set=ALIAS_UNIVERSE,
                                                                  div_by=div_by,
                                                                  may_alias_internally=True))
                state.list_array_tracker.update(op.result_var, ALWAYS_TRUE_AGG_PREDICATE)
            else:
                state.set_always_true(op.result_var)
        elif isinstance(op, RawBinaryArithmeticOperation):
            new_div_by = _get_divisibility_for_binary_op(op, state.tracker)
            pred = ALWAYS_TRUE_PREDICATE.replace(div_by=new_div_by)
            state.tracker.update(op.result_var, pred)
            state.list_array_tracker.update(op.result_var, ALWAYS_TRUE_AGG_PREDICATE)
        elif isinstance(op, Unary):
            new_div_by = _get_divisibility_for_unary_op(op, state.tracker)
            pred = ALWAYS_TRUE_PREDICATE.replace(div_by=new_div_by)
            state.tracker.update(op.result_var, pred)
            state.list_array_tracker.update(op.result_var, ALWAYS_TRUE_AGG_PREDICATE)
        elif isinstance(op, Loop):
            if op.is_for_loop:
                state.set_always_true(op.induction_var)

            for init, body, result in zip(op.initial_values, op.body_vars, op.result_vars,
                                          strict=True):
                # Loop initial values flow into body values.
                state.propagate(init, body)

                # `For` loop initial values can flow into result values if
                # loop runs for 0 iteration.
                if op.is_for_loop:
                    state.propagate(init, result)

            _analyze_aliases_in_block(op.body, state, op, None)
        elif isinstance(op, Continue):
            for next, body, result in zip(op.values, innermost_loop.body_vars,
                                          innermost_loop.result_vars, strict=True):
                # Loop next values can flow into body values
                state.propagate(next, body)

                # `For` loop next values can flow into result values when
                # the iterator is exhausted.
                if innermost_loop.is_for_loop:
                    state.propagate(next, result)

        elif isinstance(op, Break):
            for output, result in zip(op.values, innermost_loop.result_vars, strict=True):
                state.propagate(output, result)

        elif isinstance(op, IfElse):
            _analyze_aliases_in_block(op.then_block, state, innermost_loop, op)

            _analyze_aliases_in_block(op.else_block, state, innermost_loop, op)

        elif isinstance(op, EndBranch):
            for output, result in zip(op.outputs, innermost_branch.result_vars, strict=True):
                state.propagate(output, result)

        elif isinstance(op, TileReduce | TileScan):
            for v in op.body.params:
                state.set_always_true(v)
            _analyze_aliases_in_block(op.body, state, None, op)

        else:
            assert len(op.nested_blocks) == 0
            for v in op.result_vars:
                state.set_always_true(v)
