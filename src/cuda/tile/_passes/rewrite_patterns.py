# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Any, Dict, Sequence, List, Mapping, Set

from cuda.tile import _datatype as datatype
from cuda.tile._exception import Loc
from cuda.tile._ir.ir import Operation, Var, Block, IRContext
from cuda.tile._ir.ops import RawBinaryArithmeticOperation, FusedMulAddOperation, Unary
from cuda.tile._ir.ops_utils import get_dtype, get_default_rounding_mode
from cuda.tile._ir.type import Type


class NoMatch(Exception):
    pass


@dataclass
class Rewrite:
    to_remove: Sequence[Operation]
    to_add: Sequence[Operation]


class MatchContext:
    def __init__(self, ir_ctx: IRContext) -> None:
        self._rewrites: List[Rewrite] = []
        self._matches = [dict() for _ in _patterns]
        self._ir_ctx = ir_ctx

    # FIXME: remove this after moving operands to attributes
    @property
    def _constants(self):
        return self._ir_ctx.constants

    def typeof(self, var: Var) -> Type:
        return var.get_type()

    def set_type(self, var: Var, ty: Type):
        assert var.name not in self._ir_ctx.typemap
        self._ir_ctx.typemap[var.name] = ty

    def get_match(self, var: Var, pattern: "Pattern", default=None):
        return self._matches[pattern.pattern_id].get(var.name, default)

    def add_rewrite(self, to_remove: Sequence[Operation], to_add: Sequence[Operation]):
        self._rewrites.append(Rewrite(to_remove, to_add))

    def make_temp_var(self, loc: Loc) -> Var:
        return self._ir_ctx.make_temp(loc)


Predicate = Callable[[Operation, MatchContext], Any]


@dataclass
class Pattern:
    pattern_id: int
    op_class: type
    predicate: Predicate


_patterns = []
_patterns_by_op_class: Dict[type, List[Pattern]] = dict()


def pattern(op_class) -> Callable[[Predicate], Pattern]:
    def decorate(predicate) -> Pattern:
        pattern_id = len(_patterns)
        pat = Pattern(pattern_id, op_class, predicate)
        _patterns.append(pat)
        if op_class not in _patterns_by_op_class:
            _patterns_by_op_class[op_class] = []
        _patterns_by_op_class[op_class].append(pat)
        return pat

    return decorate


@pattern(RawBinaryArithmeticOperation)
def match_float_mul(op: RawBinaryArithmeticOperation,
                    ctx: MatchContext) -> RawBinaryArithmeticOperation:
    if op.fn != "mul":
        raise NoMatch("not a mul binop")
    if not datatype.is_float(get_dtype(ctx.typeof(op.result_var))):
        raise NoMatch("not a float mul")
    return op


@pattern(RawBinaryArithmeticOperation)
def fuse_mul_addsub(op: RawBinaryArithmeticOperation, ctx: MatchContext):
    if op.fn not in ("add", "sub"):
        raise NoMatch("not an add/sub binop")
    if (mul_op := ctx.get_match(op.lhs, match_float_mul)) is not None:
        acc = op.rhs
    elif op.fn == "add" and (mul_op := ctx.get_match(op.rhs, match_float_mul)) is not None:
        acc = op.lhs
    else:
        raise NoMatch("no float mul operand")

    rm = op.rounding_mode or get_default_rounding_mode()
    rm2 = mul_op.rounding_mode or get_default_rounding_mode()
    if rm != rm2:
        raise NoMatch("rounding mode mismatch")

    ftz = op.flush_to_zero
    ftz2 = mul_op.flush_to_zero
    if ftz != ftz2:
        raise NoMatch("flush-to-zero mismatch")

    # FIXME: fuse location
    new_ops = []
    if op.fn == "sub":
        negated_acc = ctx.make_temp_var(op.loc)
        ctx.set_type(negated_acc, ctx.typeof(acc))
        new_ops.append(Unary(fn="neg", operand=acc, rounding_mode=None, flush_to_zero=False,
                             result_vars=(negated_acc,), loc=op.loc))
        acc = negated_acc

    new_ops.append(FusedMulAddOperation(lhs=mul_op.lhs, rhs=mul_op.rhs, acc=acc,
                                        rounding_mode=rm, flush_to_zero=ftz,
                                        result_vars=(op.result_var,), loc=op.loc))
    ctx.add_rewrite((mul_op, op), new_ops)


def rewrite_patterns(root_block: Block):
    ctx = MatchContext(root_block.ctx)
    uses = defaultdict(list)
    for op in root_block.traverse():
        for pat in _patterns_by_op_class.get(type(op), ()):
            try:
                match_res = pat.predicate(op, ctx)
                ctx._matches[pat.pattern_id][op.result_var.name] = match_res
            except NoMatch:
                pass
        for var in op.all_inputs():
            uses[var.name].append(op)

    replacements = dict()
    rewritten_ops = set()
    for r in ctx._rewrites:
        if any(op in rewritten_ops for op in r.to_remove):
            # Operation already rewritten -- can't rewrite
            continue

        new_results = set(v.name for op in r.to_add for v in op.result_vars)
        old_results = set(v.name for op in r.to_remove for v in op.result_vars)
        deleted_results = old_results - new_results
        if any(op not in r.to_remove for name in deleted_results for op in uses[name]):
            # External use -- can't rewrite
            continue

        new_inputs = set(v.name for op in r.to_add for v in op.all_inputs())
        if deleted_results & new_inputs:
            # New operations use deleted results -- can't rewrite
            continue

        # For now, we insert the new operations at the location of the last matched op.
        # This is not always correct for maintaining topological sorting, in case if matches
        # have multiple outputs. However, currently we only care about rewriting subgraphs
        # with a single result, so this is sufficient.
        replacements[r.to_remove[-1]] = r.to_add
        rewritten_ops.update(r.to_remove)

    _apply_rewrites(root_block, rewritten_ops, replacements)


def _apply_rewrites(block: Block,
                    rewritten_ops: Set[Operation],
                    replacements: Mapping[Operation, Sequence[Operation]]):
    new_block = block.empty_like_self()
    for op in block:
        for nb in op.nested_blocks:
            _apply_rewrites(nb, rewritten_ops, replacements)

        new_ops = replacements.get(op)
        if new_ops is None:
            if op not in rewritten_ops:
                new_block.append(op)
        else:
            new_block.extend(new_ops)
    block[:] = new_block.detach_all()
