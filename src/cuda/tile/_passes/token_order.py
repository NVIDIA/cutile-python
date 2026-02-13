# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import dataclasses
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Tuple, Dict, Set, Optional

from cuda.tile._ir.type import TokenTy
from cuda.tile._memory_model import MemoryOrder
from cuda.tile._exception import Loc, TileInternalError
from cuda.tile._ir.ir import Block, IRContext, Var, Operation, MemoryEffect
from cuda.tile._ir.ops import (
    Break, Continue, EndBranch, IfElse,
    JoinTokens, Loop, MakeToken,
    TileAtomicCAS, TileAtomicCASTokenOrdered,
    TileAtomicRMW, TileAtomicRMWTokenOrdered, LoadPointer, LoadPointerTokenOrdered,
    TileLoad, TileLoadTokenOrdered, StorePointer, StorePointerTokenOrdered,
    TileStore, TileStoreTokenOrdered, TileAssert, TilePrintf,
)
from cuda.tile._ir.ops_utils import memory_order_has_acquire, memory_order_has_release
from cuda.tile._passes.alias_analysis import ALIAS_UNIVERSE, AliasResult, AliasSet


class MemoryEffects:
    def __init__(self,
                 array_mem_effects: OrderedDict[AliasSet, MemoryEffect],
                 has_acquire_order: bool):
        self._array_mem_effects = MappingProxyType(array_mem_effects)
        self._has_acquire_order = has_acquire_order

    @property
    def has_acquire_order(self) -> bool:
        return self._has_acquire_order

    def items(self):
        return self._array_mem_effects.items()

    def __getitem__(self, alias_name: AliasSet):
        return self._array_mem_effects.get(alias_name, MemoryEffect.NONE)

    def __or__(self, other) -> "MemoryEffects":
        if not isinstance(other, MemoryEffects):
            return NotImplemented

        union_effects = OrderedDict(self._array_mem_effects)
        for a in other._array_mem_effects.keys():
            union_effects[a] = max(self[a], other[a])
        return MemoryEffects(union_effects, self.has_acquire_order | other.has_acquire_order)


EMPTY_MEMORY_EFFECTS = MemoryEffects(OrderedDict(), False)


class AcquireTokenKeyClass:
    def __repr__(self):
        return "ACQUIRE_TOKEN_KEY"


ACQUIRE_TOKEN_KEY = AcquireTokenKeyClass()


class TokenRole(Enum):
    LAST_OP = auto()
    LAST_STORE = auto()


@dataclass(frozen=True)
class AliasTokenKey:
    alias_set: AliasSet
    role: TokenRole


TokenKey = AliasTokenKey | AcquireTokenKeyClass


@dataclass(frozen=True)
class IfElseInfo:
    ifelse_mem_effects: MemoryEffects


@dataclass(frozen=True)
class InnermostLoopInfo:
    loop_mem_effects: MemoryEffects
    # TODO: remove once parallel loop store no longer depends on these two
    loop_parallel_stores: Set[Operation]
    parent_token_map: Dict[TokenKey, Var]


@dataclass(frozen=True)
class VarInfo:
    root_var: Dict[str, str]


@dataclass(frozen=True)
class TokenOrderContext:
    alias_result: AliasResult
    block_memory_effects: Dict[Block, MemoryEffects]


_TOKEN_ORDERED_OP_MAP = {
    TileLoad: TileLoadTokenOrdered,
    TileStore: TileStoreTokenOrdered,
    LoadPointer: LoadPointerTokenOrdered,
    StorePointer: StorePointerTokenOrdered,
    TileAtomicCAS: TileAtomicCASTokenOrdered,
    TileAtomicRMW: TileAtomicRMWTokenOrdered,
}


def token_order_pass(root_block: Block, alias_result: AliasResult):
    block_memory_effects = {}
    _get_block_memory_effects(root_block, alias_result, block_memory_effects)
    context = TokenOrderContext(alias_result, block_memory_effects)

    root_tok = _make_token_var(root_block.ctx, root_block.loc)
    token_map = defaultdict(lambda: root_tok)
    _to_token_order_in_block(root_block, context, token_map)
    # Ensures Operation.parent_block is correctly set
    root_block[:0] = (MakeToken(result_vars=(root_tok,), loc=root_block.loc),)


def _get_input_var(op: Operation):
    return op.array if "array" in op.operands else op.pointer


def _get_block_memory_effects(block: Block,
                              alias_result: AliasResult,
                              block_memory_effects: Dict[Block, MemoryEffects]):

    def get_memory_effects(cur_op):
        effect = cur_op.memory_effect
        if effect == MemoryEffect.NONE or isinstance(cur_op, TileAssert | TilePrintf):
            return EMPTY_MEMORY_EFFECTS

        has_acquire_order = False
        if isinstance(cur_op, (TileAtomicCAS, TileAtomicRMW)):
            has_acquire_order = memory_order_has_acquire(cur_op.memory_order)

        return MemoryEffects({alias_result[_get_input_var(cur_op).name]: effect}, has_acquire_order)

    blk_mem_effects = EMPTY_MEMORY_EFFECTS
    for op in block.operations:
        blk_mem_effects = blk_mem_effects | get_memory_effects(op)
        # Include nested blocks' memory effects in parent block
        for nested_block in op.nested_blocks:
            _get_block_memory_effects(nested_block, alias_result, block_memory_effects)
            blk_mem_effects = blk_mem_effects | block_memory_effects[nested_block]

    block_memory_effects[block] = blk_mem_effects


def _to_token_order_in_block(block: Block,
                             context: TokenOrderContext,
                             token_map: Dict[TokenKey, Var],
                             *,
                             innermost_loop_info: Optional[InnermostLoopInfo] = None,
                             ifelse_info: Optional[IfElseInfo] = None,):
    operations = []

    # Convert the old ops to token ordered ops,
    # including control flow ops containing token ordered ops
    for op in block.operations:
        if isinstance(op, (TileLoad, LoadPointer)):
            alias_set = context.alias_result[_get_input_var(op).name]
            last_op_key = _last_op_key(alias_set)
            last_store_key = _last_store_key(alias_set)

            input_tok, maybe_input_tok_join_op = _get_input_token(last_store_key, op,
                                                                  token_map, None)
            if maybe_input_tok_join_op:
                operations.append(maybe_input_tok_join_op)

            # Convert
            result_tok = _make_token_var(block.ctx, op.loc)
            tko_load_op = _to_token_ordered_mem_op(op, input_tok, result_tok)
            operations.append(tko_load_op)

            # Eagerly join with last_op_token
            new_last_op_tok = _make_token_var(block.ctx, op.loc)
            join_op = JoinTokens(tokens=(token_map[last_op_key], result_tok),
                                 result_vars=(new_last_op_tok,), loc=op.loc)
            operations.append(join_op)

            token_map[last_op_key] = new_last_op_tok

        elif isinstance(op, (TileStore, StorePointer)):
            # Try to parallelize the store in the innermost loop if possible
            if (
                isinstance(op, TileStore)
                and (parallel_store_ops := _try_loop_parallel_store(op, context.alias_result,
                                                                    token_map, innermost_loop_info))
            ):
                operations.extend(parallel_store_ops)
                continue

            alias_set = context.alias_result[_get_input_var(op).name]
            last_op_key = _last_op_key(alias_set)
            last_store_key = _last_store_key(alias_set)

            input_tok, maybe_input_tok_join_op = _get_input_token(last_op_key, op, token_map,
                                                                  None)
            if maybe_input_tok_join_op:
                operations.append(maybe_input_tok_join_op)

            # Convert
            result_tok = _make_token_var(block.ctx, op.loc)
            tko_store_op = _to_token_ordered_mem_op(op, input_tok, result_tok)
            operations.append(tko_store_op)

            token_map[last_op_key] = result_tok
            token_map[last_store_key] = result_tok

        elif isinstance(op, (TileAtomicCAS, TileAtomicRMW)):
            alias_set = context.alias_result[_get_input_var(op).name]
            last_op_key = _last_op_key(alias_set)
            last_store_key = _last_store_key(alias_set)

            input_tok, maybe_input_tok_join_op = _get_input_token(last_op_key, op, token_map,
                                                                  op.memory_order)
            if maybe_input_tok_join_op:
                operations.append(maybe_input_tok_join_op)

            result_tok = _make_token_var(block.ctx, op.loc)
            tko_atomic_op = _to_token_ordered_mem_op(op, input_tok, result_tok)
            operations.append(tko_atomic_op)

            token_map[last_op_key] = result_tok
            token_map[last_store_key] = result_tok

            if memory_order_has_acquire(op.memory_order):
                token_map[ACQUIRE_TOKEN_KEY] = result_tok

        elif isinstance(op, Loop):
            body_mem_effects = context.block_memory_effects[op.body]

            new_initial_values = list(op.initial_values)
            new_body_params = list(op.body.params)
            new_result_vars = list(op.result_vars)

            def append_new_carried_var(init_var: Var):
                new_initial_values.append(init_var)
                body_var = _make_token_var(block.ctx, op.loc)
                new_body_params.append(body_var)
                res_var = _make_token_var(block.ctx, op.loc)
                new_result_vars.append(res_var)
                return body_var, res_var

            result_token_map = token_map.copy()
            body_token_map = token_map.copy()
            for alias_set, effect in body_mem_effects.items():
                last_op_key = _last_op_key(alias_set)
                last_store_key = _last_store_key(alias_set)

                if effect == MemoryEffect.NONE:
                    continue
                elif effect == MemoryEffect.LOAD:
                    body_token_map[last_op_key], result_token_map[last_op_key] = \
                        append_new_carried_var(token_map[last_op_key])
                elif effect == MemoryEffect.STORE:
                    body_token_map[last_op_key], result_token_map[last_op_key] = \
                        append_new_carried_var(token_map[last_op_key])
                    body_token_map[last_store_key], result_token_map[last_store_key] = \
                        append_new_carried_var(token_map[last_store_key])
                else:
                    raise TileInternalError(f"Unexpected memory effect: {effect}")

            if body_mem_effects.has_acquire_order:
                body_token_map[ACQUIRE_TOKEN_KEY], result_token_map[ACQUIRE_TOKEN_KEY] = \
                    append_new_carried_var(token_map[ACQUIRE_TOKEN_KEY])

            # Get parallel stores in the body block
            parallel_stores = _get_parallel_stores(op, context)

            _to_token_order_in_block(op.body, context, body_token_map,
                                     ifelse_info=None,
                                     innermost_loop_info=InnermostLoopInfo(body_mem_effects,
                                                                           parallel_stores,
                                                                           token_map))

            op.body.params = tuple(new_body_params)
            new_loop_op = dataclasses.replace(op, initial_values=tuple(new_initial_values),
                                              result_vars=tuple(new_result_vars))
            operations.append(new_loop_op)

            token_map = result_token_map

        elif isinstance(op, Continue):
            tokens = _get_cf_exit_tokens(innermost_loop_info.loop_mem_effects, token_map)

            new_continue_op = dataclasses.replace(op, values=tuple(op.values) + tokens)
            operations.append(new_continue_op)

        elif isinstance(op, Break):
            tokens = _get_cf_exit_tokens(innermost_loop_info.loop_mem_effects, token_map)

            new_break_op = dataclasses.replace(op, values=tuple(op.values) + tokens)
            operations.append(new_break_op)

        elif isinstance(op, IfElse):
            # Merge memory effects from both branches
            then_mem_effects = context.block_memory_effects[op.then_block]
            else_mem_effects = context.block_memory_effects[op.else_block]
            merged_mem_effects = then_mem_effects | else_mem_effects

            result_token_map = token_map.copy()
            new_result_vars = list(op.result_vars)

            def add_new_ifelse_result():
                x = _make_token_var(block.ctx, op.loc)
                new_result_vars.append(x)
                return x

            for alias_set, effect in merged_mem_effects.items():
                last_op_key = _last_op_key(alias_set)
                last_store_key = _last_store_key(alias_set)

                if effect == MemoryEffect.NONE:
                    continue
                elif effect == MemoryEffect.LOAD:
                    result_token_map[last_op_key] = add_new_ifelse_result()
                elif effect == MemoryEffect.STORE:
                    result_token_map[last_op_key] = add_new_ifelse_result()
                    result_token_map[last_store_key] = add_new_ifelse_result()
                else:
                    raise TileInternalError(f"Unexpected memory effect: {effect}")

            if merged_mem_effects.has_acquire_order:
                result_token_map[ACQUIRE_TOKEN_KEY] = add_new_ifelse_result()

            # Branch to then and else blocks
            for nested_block in op.nested_blocks:
                _to_token_order_in_block(nested_block, context, token_map.copy(),
                                         innermost_loop_info=innermost_loop_info,
                                         ifelse_info=IfElseInfo(merged_mem_effects))

            new_ifelse_op = dataclasses.replace(op, result_vars=tuple(new_result_vars))
            operations.append(new_ifelse_op)

            token_map = result_token_map

        elif isinstance(op, EndBranch):
            tokens = _get_cf_exit_tokens(ifelse_info.ifelse_mem_effects, token_map)

            new_end_branch_op = dataclasses.replace(op, outputs=tuple(op.outputs) + tokens)
            operations.append(new_end_branch_op)

        else:
            operations.append(op)

    block.operations = operations


def _last_op_key(alias_set: AliasSet):
    return AliasTokenKey(alias_set, TokenRole.LAST_OP)


def _last_store_key(alias_set: AliasSet):
    return AliasTokenKey(alias_set, TokenRole.LAST_STORE)


def _make_token_var(ir_ctx: IRContext, loc: Loc) -> Var:
    var = ir_ctx.make_var("$token", loc)
    var.set_type(TokenTy())
    return var


def _collect_join_tokens(token_key: TokenKey,
                         token_map: Dict[TokenKey, Var],
                         memory_order: MemoryOrder | None):

    def should_join(other_key):
        if other_key == ACQUIRE_TOKEN_KEY:
            return True

        assert isinstance(other_key, AliasTokenKey)

        mem_order_release_join = (memory_order is not None and
                                  memory_order_has_release(memory_order) and
                                  other_key.role == TokenRole.LAST_OP)
        alias_set_overlap_join = (other_key.role == token_key.role and
                                  (other_key.alias_set & token_key.alias_set))
        return mem_order_release_join or alias_set_overlap_join

    # Preserve the order of the tokens to join
    tokens_to_join = [token_map[token_key]]
    for other_key, other_tok in token_map.items():
        if not should_join(other_key):
            continue
        if other_tok not in tokens_to_join:
            tokens_to_join.append(other_tok)

    return tokens_to_join


def _get_input_token(token_key: TokenKey,
                     op: Operation,
                     token_map: Dict[TokenKey, Var],
                     memory_order: MemoryOrder | None) -> Tuple[Var, Operation | None]:
    tokens_to_join = _collect_join_tokens(token_key, token_map, memory_order)

    if len(tokens_to_join) == 1:
        return tokens_to_join[0], None

    ret_tok = _make_token_var(op.parent_block.ctx, op.loc)
    ret_op = JoinTokens(tokens=tuple(tokens_to_join), result_vars=(ret_tok,), loc=op.loc)
    return ret_tok, ret_op


def _to_token_ordered_mem_op(op, token: Var, result_token: Var) -> Operation:

    new_class = _TOKEN_ORDERED_OP_MAP[op.__class__]
    new_kwargs = dict(op.operands)
    new_kwargs.update(op.attributes)

    if new_class in (TileLoadTokenOrdered, LoadPointerTokenOrdered,
                     TileAtomicCASTokenOrdered, TileAtomicRMWTokenOrdered):
        new_kwargs["result_vars"] = (op.result_var, result_token)
    else:
        new_kwargs["result_vars"] = (result_token,)

    return new_class(token=token, loc=op.loc, **new_kwargs)


def _get_cf_exit_tokens(cf_mem_effects: MemoryEffects,
                        token_map: Dict[TokenKey, Var]) -> Tuple[Var, ...]:
    tokens = []
    for alias_set, effect in cf_mem_effects.items():
        last_op_key = _last_op_key(alias_set)
        last_store_key = _last_store_key(alias_set)

        if effect == MemoryEffect.NONE:
            continue
        if effect == MemoryEffect.LOAD:
            tokens.append(token_map[last_op_key])
        elif effect == MemoryEffect.STORE:
            tokens.extend((token_map[last_op_key], token_map[last_store_key]))

    if cf_mem_effects.has_acquire_order:
        tokens.append(token_map[ACQUIRE_TOKEN_KEY])

    return tuple(tokens)


# === LOOP PARALLEL STORE OPTIMIZATION ===


def _get_parallel_stores(
    loop_op: Loop,
    context: TokenOrderContext
) -> Set[Operation]:
    """
    A specific optimization for when there's only one TileStore
    on a given array in the for-loop body (including nested blocks),
    and the index doesn't overlap across iterations.
    We can parallelize the TileStore.

    Common in LayerNorm and RMSNorm patterns.
    """

    if not loop_op.is_for_loop:
        return set()

    # Skips this optimization if alias_set size > 1 is present in the loop body
    body_mem_effects = context.block_memory_effects[loop_op.body]
    if any(alias_set == ALIAS_UNIVERSE or len(alias_set) > 1
           for alias_set, _ in body_mem_effects.items()):
        return set()

    nested_mem_effects = _get_nested_mem_effects(loop_op, context.block_memory_effects)

    alias_set_to_mem_ops = defaultdict(list)
    for op in loop_op.body.operations:
        if isinstance(op, (TileLoad, StorePointer, LoadPointer, TileStore,
                           TileAtomicCAS, TileAtomicRMW)):
            alias_set_to_mem_ops[context.alias_result[_get_input_var(op).name]].append(op)

    tile_store_candidates = set()
    for alias_set, mem_ops in alias_set_to_mem_ops.items():
        if len(mem_ops) != 1:
            # More than 1 memory ops on the same array in loop
            continue
        elif not isinstance(mem_ops[0], TileStore):
            # The memory op on array is not TileStore
            continue
        elif nested_mem_effects[alias_set] != MemoryEffect.NONE:
            # The nested blocks / func calls have memory effects on array
            continue
        tile_store_candidates.add(mem_ops[0])

    # Filter in stores that have non-overlapping indices
    res = _filter_by_store_index(loop_op, tile_store_candidates)
    return res


def _get_nested_mem_effects(
    loop_op: Loop,
    block_memory_effects: Dict[Block, MemoryEffects]
) -> MemoryEffects:
    nested_blocks = [b for op in loop_op.body.operations for b in op.nested_blocks]
    nested_mem_effects = EMPTY_MEMORY_EFFECTS
    for b in nested_blocks:
        nested_mem_effects = nested_mem_effects | block_memory_effects[b]
    return nested_mem_effects


def _filter_by_store_index(loop_op: Loop,
                           tile_store_candidates: Set[Operation]) -> Set[Operation]:

    def is_idx_injective(idx_var: Var) -> bool:
        # TODO: allow more complex injective check: j = i * 2 + 3
        return loop_op.is_for_loop and idx_var.name == loop_op.induction_var.name

    return set(store_op for store_op in tile_store_candidates
               if _get_input_var(store_op).get_type().elements_disjoint
               and any(is_idx_injective(idx_var) for idx_var in store_op.index))


def _try_loop_parallel_store(
    store_op: TileStore,
    alias_result: AliasResult,
    token_map: Dict[TokenKey, Var],
    innermost_loop_info: Optional[InnermostLoopInfo]
) -> Optional[Tuple[Operation, ...] | Operation]:

    if (not innermost_loop_info or
            store_op not in innermost_loop_info.loop_parallel_stores):
        return None

    alias_set = alias_result[_get_input_var(store_op).name]
    last_op_key = _last_op_key(alias_set)
    last_store_key = _last_store_key(alias_set)

    # Convert to parallellized store
    parent_token_map = innermost_loop_info.parent_token_map
    before_loop_last_op_tok = parent_token_map[last_op_key]

    if (ACQUIRE_TOKEN_KEY in token_map and
            before_loop_last_op_tok is not token_map[ACQUIRE_TOKEN_KEY]):
        input_tok = _make_token_var(store_op.parent_block.ctx, store_op.loc)
        maybe_input_tok_join_op = JoinTokens(
            tokens=(before_loop_last_op_tok, token_map[ACQUIRE_TOKEN_KEY]),
            result_vars=(input_tok,), loc=store_op.loc)
    else:
        input_tok = before_loop_last_op_tok
        maybe_input_tok_join_op = None

    result_tok = _make_token_var(store_op.parent_block.ctx, store_op.loc)
    tko_store_op = _to_token_ordered_mem_op(store_op, input_tok, result_tok)

    # Eagerly join with loop_last_op_tok
    loop_last_op_tok = token_map[last_op_key]
    new_last_op_tok = _make_token_var(store_op.parent_block.ctx, store_op.loc)
    join_op = JoinTokens(tokens=(loop_last_op_tok, result_tok),
                         result_vars=(new_last_op_tok,), loc=store_op.loc)

    token_map[last_op_key] = new_last_op_tok
    token_map[last_store_key] = new_last_op_tok

    return ((maybe_input_tok_join_op,) if maybe_input_tok_join_op else ()) + (tko_store_op, join_op)
