# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum
import math
import operator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Union, Any, List, Callable, Iterator, Iterable

from typing_extensions import override

import cuda.tile._stub as ct
from cuda.tile import _datatype as datatype, TileValueError
from cuda.tile import RoundingMode, MemoryOrder, MemoryScope
from cuda.tile._mutex import tile_mutex
from cuda.tile._exception import TileTypeError, TileSyntaxError
from cuda.tile._ir.ir import (
    Operation, Var, Loc, Block,
    terminator, add_operation, TypedOperation, Builder,
    has_multiple_results, nested_block, PhiState, LoopVarState,
    TupleValue, make_aggregate, RangeValue, BoundMethodValue, ArrayValue, ConstantState,
    ListValue, ClosureValue, MemoryEffect, memory_effect
)
from cuda.tile._ir.load_store_impl import (
    check_load_store_hints,
    tile_load_generate_bytecode, tile_store_generate_bytecode,
    load_pointer_lowering, store_pointer_lowering,
)
from . import hir
from .hir import ResolvedName
from .op_impl import (
    impl, require_constant_int, require_constant_int_tuple,
    require_signed_integer_scalar_or_0d_tile_type,
    require_tile_type, normalize_axis, require_dtype_spec,
    require_tile_or_scalar_type, require_constant_bool, require_optional_constant_enum,
    require_constant_str, require_array_type, require_tuple_type, require_constant_slice,
    require_list_type, require_scalar_or_0d_tile_type,
    require_index_or_index_tuple_type, require_constant_shape, require_constant_axis_order,
    require_constant_enum, require_optional_constant_int, require_optional_constant_bool,
    require_optional_constant_str, PrintfValidator, require_tile_or_scalar_maybe_loose_type,
    require_scalar_or_0d_tile_maybe_loose_type, require_bool, require_optional_range_type,
    require_tile_or_tile_tuple_type, require_constant_scalar_tuple, require_constant_scalar,
    require_callable_type)
from .ops_utils import (
    BINOP_REGISTRY, UNARYOP_REGISTRY,
    check_rd_and_ftz, PaddingMode,
    rounding_mode_to_bytecode, get_dtype, change_dtype, memory_order_to_bytecode,
    memory_scope_to_bytecode, broadcast_shapes2, is_shape_broadcastable_to, BroadcastError,
    promote_types, promote_dtypes, check_implicit_cast
)
from .scope import Scope, JumpInfo, ControlFlowInfo
from .typing_support import (
    BYTE_BITWIDTH, typeof_pyval, dtype_registry, loose_type_of_pyval, get_constant_value
)
from .type import (
    TupleTy, TileTy, NoneType, BoundMethodTy, SizeTy, ArrayTy,
    ListTy, make_tile_ty, SliceType, DTypeConstructor, RangeIterType, Type,
    NONE, ModuleTy, TypeTy, LooselyTypedScalar, DTypeSpec, StringTy, InvalidType,
    array_size_type, ClosureTy, LiveCapturedScope
)
from cuda.tile._datatype import (
    DType, is_integral, is_float, is_signed, is_boolean, is_restricted_float,
)
from cuda.tile._ir2bytecode import (
    lower_scan, BytecodeContext, typeid,
    generate_bytecode_for_block, convert_dtype, get_list_item_repr_size_in_words,
    get_list_partition_view_tile_size, tensor_view_typeid, tensor_view_typeid_for_list
)
import cuda.tile._bytecode as bc
from .._debug import CUDA_TILE_TESTING_DISABLE_DIV


# ================================================
# Control flow operations
# ================================================


@has_multiple_results
class Loop(TypedOperation):
    def __init__(
        self,
        start: Optional[Var],
        stop: Optional[Var],
        step: Optional[Var],
        initial_values: tuple[Var, ...],
        result_vars: tuple[Var, ...],
        body: Block,
        loc: Loc
    ):
        assert (start is None) == (stop is None) == (step is None)
        super().__init__(
            "loop",
            operands={"start": start, "stop": stop, "step": step, "initial_values": initial_values},
            result_vars=list(result_vars),
            nested_blocks=[body],
            loc=loc,
        )

    @property
    def body(self):
        return self.nested_blocks[0]

    @property
    def is_for_loop(self) -> bool:
        return self.start is not None

    @property
    def induction_var(self):
        assert self.is_for_loop
        return self.nested_blocks[0].params[0]

    @property
    def body_vars(self) -> tuple[Var, ...]:
        return self.body.params[1:] if self.is_for_loop else self.body.params

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, ...]:
        types = tuple(x.get_type() for x in self.body_vars)
        initial_values = [ctx.get_value_allow_undefined(input_var, ty)
                          for input_var, ty in zip(self.initial_values, types, strict=True)]
        result_type_ids = [typeid(ctx.type_table, ty) for ty in types]

        if self.is_for_loop:
            start, stop, step = (ctx.get_value(x) for x in (self.start, self.stop, self.step))
            nested_builder = bc.encode_ForOp(ctx.builder, result_type_ids, start, stop, step,
                                             initial_values, unsignedCmp=False)
            induction_var_type_id = ctx.typeid_of(self.induction_var)
            block_arg_type_ids = (induction_var_type_id, *result_type_ids)
        else:
            nested_builder = bc.encode_LoopOp(ctx.builder, result_type_ids, initial_values)
            block_arg_type_ids = result_type_ids

        with nested_builder.new_block(block_arg_type_ids) as block_args, ctx.enter_loop(self):
            block_args = iter(block_args)
            if self.is_for_loop:
                ctx.set_value(self.induction_var, next(block_args))
            for var, value in zip(self.body_vars, block_args, strict=True):
                ctx.set_value(var, value)
            generate_bytecode_for_block(ctx, self.body)

        return nested_builder.done()

    @override
    def _to_string_block_prefixes(self) -> List[str]:
        return ["do"]

    @override
    def _to_string_rhs(self) -> str:
        def format_var(var):
            ty = var.try_get_type()
            if ty is None:
                return var.name
            return f"{var.name}: {ty}"

        if self.is_for_loop:
            body_vars = self.body.params[1:]
            header_str = (f"for {self.body.params[0].name}"
                          f" in range({self.start.name}, {self.stop.name}, {self.step.name})")
        else:
            body_vars = self.body.params
            header_str = "loop"

        carried_vars_str = ", ".join(f"{format_var(b)} = {i.name}"
                                     for b, i in zip(body_vars, self.initial_values))
        return f"{header_str} (with {carried_vars_str})"


@impl(hir.loop)
async def loop_impl(body: hir.Block, iterable: Var):
    from .._passes.hir2ir import dispatch_hir_block

    scope = Scope.get_current()
    range_ty = require_optional_range_type(iterable)
    if range_ty is None and body.jump == hir.Jump.BREAK and not _have_nested_jump(body.calls):
        # In ast2hir, we create a loop around the function body in order to support early returns.
        # But if there is no early return, we can remove the loop. In this case, the loop will only
        # have a "break" at the end of the loop body, and no other break/continue statements.
        info = ControlFlowInfo((), flatten=True)
        with scope.change_loop_info(info):
            await dispatch_hir_block(body)
        return

    builder = Builder.get_current()
    stored_locals = tuple(sorted(scope.get_local_index(name) for name in body.stored_names))
    var_states = tuple(LoopVarState(PhiState(initial_constant_state=ConstantState.NONCONSTANT),
                                    PhiState())
                       for _ in stored_locals)
    initial_values = tuple(scope.local.get(index, builder.loc) for index in stored_locals)
    body_params = []

    # Logic specific to `for` loops:
    if range_ty is not None:
        # A `for` loop may have 0 iterations, so initial values need to be propagated
        # to the loop's results.
        for initial_var, state in zip(initial_values, var_states, strict=True):
            state.result_phi.propagate(initial_var)
        # Create an induction variable
        induction_var = builder.ir_ctx.make_temp(builder.loc)
        induction_var.set_type(range_ty.dtype)
        scope.hir2ir_varmap[body.params[0].id] = induction_var
        body_params.append(induction_var)

    # Process the loop body
    loop_info = ControlFlowInfo(stored_locals)
    body_loc = body.loc.with_call_site(scope.call_site)
    with nested_block(body_loc) as new_body, scope.change_loop_info(loop_info), \
            scope.local.enter_branch():
        # Define body variables. Not all of them will eventually be kept,
        # so we don't set the block parameters yet.
        body_vars = []
        for local_idx, initial_var, state in zip(
                stored_locals, initial_values, var_states, strict=True):
            state.body_phi.propagate(initial_var, allow_loose_typing=False)
            body_var = scope.local.redefine(local_idx, state.body_phi.last_loc)
            body_var.set_type(state.body_phi.ty)
            body_vars.append(body_var)

        flat_body_vars = flatten_block_parameters(body_vars)

        # Dispatch the body (hir.Block) to populate the new_body (ir.Block) with Operations
        await dispatch_hir_block(body)

    # Propagate type information from Continue/Break to body/result phis
    for jump_info in loop_info.jumps:
        is_continue = isinstance(jump_info.jump_op, Continue)
        assert is_continue or isinstance(jump_info.jump_op, Break)
        for output, state in zip(jump_info.outputs, var_states, strict=True):
            if is_continue:
                state.body_phi.propagate(output, fail_eagerly=True)
            if range_ty is not None or not is_continue:
                state.result_phi.propagate(output)

    # Determine the final loop variable types and filter out invalid variables
    mask = []
    for i, (body_var, state) in enumerate(zip(body_vars, var_states, strict=True)):
        was_valid = not isinstance(body_var.get_type_allow_invalid(), InvalidType)
        state.finalize_loopvar_type(body_var)
        ty = body_var.get_type_allow_invalid()
        is_valid = not isinstance(ty, InvalidType)
        mask.append(is_valid)
        if not is_valid:
            body_var.set_undefined()
        elif not was_valid and ty.is_aggregate():
            # The initial variable is invalid but the loop variable is preserved,
            # and the loop variable is aggregate. In this case, `flat_body_vars[i]`
            # will contain a single variable (previously of InvalidType,
            # and now of an aggregate type). Thus, we need to update it with
            # the according number of flattened undefined variables.
            assert len(flat_body_vars[i]) == 1
            undefined_items = expand_aggregate_var(body_var)
            flat_body_vars[i] = undefined_items
            # Create a fake aggregate value so that flatten_aggregates() doesn't fail
            # when we update the Continue/Break statements later.
            body_var.set_aggregate(TupleValue(undefined_items))

    # Set the block's parameters
    all_flattened_body_vars = sum((flattened for flattened, is_valid
                                   in zip(flat_body_vars, mask, strict=True) if is_valid), ())
    body_params.extend(all_flattened_body_vars)
    new_body.params = tuple(body_params)
    valid_var_types = tuple(v.get_type() for v, is_valid in zip(body_vars, mask, strict=True)
                            if is_valid)

    # Update Continue/Break statements
    for jump_info in loop_info.jumps:
        values = tuple(out
                       for out, is_valid in zip(jump_info.outputs, mask, strict=True)
                       if is_valid)
        flat_values = flatten_aggregates(values, valid_var_types)
        assert len(flat_values) == len(all_flattened_body_vars)
        jump_info.jump_op.update_operand("values", flat_values)

    # Create the loop Operation
    valid_initial_values = tuple(v for v, is_valid in zip(initial_values, mask, strict=True)
                                 if is_valid)
    flat_initial_values = flatten_aggregates(valid_initial_values, valid_var_types)
    assert len(flat_initial_values) == len(all_flattened_body_vars)

    flat_var_types = flatten_aggregate_types(valid_var_types)
    if range_ty is None:
        start = stop = step = None
    else:
        range_val = iterable.get_aggregate()
        assert isinstance(range_val, RangeValue)
        start, stop, step = range_val.start, range_val.stop, range_val.step
    flat_result_vars = add_operation(Loop, flat_var_types,
                                     start=start,
                                     stop=stop,
                                     step=step,
                                     initial_values=flat_initial_values,
                                     body=new_body)

    result_types = tuple(s.result_phi.ty for s, is_valid in zip(var_states, mask, strict=True)
                         if is_valid)
    result_vars = unflatten_aggregates(flat_result_vars, valid_var_types, result_types)

    # Finalize the scope & type information for valid result variables
    valid_var_states = tuple(s for s, is_valid in zip(var_states, mask, strict=True)
                             if is_valid)
    valid_stored_locals = tuple(local_idx
                                for local_idx, is_valid in zip(stored_locals, mask, strict=True)
                                if is_valid)
    for res, state, local_idx in zip(result_vars, valid_var_states, valid_stored_locals,
                                     strict=True):
        state.result_phi.finalize_constant_and_loose_type(res)
        store_var(local_idx, res, state.result_phi.last_loc)

    # For any names that are stored within the loop body but have an invalid result type,
    # we update the scope to point to an undefined variable of this invalid type, so that
    # using that variable afterwards would result in a type error.
    for body_var, state, local_idx, is_valid in zip(body_vars, var_states, stored_locals, mask,
                                                    strict=True):
        if not is_valid:
            store_undefined(local_idx, body_var.get_type_allow_invalid(), state.result_phi.last_loc)

    # Do this check at the end because this may be an automatically inserted loop
    # around the helper function's body.
    if builder.reduction_body:
        raise TileSyntaxError("Loops inside reduction body are not supported")


def _have_nested_jump(calls: Sequence[hir.Call]) -> bool:
    return any(
        block.jump != hir.Jump.END_BRANCH or _have_nested_jump(block.calls)
        for c in calls
        if c.callee is hir.if_else
        for block in c.args[1:]
    )


@has_multiple_results
class IfElse(TypedOperation):
    def __init__(
        self,
        cond: Var,
        then_block: Block,
        else_block: Block,
        result_vars: Sequence[Var],
        loc: Loc,
    ):
        super().__init__(
            "ifelse",
            operands={"cond": cond},
            result_vars=list(result_vars),
            nested_blocks=[then_block, else_block],
            loc=loc,
        )

    @property
    def then_block(self):
        return self.nested_blocks[0]

    @property
    def else_block(self):
        return self.nested_blocks[1]

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, ...]:
        cond_val = ctx.get_value(self.cond)
        result_types = tuple(ctx.typeof(v) for v in self.result_vars)
        result_type_ids = tuple(typeid(ctx.type_table, t) for t in result_types)
        nested_builder = bc.encode_IfOp(ctx.builder, result_type_ids, cond_val)

        for block in (self.then_block, self.else_block):
            with nested_builder.new_block(()):
                generate_bytecode_for_block(ctx, block)

        return nested_builder.done()

    @override
    def _to_string_block_prefixes(self) -> List[str]:
        return ["then", "else"]

    @override
    def _to_string_rhs(self) -> str:
        return f"if(cond={self.cond})"


async def _flatten_branch(branch: hir.Block) -> Var | None:
    from .._passes.hir2ir import dispatch_hir_block
    info = ControlFlowInfo((), flatten=True)
    with Scope.get_current().change_if_else_info(info):
        await dispatch_hir_block(branch)
    if len(info.jumps) == 0:
        return None
    else:
        assert len(info.jumps) == 1
        jump = info.jumps[0]
        assert len(jump.outputs) in (0, 1)
        return None if len(jump.outputs) == 0 else jump.outputs[0]


@impl(hir.if_else)
async def if_else_impl(cond: Var, then_block: hir.Block, else_block: hir.Block) -> Var | None:
    from .._passes.hir2ir import dispatch_hir_block

    require_bool(cond)
    if cond.is_constant():
        branch_taken = then_block if cond.get_constant() else else_block
        return await _flatten_branch(branch_taken)

    builder = Builder.get_current()
    if builder.reduction_body:
        raise TileSyntaxError("Branching inside reduction body is not supported."
                              " Consider ct.where() as a workaround.")

    # Get the total number of results by adding the number of stored variables.
    # Note: we sort the stored variable indices to make the order deterministic.
    scope = Scope.get_current()
    stored_locals = tuple(sorted(scope.get_local_index(name)
                                 for name in (then_block.stored_names | else_block.stored_names)))

    # Convert the "then" branch from HIR to IR
    info = ControlFlowInfo(stored_locals)
    then_loc = then_block.loc.with_call_site(scope.call_site)
    with nested_block(then_loc) as new_then_block, scope.change_if_else_info(info), \
            scope.local.enter_branch():
        await dispatch_hir_block(then_block)

    # If "then" branch doesn't yield, transform our if-else into the following:
    #    if cond:
    #        <then_block>
    #    else:
    #        EndBranch
    #    <else_block>
    # This is to avoid the situation where none of the branches yield.
    else_loc = else_block.loc.with_call_site(scope.call_site)
    if len(info.jumps) == 0:
        info = ControlFlowInfo(())
        with nested_block(else_loc) as new_else_block, scope.change_if_else_info(info), \
                scope.local.enter_branch():
            end_branch(None)
        add_operation(IfElse, (),
                      cond=cond, then_block=new_then_block, else_block=new_else_block)
        return await _flatten_branch(else_block)

    # Convert the "else" branch from HIR to IR
    with nested_block(else_loc) as new_else_block, scope.change_if_else_info(info), \
            scope.local.enter_branch():
        await dispatch_hir_block(else_block)

    # Do type/constant propagation
    num_results = len(info.jumps[0].outputs)
    result_phis = tuple(PhiState() for _ in range(num_results))
    for jump_info in info.jumps:
        for phi, v in zip(result_phis, jump_info.outputs, strict=True):
            phi.propagate(v)

    # Determine which results have valid types
    mask = tuple(not isinstance(phi.ty, InvalidType) for phi in result_phis)
    valid_result_types = tuple(phi.ty for phi, is_valid in zip(result_phis, mask) if is_valid)

    # Update the EndBranch operations by setting the outputs
    for jump_info in info.jumps:
        outputs = tuple(v for v, is_valid in zip(jump_info.outputs, mask, strict=True)
                        if is_valid)
        flat_outputs = flatten_aggregates(outputs, valid_result_types)
        jump_info.jump_op.update_operand("outputs", flat_outputs)

    # Generate an IfElse op
    flat_types = flatten_aggregate_types(valid_result_types)
    flat_results = add_operation(IfElse, flat_types,
                                 cond=cond, then_block=new_then_block, else_block=new_else_block)
    valid_results = unflatten_aggregates(flat_results, valid_result_types, valid_result_types)

    # Finalize the constant/loose type information
    valid_result_phis = tuple(phi for phi, is_valid in zip(result_phis, mask) if is_valid)
    for var, phi in zip(valid_results, valid_result_phis, strict=True):
        phi.finalize_constant_and_loose_type(var)

    it = iter(valid_results)
    all_results = tuple(next(it) if is_valid else None for is_valid in mask)
    assert next(it, None) is None

    # Get/create variables for the explicit result
    num_explicit_results = num_results - len(stored_locals)
    if num_explicit_results == 0:
        ret = None
    else:
        assert num_explicit_results == 1
        ret = all_results[0]
        if ret is None:
            ret = builder.ir_ctx.make_temp(builder.loc, undefined=True)

    # Update the scope for stored named
    for res_var, phi, local_idx in zip(all_results[num_explicit_results:],
                                       result_phis[num_explicit_results:],
                                       stored_locals, strict=True):
        if res_var is None:
            store_undefined(local_idx, phi.ty, phi.last_loc)
        else:
            store_var(local_idx, res_var, phi.last_loc)

    return ret


# Maps to ContinueOp in TileIR
@terminator
class Continue(TypedOperation):
    def __init__(
        self,
        loc: Loc,
        values: Tuple[Var, ...] = (),
    ):
        super().__init__("continue", operands={"values": values}, result_vars=[], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        next_values = [ctx.get_value_allow_undefined(var, ctx.typeof(body_var))
                       for var, body_var
                       in zip(self.values, ctx.innermost_loop.body_vars, strict=True)]
        bc.encode_ContinueOp(ctx.builder, next_values)
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return f"continue {', '.join([x.name for x in self.values])}"


def continue_():
    scope = Scope.get_current()
    info = scope.loop_info
    assert info is not None
    assert not info.flatten

    builder = Builder.get_current()
    builder.add_operation(Continue, (), dict(values=()))
    op = builder.ops[-1]
    next_values = tuple(scope.local.get(local_idx, builder.loc) for local_idx in info.stored_locals)
    info.jumps.append(JumpInfo(op, next_values))


# Maps to BreakOp
@terminator
class Break(TypedOperation):
    def __init__(
        self,
        loc: Loc,
        values: Tuple[Var, ...] = (),
    ):
        super().__init__("break", operands={"values": values}, result_vars=[], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        # body_vars is not a typo. We use body variables because they always contain the actual
        # types of the loop variables, whereas result variables may have an InvalidType.
        output_values = [ctx.get_value_allow_undefined(var, ctx.typeof(body_var))
                         for var, body_var
                         in zip(self.values, ctx.innermost_loop.body_vars, strict=True)]
        bc.encode_BreakOp(ctx.builder, output_values)
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return f"break {', '.join([x.name for x in self.values])}"


def break_():
    scope = Scope.get_current()
    info = scope.loop_info
    assert info is not None

    if info.flatten:
        return

    builder = Builder.get_current()
    builder.add_operation(Break, (), dict(values=()))
    op = builder.ops[-1]
    outputs = tuple(scope.local.get(local_idx, builder.loc) for local_idx in info.stored_locals)
    info.jumps.append(JumpInfo(op, outputs))


# Maps to YieldOp
@terminator
class EndBranch(TypedOperation):
    def __init__(
        self,
        loc: Loc,
        outputs: Tuple[Var, ...] = (),
    ):
        super().__init__("end_branch", operands={"outputs": outputs}, result_vars=[], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        output_values = tuple(ctx.get_value(var) for var in self.outputs)
        bc.encode_YieldOp(ctx.builder, output_values)
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return f"yield {', '.join([x.name for x in self.outputs])}"


def end_branch(output: Var | None):
    scope = Scope.get_current()
    info = scope.if_else_info
    outputs = () if output is None else (output,)
    if info.flatten:
        op = None
    else:
        builder = Builder.get_current()
        builder.add_operation(EndBranch, (), dict(outputs=()))
        op = builder.ops[-1]
        outputs += tuple(scope.local.get(local_idx, builder.loc)
                         for local_idx in info.stored_locals)
    info.jumps.append(JumpInfo(op, outputs))


@terminator
class Return(TypedOperation):
    def __init__(self, loc: Loc):
        super().__init__("return", operands={}, result_vars=[], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        bc.encode_ReturnOp(ctx.builder, ())
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return "return"


def return_(value: Var | None):
    if value is not None and value.get_type() is not NONE:
        raise TileTypeError("Tile kernels cannot return values")
    add_operation(Return, ())


def _check_value_numeric_type(value: Any, dtype: DType) -> None:
    value_type = typeof_pyval(value)
    if datatype.is_arithmetic(value_type):
        if not datatype.is_arithmetic(dtype):
            raise TileTypeError(f"Expect \"value\" to be a non-numeric dtype {dtype}, "
                                f"got numeric dtype {value_type}")
        # TODO: Both are numeric types, check the data range after ir dtype supports it.
    else:
        if value_type != dtype:
            raise TileTypeError(f"Expect \"value\" to be a {dtype}, got {value_type}")


class TypedConst(TypedOperation):
    def __init__(self, value: Any, result_var: Var, loc: Loc):
        super().__init__("typed_const", operands={}, result_vars=[result_var],
                         attributes={"value": value}, loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return ctx.constant(self.value, ctx.typeof(self.result_var))


def loosely_typed_const(value: Any,
                        ty: Optional[Type] = None,
                        loose_ty: Optional[Type] = None) -> Var:
    if ty is None:
        ty = typeof_pyval(value)
    ret = strictly_typed_const(value, ty)
    if loose_ty is None:
        loose_ty = loose_type_of_pyval(value)
    ret.set_loose_type(loose_ty)
    return ret


def strictly_typed_const(value: Any, ty: Type) -> Var:
    ret = add_operation(TypedConst, ty, value=value)
    if not isinstance(ty, TileTy):
        # We currently don't have a way to represent a tile constant
        ret.set_constant(value)
    return ret


# Computes lhs*rhs + acc.  Also known as FMA.
class FusedMulAddOperation(Operation):
    def __init__(self, lhs: Var, rhs: Var, acc: Var,
                 rounding_mode: RoundingMode, flush_to_zero: bool,
                 result_var: Var, loc: Loc):
        super().__init__(
            "fma",
            operands={"lhs": lhs, "rhs": rhs, "acc": acc},
            attributes={"rounding_mode": rounding_mode, "flush_to_zero": flush_to_zero},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        result_type = ctx.typeof(self.result_var)
        lhs = ctx.cast(ctx.get_value(self.lhs), ctx.typeof(self.lhs), result_type)
        rhs = ctx.cast(ctx.get_value(self.rhs), ctx.typeof(self.rhs), result_type)
        acc = ctx.cast(ctx.get_value(self.acc), ctx.typeof(self.acc), result_type)
        return bc.encode_FmaOp(ctx.builder,
                               ctx.typeid_of(self.result_var),
                               lhs, rhs, acc,
                               rounding_mode_to_bytecode[self.rounding_mode],
                               self.flush_to_zero)


# Does not do broadcasting or type promotion, hence the name "Raw"
class RawComparisonOperation(TypedOperation):
    def __init__(self, fn: str, lhs: Var, rhs: Var, result_var: Var, loc: Loc):
        super().__init__(
            "raw_cmp",
            operands={"lhs": lhs, "rhs": rhs},
            attributes={"fn": fn},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        from .._ir2bytecode import encode_comparison
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)
        dtype = get_dtype(self.lhs.get_type())
        result_typeid = ctx.typeid_of(self.result_var)
        return encode_comparison(ctx.builder, self.fn, lhs, rhs, dtype, result_typeid)


def raw_comparison(fn: str, x: Var, y: Var) -> Var:
    if fn == 'is':
        raise TileTypeError("\"is\" only supports constants")

    ty = x.get_type()
    assert ty == y.get_type()
    res_ty = change_dtype(ty, datatype.bool_)
    return add_operation(RawComparisonOperation, res_ty, fn=fn, lhs=x, rhs=y)


@contextmanager
def _reraise_tile_exception():
    try:
        yield
    except (ZeroDivisionError, ValueError) as e:
        raise TileValueError(str(e))
    except TypeError as e:
        raise TileTypeError(str(e))


def _binop_propagate_constant(fn: str, x: Any, y: Any, type: Optional[Type]) -> Var:
    impl = BINOP_REGISTRY[fn].impl
    with _reraise_tile_exception():
        res = impl(x, y)

    if type is None:
        return loosely_typed_const(res)
    else:
        return strictly_typed_const(res, type)


@impl(ct.equal, fixed_args=["eq"])
@impl(ct.greater, fixed_args=["gt"])
@impl(ct.not_equal, fixed_args=["ne"])
@impl(ct.greater_equal, fixed_args=["ge"])
@impl(ct.less, fixed_args=["lt"])
@impl(ct.less_equal, fixed_args=["le"])
def comparison(fn: str, x: Var, y: Var) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    common_ty = promote_types(x_ty, y_ty)
    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        res_ty = change_dtype(common_ty, datatype.bool_)
        return _binop_propagate_constant(fn, x.get_constant(), y.get_constant(), res_ty)

    return raw_comparison(fn, x, y)


def _is_none_compare(x: Var, y: Var, *, negate: bool, op_name: str) -> Var:
    x_is_none = x.get_type() is NONE
    y_is_none = y.get_type() is NONE
    if not (x_is_none or y_is_none):
        raise TileTypeError(f"Operator '{op_name}' expects one of the operands to be None")
    return loosely_typed_const((x_is_none == y_is_none) ^ negate)


@impl(operator.is_)
def operator_is_impl(x: Var, y: Var):
    return _is_none_compare(x, y, negate=False, op_name="is")


@impl(operator.is_not)
def operator_is_not_impl(x: Var, y: Var):
    return _is_none_compare(x, y, negate=True, op_name="is not")


@impl(operator.eq, fixed_args=["eq"])
@impl(operator.ne, fixed_args=["ne"])
@impl(operator.lt, fixed_args=["lt"])
@impl(operator.le, fixed_args=["le"])
@impl(operator.gt, fixed_args=["gt"])
@impl(operator.ge, fixed_args=["ge"])
def comparison_operator_impl(fn: str, x: Var, y: Var) -> Var:
    x_ty = x.get_type()
    y_ty = y.get_type()

    match x_ty, y_ty:
        case DTypeSpec(), DTypeSpec():
            return _binop_propagate_constant(fn, x_ty.dtype, y_ty.dtype, None)
        case StringTy(), StringTy():
            return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)
        case _, _:
            return comparison(fn, x, y)


def _promote_and_broadcast_to(x: Var, ty: TileTy | DType) -> Var:
    is_tile = isinstance(ty, TileTy)
    shape = ty.shape_value if is_tile else ()
    return broadcast_to(astype(x, get_dtype(ty)), shape, keep_scalar=not is_tile)


# Does not do broadcasting or type promotion, hence the name "Raw"
class RawBinaryBitwiseOperation(TypedOperation):
    def __init__(self, fn: str, lhs: Var, rhs: Var, result_var: Var, loc: Loc):
        super().__init__(
            "raw_binary_bitwise",
            operands={"lhs": lhs, "rhs": rhs},
            attributes={"fn": fn},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        res_typeid = ctx.typeid_of(self.result_var)
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)
        match self.fn:
            case "and_": return bc.encode_AndIOp(ctx.builder, res_typeid, lhs, rhs)
            case "or_": return bc.encode_OrIOp(ctx.builder, res_typeid, lhs, rhs)
            case "xor": return bc.encode_XOrIOp(ctx.builder, res_typeid, lhs, rhs)
            case _:
                raise NotImplementedError(f"Missing binary bitwise implementation for {self.fn}")


def raw_binary_bitwise(fn: str, x: Var, y: Var) -> Var:
    ty = x.get_type()
    assert ty == y.get_type()
    return add_operation(RawBinaryBitwiseOperation, ty, fn=fn, lhs=x, rhs=y)


@impl(ct.bitwise_and, fixed_args=["and_"])
@impl(ct.bitwise_or, fixed_args=["or_"])
@impl(ct.bitwise_xor, fixed_args=["xor"])
@impl(operator.and_, fixed_args=["and_"])
@impl(operator.or_, fixed_args=["or_"])
@impl(operator.xor, fixed_args=["xor"])
def binary_bitwise(fn: str, x: Var, y: Var) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    lhs_dtype = get_dtype(x_ty)
    rhs_dtype = get_dtype(y_ty)

    if not (datatype.is_integral(lhs_dtype) or datatype.is_boolean(lhs_dtype)) \
            or not (datatype.is_integral(rhs_dtype) or datatype.is_boolean(rhs_dtype)):
        raise TileTypeError("Bitwise operations require integers or booleans."
                            " Use an explicit cuda.tile.bitcast() for non-integer operands.")

    x_loose = isinstance(x_ty, LooselyTypedScalar)
    y_loose = isinstance(y_ty, LooselyTypedScalar)
    if x_loose == y_loose and lhs_dtype != rhs_dtype:
        msg = "Bitwise operands must have same data type, got:"
        msg += f" {lhs_dtype} and {rhs_dtype}"
        raise TileTypeError(msg)

    if {lhs_dtype, rhs_dtype} == {datatype.bool_, datatype.int8}:
        raise TileTypeError("Bitwise op does not support bool and int8")

    common_ty = promote_types(x_ty, y_ty)
    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        return _binop_propagate_constant(fn, x.get_constant(), y.get_constant(), common_ty)

    return raw_binary_bitwise(fn, x, y)


# Does not do broadcasting or type promotion, hence the name "Raw"
class RawBitwiseShiftOperation(TypedOperation):
    def __init__(self, fn: str, lhs: Var, rhs: Var, result_var: Var, loc: Loc):
        super().__init__(
            "raw_bitwise_shift",
            operands={"lhs": lhs, "rhs": rhs},
            attributes={"fn": fn},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_ty = self.result_var.get_type()
        res_type_id = typeid(ctx.type_table, res_ty)
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)
        match self.fn:
            case "lshift":
                return bc.encode_ShLIOp(ctx.builder, res_type_id, lhs, rhs, bc.IntegerOverflow.NONE)
            case "rshift":
                return bc.encode_ShRIOp(ctx.builder, res_type_id, lhs, rhs,
                                        datatype.get_signedness(get_dtype(res_ty)))
            case _: raise NotImplementedError()


def raw_bitwise_shift(fn: str, x: Var, y: Var) -> Var:
    ty = x.get_type()
    assert ty == y.get_type()
    return add_operation(RawBitwiseShiftOperation, ty, fn=fn, lhs=x, rhs=y)


@impl(ct.bitwise_lshift, fixed_args=["lshift"])
@impl(ct.bitwise_rshift, fixed_args=["rshift"])
@impl(operator.lshift, fixed_args=["lshift"])
@impl(operator.rshift, fixed_args=["rshift"])
def bitwise_shift(fn: str, x: Var, y: Var) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    lhs_dtype = get_dtype(x_ty)
    if not datatype.is_integral(lhs_dtype):
        msg = f'Bitwise shift requires an integer for left-hand side, got: {lhs_dtype}'
        raise TileTypeError(msg)

    rhs_dtype = get_dtype(y_ty)
    if not datatype.is_integral(rhs_dtype):
        msg = f'Bitwise shift requires an integer for right-hand side, got: {rhs_dtype}'
        raise TileTypeError(msg)

    common_ty = promote_types(x_ty, y_ty)
    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        return _binop_propagate_constant(fn, x.get_constant(), y.get_constant(), common_ty)

    return raw_bitwise_shift(fn, x, y)


# Does not do broadcasting or type promotion, hence the name "Raw"
class RawBinaryArithmeticOperation(TypedOperation):
    def __init__(self, fn: str, lhs: Var, rhs: Var,
                 rounding_mode: Optional[RoundingMode], flush_to_zero: bool,
                 result_var: Var, loc: Loc):
        super().__init__(
            "raw_binary_arith",
            operands={"lhs": lhs, "rhs": rhs},
            attributes={"fn": fn, "rounding_mode": rounding_mode, "flush_to_zero": flush_to_zero},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        result_ty = self.result_var.get_type()
        dtype = get_dtype(result_ty)
        kind = "float" if datatype.is_float(dtype) else "int"
        res_typeid = typeid(ctx.type_table, result_ty)
        rounding_mode = rounding_mode_to_bytecode[self.rounding_mode]
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)

        match self.fn, kind:
            case "add", "int":
                return bc.encode_AddIOp(ctx.builder, res_typeid, lhs, rhs,
                                        overflow=bc.IntegerOverflow.NONE)
            case "add", "float":
                return bc.encode_AddFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "sub", "int":
                return bc.encode_SubIOp(ctx.builder, res_typeid, lhs, rhs,
                                        overflow=bc.IntegerOverflow.NONE)
            case "sub", "float":
                return bc.encode_SubFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "mul", "int":
                return bc.encode_MulIOp(ctx.builder, res_typeid, lhs, rhs,
                                        overflow=bc.IntegerOverflow.NONE)
            case "mul", "float":
                return bc.encode_MulFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "floordiv", "int":
                return bc.encode_DivIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype),
                                        rounding=bc.RoundingMode.NEGATIVE_INF)
            case "cdiv", "int":
                return bc.encode_DivIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype),
                                        rounding=bc.RoundingMode.POSITIVE_INF)
            case "truediv", "float":
                return bc.encode_DivFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "pow", "float":
                return bc.encode_PowOp(ctx.builder, res_typeid, lhs, rhs)
            case "min", "int":
                return bc.encode_MinIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype))
            case "min", "float":
                return bc.encode_MinFOp(ctx.builder, res_typeid, lhs, rhs,
                                        propagate_nan=False,
                                        flush_to_zero=self.flush_to_zero)
            case "max", "int":
                return bc.encode_MaxIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype))
            case "max", "float":
                return bc.encode_MaxFOp(ctx.builder, res_typeid, lhs, rhs,
                                        propagate_nan=False,
                                        flush_to_zero=self.flush_to_zero)
            case "c_mod", "float":
                # C-style modulo
                return bc.encode_RemFOp(ctx.builder, res_typeid, lhs, rhs)
            case "c_mod", "int":
                # C-style modulo
                return bc.encode_RemIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype))
            case _:
                raise NotImplementedError(f"Missing binary arithmetic implementation"
                                          f" for {self.fn}, {kind}")


def raw_binary_arithmetic(fn: str, x: Var, y: Var, rounding_mode: Optional[RoundingMode] = None,
                          flush_to_zero: bool = False) -> Var:
    ty = x.get_type()
    assert ty == y.get_type(), f"{ty} != {y.get_type()}"
    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, get_dtype(ty))
    return add_operation(RawBinaryArithmeticOperation, ty, fn=fn, lhs=x, rhs=y,
                         rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


def binary_arithmetic(fn: str, x: Var, y: Var, rounding_mode: Optional[RoundingMode] = None,
                      flush_to_zero: bool = False) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    if get_dtype(x_ty) == get_dtype(y_ty) == datatype.bool_:
        raise TileTypeError(f'Binary arithmetic op `{fn}` does not support bool, '
                            f'please cast bool to int')

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    force_float = (fn == "truediv")
    common_ty = promote_types(x_ty, y_ty, force_float=force_float)

    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        return _binop_propagate_constant(fn, x.get_constant(), y.get_constant(), common_ty)

    return raw_binary_arithmetic(fn, x, y, rounding_mode, flush_to_zero)


@impl(ct.floordiv, fixed_args=["floordiv"])
@impl(ct.cdiv, fixed_args=["cdiv"])
@impl(ct.pow, fixed_args=["pow"])
@impl(operator.add, fixed_args=["add"])
@impl(operator.sub, fixed_args=["sub"])
@impl(operator.mul, fixed_args=["mul"])
@impl(operator.floordiv, fixed_args=["floordiv"])
@impl(operator.truediv, fixed_args=["truediv"])
@impl(operator.pow, fixed_args=["pow"])
@impl(min, fixed_args=["min"])
@impl(max, fixed_args=["max"])
def binary_arithmetic_impl(fn: str, x: Var, y: Var) -> Var:
    return binary_arithmetic(fn, x, y)


@impl(ct.minimum, fixed_args=["min"])
@impl(ct.maximum, fixed_args=["max"])
def binary_arithmetic_impl_with_ftz(fn: str, x: Var, y: Var, flush_to_zero: Var) -> Var:
    flush_to_zero = require_constant_bool(flush_to_zero)
    return binary_arithmetic(fn, x, y, flush_to_zero=flush_to_zero)


@impl(ct.add, fixed_args=["add"])
@impl(ct.sub, fixed_args=["sub"])
@impl(ct.mul, fixed_args=["mul"])
@impl(ct.truediv, fixed_args=["truediv"])
def binary_arithmetic_impl_with_rd_and_ftz(fn: str, x: Var, y: Var,
                                           rounding_mode: Var, flush_to_zero: Var) -> Var:
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return binary_arithmetic(fn, x, y, rounding_mode, flush_to_zero)


@impl(operator.mod)
@impl(ct.mod)
def mod(x: Var, y: Var) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)
    if get_dtype(x_ty) == get_dtype(y_ty) == datatype.bool_:
        raise TileTypeError('Modulo operation does not support bool')

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        with _reraise_tile_exception():
            res = x_ty.value % y_ty.value
        return loosely_typed_const(res)

    # Usual promote & broadcast logic
    common_ty = promote_types(x_ty, y_ty)
    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        with _reraise_tile_exception():
            res = x.get_constant() % y.get_constant()
        return strictly_typed_const(res, common_ty)

    # TileOR rem follows the C behavior while Python's mod behavior differs.
    # So we generate the C-style mod first and then apply a correction.
    value = raw_binary_arithmetic("c_mod", x, y)

    # If the sign of `value` does not match the sign of `y`, apply a correction.
    zero = strictly_typed_const(0, common_ty)
    value_sign = comparison("lt", value, zero)
    y_sign = comparison("lt", y, zero)

    # need_fix = (value_sign ^ y_sign) & (value != 0)
    sign_mismatch = binary_bitwise("xor", value_sign, y_sign)
    value_not_zero = comparison("ne", value, zero)
    need_fix = binary_bitwise("and_", sign_mismatch, value_not_zero)

    fixed_value = binary_arithmetic("add", value, y)
    return where(need_fix, fixed_value, value)


@impl(slice)
def slice_impl(start: Var, stop: Var, step: Var) -> Var:
    if not (start.is_constant() and stop.is_constant() and step.is_constant()):
        raise TileTypeError("Non-constant slices are not supported")
    return loosely_typed_const(
        slice(start.get_constant(), stop.get_constant(), step.get_constant()))


def tuple_item(tup: TupleValue, index: int) -> Var:
    assert isinstance(tup, TupleValue)
    try:
        return tup.items[index]
    except IndexError:
        raise TileTypeError(
                f"Index {index} is out of range for a tuple of length {len(tup.items)}")


def tuple_slice(tup: TupleValue, slc: slice) -> Var:
    assert isinstance(tup, TupleValue)
    items = tup.items[slc]
    return build_tuple(items)


def tuple_getitem(x: Var, key: Var) -> Var:
    tuple_val = x.get_aggregate()
    key_ty = key.get_type()
    if isinstance(key_ty, SliceType):
        slc = require_constant_slice(key)
        return tuple_slice(tuple_val, slc)
    idx = require_constant_int(key)
    return tuple_item(tuple_val, idx)


def tile_expand_dims(x: Var, index: Tuple[Any, ...]) -> Var:
    x_type = x.get_type()

    for idx in index:
        if idx not in (None, Ellipsis, slice(None)):
            raise TileTypeError(
                f"Expected `None|np.newaxis` or `ellipsis` or full slice (`:`), "
                f"but got {idx}. Hint: Directly indexing a tile is not supported, "
                f"use `extract` or `item`.")

    num_slices = sum(1 for idx in index if isinstance(idx, slice))
    if num_slices > x_type.ndim:
        raise TileTypeError(f"Tile is {x_type.ndim}-dimensional, "
                            f"but {num_slices} were indexed")
    axes = []
    ellipsis_idx = None
    for i, idx in enumerate(index):
        if idx is Ellipsis:
            if ellipsis_idx is not None:
                raise TileTypeError("Only one ellipsis is allowed")
            ellipsis_idx = i
        elif idx is None:
            axes.append(i - len(index) if ellipsis_idx is not None and i > ellipsis_idx else i)
    new_rank = x_type.ndim + len(axes)
    new_shape = list(x_type.shape_value)
    for axis in axes:
        normalized_axis = axis + new_rank if axis < 0 else axis
        new_shape.insert(normalized_axis, 1)
    return reshape(x, tuple(new_shape))


def tile_getitem(x: Var, key: Var) -> Var:
    key_ty = key.get_type()
    if isinstance(key_ty, NoneType):
        return tile_expand_dims(x, (None,))
    elif isinstance(key_ty, TupleTy):
        if not key.is_constant():
            raise TileTypeError("Tile subscript must be a constant tuple")
        return tile_expand_dims(x, key.get_constant())
    else:
        raise TileTypeError("Directly indexing a tile is not supported; "
                            "use `extract()` or `item()` instead.")


class GetArrayListItem(TypedOperation):
    def __init__(self, x: Var, index: Var, result_vars: list[Var], loc: Loc):
        super().__init__("get_array_list_item",
                         operands={"x": x, "index": index},
                         result_vars=result_vars,
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        list_ty = ctx.typeof(self.x)
        assert isinstance(list_ty, ListTy)

        # First, load a (1 x item_tile_size) tile that represents the item
        partition_view = ctx.get_value(self.x)
        item_size = get_list_item_repr_size_in_words(list_ty.item_type)
        item_tile_size = get_list_partition_view_tile_size(item_size)
        pv_tile_type_id = ctx.type_table.tile(ctx.type_table.I64, (1, item_tile_size))
        index = ctx.get_value(self.index)
        index_i32 = ctx.cast(index, ctx.typeof(self.index), datatype.int32)

        zero_i32 = ctx.constant(0, datatype.int32)

        loaded_tile, _token = bc.encode_LoadViewTkoOp(
            ctx.builder,
            tile_type=pv_tile_type_id,
            result_token_type=ctx.type_table.Token,
            view=partition_view,
            index=(index_i32, zero_i32),
            token=None,
            memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
            memory_scope=None,
            optimization_hints=None
        )

        item_typeid_tuple = tuple(typeid(ctx.type_table, ty)
                                  for ty in list_ty.item_type.flatten_aggregate())

        # Next, unpack the tile into individual values that represent the item
        assert isinstance(list_ty.item_type, ArrayTy)
        assert len(item_typeid_tuple) == item_size

        # Extract and reshape each element of the (1 x item_tile_size) tile
        # as a separate i64 scalar
        i64_scalar_ty = ctx.type_table.tile(ctx.type_table.I64, ())
        i64_1x1_ty = ctx.type_table.tile(ctx.type_table.I64, (1, 1))
        extracted_words = tuple(
            bc.encode_ReshapeOp(
                ctx.builder,
                i64_scalar_ty,
                bc.encode_ExtractOp(ctx.builder, i64_1x1_ty, loaded_tile,
                                    (zero_i32, ctx.constant(i, datatype.int32)),),
            )
            for i in range(item_size)
        )

        # Cast each of the i64 words to appropriate types
        return (
            # Cast the first word to data pointer
            bc.encode_IntToPtrOp(ctx.builder, item_typeid_tuple[0], extracted_words[0]),
            # Cast the remaining words to i32 shape/strides
            *(bc.encode_TruncIOp(ctx.builder, ty, w, bc.IntegerOverflow.NONE)
              for ty, w in zip(item_typeid_tuple[1:], extracted_words[1:], strict=True))
        )


def list_item(x: Var, index: Var) -> Var:
    list_ty = require_list_type(x)
    index_ty = require_scalar_or_0d_tile_type(index)
    index_dtype = get_dtype(index_ty)
    if not (isinstance(index_dtype, DType) and is_integral(index_dtype)):
        raise TypeError(f"Index must be an integer scalar or 0D Tile, got {index_ty}")
    item_ty = list_ty.item_type

    if not isinstance(item_ty, ArrayTy):
        raise TileTypeError(f"Indexing a list of {list_ty.item_type} is not implemented")

    flat_types = tuple(item_ty.flatten_aggregate())
    flat_results = add_operation(GetArrayListItem, flat_types, x=x, index=index)
    [ret] = unflatten_aggregates(flat_results, (item_ty,), (item_ty,))
    return ret


@impl(operator.getitem)
def getitem(object: Var, key: Var) -> Var:
    object_ty = object.get_type()
    match object_ty:
        case TupleTy(): return tuple_getitem(object, key)
        case TileTy(): return tile_getitem(object, key)
        case ListTy(): return list_item(object, key)
    raise TileTypeError(f'Indexing an object of type {object_ty} is not supported')


@impl(len)
def len_impl(x: Var) -> Var:
    x_type = x.get_type()
    if isinstance(x_type, TupleTy):
        return loosely_typed_const(len(x_type))
    require_list_type(x)
    list_val = x.get_aggregate()
    assert isinstance(list_val, ListValue)
    return list_val.length


@impl(hir.build_tuple)
def build_tuple(items: tuple[Var, ...]) -> Var:
    ty = TupleTy(tuple(x.get_type() for x in items))
    loose_ty = TupleTy(tuple(x.get_loose_type() for x in items))
    res = make_aggregate(TupleValue(items), ty, loose_ty)
    if all(x.is_constant() for x in items):
        res.set_constant(tuple(x.get_constant() for x in items))
    return res


class Unary(TypedOperation):
    def __init__(self, fn: str, operand: Var,
                 rounding_mode: Optional[RoundingMode], flush_to_zero: bool,
                 result_var: Var, loc: Loc):
        super().__init__("unaryop",
                         operands={"operand": operand},
                         attributes={
                             "fn": fn,
                             "rounding_mode": rounding_mode,
                             "flush_to_zero": flush_to_zero,
                         },
                         result_vars=[result_var],
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x = ctx.get_value(self.operand)
        rounding_mode = rounding_mode_to_bytecode[self.rounding_mode]
        flush_to_zero = self.flush_to_zero
        input_type = ctx.typeof(self.operand)
        input_dtype = get_dtype(input_type)
        flt = is_float(input_dtype) or is_restricted_float(input_dtype)
        res_type_id = ctx.typeid_of(self.result_var)

        match self.fn, flt:
            case "abs", True: return bc.encode_AbsFOp(ctx.builder, res_type_id, x)
            case "abs", False: return bc.encode_AbsIOp(ctx.builder, res_type_id, x)
            case "neg", True: return bc.encode_NegFOp(ctx.builder, res_type_id, x)
            case "neg", False: return bc.encode_NegIOp(ctx.builder, res_type_id, x,
                                                       bc.IntegerOverflow.NONE)
            case "exp", True: return bc.encode_ExpOp(ctx.builder, res_type_id, x)
            case "exp2", True: return bc.encode_Exp2Op(ctx.builder, res_type_id, x,
                                                       flush_to_zero=flush_to_zero)
            case "sin", True: return bc.encode_SinOp(ctx.builder, res_type_id, x)
            case "cos", True: return bc.encode_CosOp(ctx.builder, res_type_id, x)
            case "sinh", True: return bc.encode_SinHOp(ctx.builder, res_type_id, x)
            case "cosh", True: return bc.encode_CosHOp(ctx.builder, res_type_id, x)
            case "tan", True: return bc.encode_TanOp(ctx.builder, res_type_id, x)
            # TODO: rounding mode support depending on bytecode version
            case "tanh", True: return bc.encode_TanHOp(ctx.builder, res_type_id, x,
                                                       rounding_mode=bc.RoundingMode.FULL)
            case "log", True: return bc.encode_LogOp(ctx.builder, res_type_id, x)
            case "log2", True: return bc.encode_Log2Op(ctx.builder, res_type_id, x)
            case "sqrt", True: return bc.encode_SqrtOp(ctx.builder, res_type_id, x,
                                                       rounding_mode=rounding_mode,
                                                       flush_to_zero=flush_to_zero)
            case "rsqrt", True: return bc.encode_RsqrtOp(ctx.builder, res_type_id, x,
                                                         flush_to_zero=flush_to_zero)
            case "floor", True: return bc.encode_FloorOp(ctx.builder, res_type_id, x)
            case "ceil", True: return bc.encode_CeilOp(ctx.builder, res_type_id, x)
            case "invert", False:
                # ~tx == tx ^ ~0
                all_ones = (-1 if datatype.is_signed(input_dtype)
                            else ~(-1 << input_dtype.bitwidth))
                all_ones_tile = ctx.constant(all_ones, input_type)
                return bc.encode_XOrIOp(ctx.builder, res_type_id, x, all_ones_tile)
            case _:
                raise NotImplementedError(f"Missing implementation for unary op: {self.fn}")


def _unary_promote_to_int(x):
    return astype(x, datatype.default_int_type)


def _unary_promote_to_float(x):
    return astype(x, datatype.default_float_type)


def _unary_preserve(x):
    return x


@dataclass
class _UnaryBehavior:
    bool_handler: Optional[Callable[[Var], Var]]
    int_handler: Optional[Callable[[Var], Var]]
    float_handler: Optional[Callable[[Var], Var]]


def _unary_propagate_constant(fn: str, arg: Any) -> Any:
    impl = UNARYOP_REGISTRY[fn].impl
    with _reraise_tile_exception():
        return impl(arg)


def unary(fn: str, behavior: _UnaryBehavior, x: Var,
          rounding_mode: Optional[RoundingMode] = None, flush_to_zero: bool = False) -> Var:
    x_type = require_tile_or_scalar_maybe_loose_type(x)
    if isinstance(x_type, LooselyTypedScalar):
        res = _unary_propagate_constant(fn, x_type.value)
        return loosely_typed_const(res)

    input_dtype = get_dtype(x_type)
    if is_boolean(input_dtype):
        if behavior.bool_handler is None:
            raise TileTypeError("Boolean inputs are not supported")
        x = behavior.bool_handler(x)
    elif is_integral(input_dtype):
        if behavior.int_handler is None:
            raise TileTypeError("Integer inputs are not supported")
        x = behavior.int_handler(x)
    elif is_float(input_dtype) or is_restricted_float(input_dtype):
        if behavior.float_handler is None:
            raise TileTypeError("Float inputs are not supported")
        x = behavior.float_handler(x)
    else:
        raise TileTypeError(f"Unexpected input dtype {input_dtype}")

    ty = x.get_type()
    if x.is_constant():
        res = _unary_propagate_constant(fn, x.get_constant())
        return strictly_typed_const(res, ty)

    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, get_dtype(ty))
    return add_operation(Unary, ty, fn=fn, operand=x,
                         rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


_UNARY_FLOAT = _UnaryBehavior(_unary_promote_to_float, _unary_promote_to_float, _unary_preserve)
_UNARY_STRICT_FLOAT = _UnaryBehavior(None, None, _unary_preserve)
_UNARY_INT_FLOAT = _UnaryBehavior(_unary_promote_to_int, _unary_preserve, _unary_preserve)
_UNARY_BOOL_INT = _UnaryBehavior(_unary_preserve, _unary_preserve, None)
_UNARY_ANYTHING = _UnaryBehavior(_unary_preserve, _unary_preserve, _unary_preserve)


@impl(operator.not_)
def logical_not_impl(x: Var) -> Var:
    ty = require_scalar_or_0d_tile_maybe_loose_type(x)

    if isinstance(ty, LooselyTypedScalar):
        return loosely_typed_const(not ty.value)

    x = astype(x, datatype.bool_)
    if x.is_constant():
        return strictly_typed_const(not x.get_constant(), x.get_type())

    return add_operation(Unary, x.get_type(), fn="invert", operand=x,
                         rounding_mode=None, flush_to_zero=False)


@impl(operator.pos)
def pos_impl(x: Var):
    ty = require_tile_or_scalar_maybe_loose_type(x)

    if isinstance(ty, LooselyTypedScalar):
        return loosely_typed_const(+ty.value)

    if get_dtype(ty) == datatype.bool_:
        return astype(x, datatype.default_int_type)
    else:
        return x


@impl(ct.log, fixed_args=["log", _UNARY_FLOAT])
@impl(ct.log2, fixed_args=["log2", _UNARY_FLOAT])
@impl(ct.tan, fixed_args=["tan", _UNARY_FLOAT])
@impl(ct.tanh, fixed_args=["tanh", _UNARY_FLOAT])
@impl(ct.sin, fixed_args=["sin", _UNARY_FLOAT])
@impl(ct.sinh, fixed_args=["sinh", _UNARY_FLOAT])
@impl(ct.cos, fixed_args=["cos", _UNARY_FLOAT])
@impl(ct.cosh, fixed_args=["cosh", _UNARY_FLOAT])
@impl(ct.exp, fixed_args=["exp", _UNARY_FLOAT])
@impl(ct.bitwise_not, fixed_args=["invert", _UNARY_BOOL_INT])
@impl(ct.floor, fixed_args=["floor", _UNARY_STRICT_FLOAT])
@impl(ct.ceil, fixed_args=["ceil", _UNARY_STRICT_FLOAT])
@impl(ct.negative, fixed_args=["neg", _UNARY_INT_FLOAT])
@impl(ct.abs, fixed_args=["abs", _UNARY_ANYTHING])
@impl(abs, fixed_args=["abs", _UNARY_ANYTHING])
@impl(operator.invert, fixed_args=["invert", _UNARY_BOOL_INT])
@impl(operator.neg, fixed_args=["neg", _UNARY_INT_FLOAT])
def unary_impl(fn: str, behavior: _UnaryBehavior, x: Var) -> Var:
    return unary(fn, behavior, x)


@impl(ct.rsqrt, fixed_args=["rsqrt", _UNARY_FLOAT])
@impl(ct.exp2, fixed_args=["exp2", _UNARY_FLOAT])
def unary_impl_with_ftz(fn: str, behavior: _UnaryBehavior, x: Var, flush_to_zero: Var) -> Var:
    flush_to_zero = require_constant_bool(flush_to_zero)
    return unary(fn, behavior, x, flush_to_zero=flush_to_zero)


@impl(ct.sqrt, fixed_args=["sqrt", _UNARY_FLOAT])
def unary_impl_with_rd_and_ftz(fn: str, behavior: _UnaryBehavior,
                               x: Var, rounding_mode: Var, flush_to_zero: Var) -> Var:
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return unary(fn, behavior, x, rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


@impl(getattr)
def getattr_impl(object: Var, name: Var) -> Var:
    ty = object.get_type()
    attr_name = require_constant_str(name)
    match ty, attr_name:
        case ArrayTy(), "dtype": return loosely_typed_const(ty.dtype)
        case ArrayTy(), "ndim": return loosely_typed_const(ty.ndim)
        case ArrayTy(), "shape": return build_tuple(object.get_aggregate().shape)
        case ArrayTy(), "strides": return build_tuple(object.get_aggregate().strides)
        case ArrayTy(), "slice": return bind_method(object, ct._m_array_slice)

        case TileTy(), "dtype": return loosely_typed_const(ty.dtype)
        case TileTy(), "shape": return loosely_typed_const(ty.shape_value)
        case TileTy(), "ndim": return loosely_typed_const(ty.ndim)

        case TileTy(), "extract": return bind_method(object, ct.extract)
        case TileTy(), "reshape": return bind_method(object, ct.reshape)
        case TileTy(), "astype": return bind_method(object, ct.astype)
        case TileTy(), "permute": return bind_method(object, ct.permute)
        case TileTy(), "transpose": return bind_method(object, ct.transpose)
        case TileTy(), "item": return bind_method(object, ct._m_tile_item)

        case ModuleTy(), _:
            try:
                return loosely_typed_const(getattr(ty.py_mod, attr_name))
            except AttributeError:
                pass

        case TypeTy(), _:
            try:
                return loosely_typed_const(getattr(ty.ty, attr_name))
            except AttributeError:
                pass

        case _: pass

    raise TileTypeError(f"No such attribute '{attr_name}' for object of type {ty}")


def bind_method(object: Var, func) -> Var:
    agg_value = BoundMethodValue(object)
    res_ty = BoundMethodTy(object.get_type(), func)
    return make_aggregate(agg_value, res_ty)


class Assign(TypedOperation):
    def __init__(self, value: Var, result_var: Var, loc: Loc):
        super().__init__(
            "assign", operands={"value": value}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        # FIXME: Ideally, all Assign ops should be eliminated before the bytecode generation stage.
        #        But keep this for now just in case.
        return ctx.get_value(self.value)

    @override
    def _to_string_rhs(self) -> str:
        return f"{self.value.name}"


def assign(value: Var, res: Var) -> None:
    Builder.get_current().append_verbatim(Assign(value, res, res.loc))
    res.ctx.copy_type_information(value, res)
    if value.is_undefined():
        res.set_undefined()


@impl(hir.identity)
def identity_impl(x: Var) -> Var:
    if x.is_constant():
        return loosely_typed_const(x.get_constant(), x.get_type(), x.get_loose_type())
    else:
        return x


@impl(range)
def range_(args: Tuple[Var, ...]) -> Var:
    if not 1 <= len(args) <= 3:
        raise TileTypeError(f"Invalid number of arguments: {len(args)}")
    for arg in args:
        require_signed_integer_scalar_or_0d_tile_type(arg)

    if len(args) == 1:
        start = strictly_typed_const(0, datatype.default_int_type)
        stop = args[0]
        step = strictly_typed_const(1, datatype.default_int_type)
    elif len(args) == 2:
        start, stop = args[0], args[1]
        step = strictly_typed_const(1, datatype.default_int_type)
    else:
        start, stop, step = args[0], args[1], args[2]
        # FIXME(Issue 314): Support negative step.
        # Error out if step is constant and not positive.
        if step.is_constant() and step.get_constant() <= 0:
            raise TileTypeError(f"Step must be positive, got {step.get_constant()}")

    agg_value = RangeValue(start, stop, step)
    ty = RangeIterType(datatype.default_int_type)
    return make_aggregate(agg_value, ty)


def _register_dtype_constructors(f):
    """
    Helper decorator to register all DType constructors automatically.
    """
    for obj, ty in dtype_registry.items():
        if isinstance(ty, DTypeConstructor):
            f = impl(obj, fixed_args=[ty.dtype])(f)
    return f


@_register_dtype_constructors
def dtype_constructor_impl(new_dtype: DType, x: Var) -> Var:
    if x.is_constant():
        try:
            const_value = new_dtype._py_type(x.get_constant())
        except (ValueError, TypeError):
            raise TileTypeError(f"Invalid argument type for {new_dtype}")
        return strictly_typed_const(const_value, ty=new_dtype)

    x_ty = require_scalar_or_0d_tile_type(x)
    if isinstance(x_ty, TileTy):
        x = tile_item(x)

    return astype(x, new_dtype)


@impl(float, fixed_args=[float])
@impl(int, fixed_args=[int])
@impl(bool, fixed_args=[bool])
def builtin_numeric_ctor_impl(ctor_obj: Any, x: Var) -> Var:
    if not x.is_constant():
        raise TileTypeError(f"{ctor_obj.__name__}() expects a constant argument")
    const = x.get_constant()
    try:
        value = ctor_obj(const)
    except (ValueError, TypeError, OverflowError):
        raise TileTypeError(f"Invalid argument for {ctor_obj.__name__}({const})")
    return loosely_typed_const(value)


# ================================================
# Tile specific operations
# ================================================

class TileBid(TypedOperation):
    def __init__(self, axis: int, result_var: Var, loc: Loc):
        super().__init__(
            "tile_bid", operands={}, attributes={"axis": axis}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        axis = self.axis
        res_typeid = ctx.typeid_of(self.result_var)
        return bc.encode_GetTileBlockIdOp(ctx.builder, res_typeid, res_typeid, res_typeid)[axis]


def bid(axis: int) -> Var:
    if axis not in (0, 1, 2):
        raise TileTypeError(f"Axis must be 0, 1, or 2, but {axis} was given.")
    return add_operation(TileBid, datatype.default_int_type, axis=axis)


@impl(ct.bid)
def bid_impl(axis: Var) -> Var:
    axis = require_constant_int(axis)
    return bid(axis)


class MakeTensorView(TypedOperation):
    def __init__(self, base_ptr: Var, shape: tuple[Var, ...], dynamic_strides: tuple[Var, ...],
                 result_var: Var, loc: Loc):
        super().__init__(
            "make_tensor_view",
            operands={"base_ptr": base_ptr, "shape": shape, "dynamic_strides": dynamic_strides},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        array_ty: ArrayTy = self.result_var.get_type()
        view_type_id = tensor_view_typeid(ctx.type_table, array_ty)
        base_ptr = ctx.get_value(self.base_ptr)
        shape = tuple(ctx.get_value(x) for x in self.shape)
        dynamic_strides = tuple(ctx.get_value(x) for x in self.dynamic_strides)
        return bc.encode_MakeTensorViewOp(ctx.builder,
                                          result_type=view_type_id,
                                          base=base_ptr,
                                          dynamicShape=shape,
                                          dynamicStrides=dynamic_strides)


class AssumeDivBy(TypedOperation):
    def __init__(self, x: Var, divisor: int, result_var: Var, loc: Loc):
        super().__init__(
            "assume_div_by",
            operands={"x": x},
            attributes={"divisor": divisor},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x = ctx.get_value(self.x)
        type_id = ctx.typeid_of(self.result_var)
        return bc.encode_AssumeOp(ctx.builder, type_id, x, bc.DivBy(self.divisor))


def assume_div_by(x: Var, divisor: int | None) -> Var:
    if divisor is None or divisor == 1 or CUDA_TILE_TESTING_DISABLE_DIV:
        return x
    return add_operation(AssumeDivBy, x.get_type(), x=x, divisor=divisor)


class AssumeBounded(TypedOperation):
    def __init__(self, x: Var, lower_bound: int | None, upper_bound: int | None,
                 result_var: Var, loc: Loc):
        super().__init__(
            "assume_bounded",
            operands={"x": x},
            attributes={"lower_bound": lower_bound, "upper_bound": upper_bound},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x = ctx.get_value(self.x)
        type_id = ctx.typeid_of(self.result_var)
        pred = bc.Bounded(lb=self.lower_bound, ub=self.upper_bound)
        return bc.encode_AssumeOp(ctx.builder, type_id, x, pred)


def assume_bounded(x: Var, lower_bound: int | None, upper_bound: int | None) -> Var:
    return add_operation(AssumeBounded, x.get_type(), x=x,
                         lower_bound=lower_bound, upper_bound=upper_bound)


class MakeListView(TypedOperation):
    def __init__(self, base_ptr: Var, length: Var, result_var: Var, loc: Loc):
        super().__init__(
            "make_list_view",
            operands={"base_ptr": base_ptr, "length": length},
            result_vars=[result_var],
            loc=loc
        )

    def generate_bytecode(self, ctx: "BytecodeContext"):
        ty = self.result_var.get_type()
        assert isinstance(ty, ListTy)
        item_size = get_list_item_repr_size_in_words(ty.item_type)
        tv_ty = tensor_view_typeid_for_list(ctx.type_table, item_size)
        pv_tile_shape = 1, get_list_partition_view_tile_size(item_size)
        # On padding value:
        # We intentionally choose to have padding_value Missing, such that
        # reading a list out of bound results in undefined memref
        # A safer choice is to have zero padding, which result in a zero shaped
        # memref which cannot be written to, but we do not want user to rely
        # on the consequence of this specific implementation.
        # Another alternative is to use a different encoding the shape/stride
        # such that zero padding will end up being FFFFF once read back. This way
        # out of bound access of list[array] will result in a memref at 0x0 with 0xFFFF
        # shape and stride, such that when there is accidental write to it, guarantees
        # illegal memory access.
        pv_ty = ctx.type_table.partition_view(pv_tile_shape, tv_ty, [0, 1],
                                              bc.PaddingValue.Missing)
        ptr = ctx.get_value(self.base_ptr)
        length = ctx.get_value(self.length)
        tv = bc.encode_MakeTensorViewOp(ctx.builder, tv_ty, ptr, [length], [])
        return bc.encode_MakePartitionViewOp(ctx.builder, pv_ty, tv)


def flatten_aggregates(vars: Sequence[Var], types: Sequence[Type]) -> tuple[Var, ...]:
    ret = []
    for x, ty in zip(vars, types, strict=True):
        item_types = tuple(ty.flatten_aggregate())
        if isinstance(x.get_type_allow_invalid(), InvalidType):
            ret.extend(x.ctx.make_temp(x.loc, undefined=True) for _ in item_types)
        else:
            items = tuple(x.flatten_aggregate())
            assert len(items) == len(item_types)
            ret.extend(items)
    return tuple(ret)


def flatten_aggregate_types(types: Sequence[Type]) -> tuple[Type, ...]:
    ret = []
    for ty in types:
        ret.extend(ty.flatten_aggregate())
    return tuple(ret)


def unflatten_aggregates(flattened: Tuple[Var, ...],
                         nominal: Sequence[Type], actual: Sequence[Type]) -> tuple[Var, ...]:
    it = iter(flattened)
    ret = tuple(_maybe_unflatten_aggregate(it, n, a) for n, a in zip(nominal, actual, strict=True))
    assert next(it, None) is None
    return ret


def _maybe_unflatten_aggregate(flattened_iter: Iterator[Var], nominal: Type, actual: Type) -> Var:
    if not nominal.is_aggregate():
        return next(flattened_iter)
    return _unflatten_proper_aggregate(flattened_iter, nominal, actual, result_var=None)


def expand_aggregate_var(var: Var) -> Tuple[Var, ...]:
    item_types = tuple(var.get_type().flatten_aggregate())
    ret = tuple(var.ctx.make_var(f"{var.get_original_name()}_{i}", var.loc)
                for i in range(len(item_types)))
    for item, item_ty in zip(ret, item_types, strict=True):
        item.set_type(item_ty)
    return ret


def flatten_block_parameters(vars: Sequence[Var]) -> list[tuple[Var, ...]]:
    ret = []
    for v in vars:
        ty = v.get_type_allow_invalid()
        if ty.is_aggregate():
            flattened_vars = expand_aggregate_var(v)
            ret.append(flattened_vars)
            it = iter(flattened_vars)
            _unflatten_proper_aggregate(it, ty, ty, v)
            assert next(it, None) is None
        else:
            ret.append((v,))
    return ret


def _unflatten_proper_aggregate(flattened_iter: Iterator[Var], nominal: Type, actual: Type,
                                result_var: Var | None) -> Var:
    nominal_item_types = nominal.aggregate_item_types()
    if isinstance(actual, InvalidType):
        # Pop values from the iterator and throw them out
        for _ in nominal_item_types:
            next(flattened_iter)
        # Return an undefined variable. It is OK that we don't create an aggregate value for it --
        # any use of it should be invalid anyway.
        builder = Builder.get_current()
        return builder.ir_crx.make_temp(builder.loc, undefined=True)

    items = tuple(_maybe_unflatten_aggregate(flattened_iter, item_nominal, item_actual)
                  for item_nominal, item_actual
                  in zip(nominal_item_types, actual.aggregate_item_types(), strict=True))
    val = nominal.make_aggregate_value(items)

    builder = Builder.get_current()
    if isinstance(nominal, ArrayTy):
        assert isinstance(val, ArrayValue)
        base_ptr = assume_div_by(val.base_ptr, nominal.base_ptr_div_by)
        shape = tuple(assume_div_by(assume_bounded(x, 0, None), divisor)
                      for x, divisor in zip(val.shape, nominal.shape_div_by, strict=True))

        all_strides = []
        dynamic_strides = []
        for x, s, divisor in zip(val.strides, nominal.strides, nominal.stride_div_by, strict=True):
            if s.maybe_value is None:
                x = assume_div_by(assume_bounded(x, 0, None), divisor)
                dynamic_strides.append(x)
            all_strides.append(x)

        operands = dict(base_ptr=base_ptr, shape=shape, dynamic_strides=tuple(dynamic_strides))
        ret = builder.add_operation(MakeTensorView, nominal, operands, result_var)
        ret.set_aggregate(ArrayValue(base_ptr, shape, tuple(all_strides)))
        return ret
    elif isinstance(nominal, ListTy):
        assert isinstance(val, ListValue)
        operands = dict(base_ptr=val.base_ptr, length=val.length)
        ret = builder.add_operation(MakeListView, nominal, operands, result_var)
        ret.set_aggregate(val)
        return ret
    else:
        return builder.make_aggregate(val, nominal, result_var=result_var)


class TileNumBlocks(TypedOperation):
    def __init__(self, axis: int, result_var: Var, loc: Loc):
        super().__init__(
            "tile_num_blocks",
            operands={},
            attributes={"axis": axis},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        t = ctx.typeid_of(self.result_var)
        return bc.encode_GetNumTileBlocksOp(ctx.builder, t, t, t)[self.axis]


@impl(ct.num_blocks)
def num_blocks(axis: Var) -> Var:
    axis = require_constant_int(axis)
    if axis not in (0, 1, 2):
        raise TileTypeError(f"Axis must be 0, 1, or 2, but {axis} was given.")
    return add_operation(TileNumBlocks, datatype.default_int_type, axis=axis)


def _infer_sliced_shape(
    array_ty: ArrayTy,
    axis: int,
    const_start: Optional[int],
    const_stop: Optional[int],
) -> Tuple[TupleTy, Tuple[Optional[int], ...]]:
    has_const_bounds = const_start is not None and const_stop is not None
    new_axis_size = const_stop - const_start if has_const_bounds else None

    # FIXME: Enable static shape in MakeTensorView for new_axis_size if static
    new_shape = TupleTy(
        SizeTy(None) if i == axis else dim
        for i, dim in enumerate(array_ty.shape)
    )

    # Preserve shape divisibility if new size is compatible
    old_div_by = array_ty.shape_div_by[axis]
    new_div_by = (
        old_div_by
        if (old_div_by is not None
            and new_axis_size is not None
            and new_axis_size % old_div_by == 0)
        else None
    )

    new_shape_div_by = tuple(
        new_div_by if i == axis else d
        for i, d in enumerate(array_ty.shape_div_by)
    )

    return new_shape, new_shape_div_by


def _infer_sliced_base_ptr_alignment(
    array_ty: ArrayTy,
    axis: int,
    const_start: Optional[int],
) -> Optional[int]:
    if array_ty.base_ptr_div_by is None:
        return None

    # Get stride divisibility in elements or use static stride if present
    stride_div_by = array_ty.stride_div_by[axis] or array_ty.strides[axis].maybe_value
    if stride_div_by is None:
        return None

    assert array_ty.dtype.bitwidth % BYTE_BITWIDTH == 0
    dtype_bytewidth = array_ty.dtype.bitwidth // BYTE_BITWIDTH
    stride_div_by_bytes = stride_div_by * dtype_bytewidth
    offset_div_by = (
        const_start * stride_div_by_bytes if const_start is not None
        else stride_div_by_bytes
    )
    return math.gcd(offset_div_by, array_ty.base_ptr_div_by)


@impl(ct._m_array_slice)
def array_slice_impl(array: Var, axis: Var, start: Var, stop: Var) -> Var:
    array_ty = require_array_type(array)
    axis = normalize_axis(require_constant_int(axis), array_ty.ndim)
    require_signed_integer_scalar_or_0d_tile_type(start)
    require_signed_integer_scalar_or_0d_tile_type(stop)

    def maybe_const_int(v: Var):
        if isinstance(v.get_type(), DType) and v.is_constant():
            v_int = v.get_constant()
            assert isinstance(v_int, int)
            return v_int
        return None

    const_start = maybe_const_int(start)
    const_stop = maybe_const_int(stop)
    if const_start is not None and const_start < 0:
        raise TileTypeError("Slice start must be non-negative")
    if const_stop is not None and const_stop < 0:
        raise TileTypeError("Slice stop must be non-negative")
    if const_start is not None and const_stop is not None and const_stop < const_start:
        raise TileTypeError("Slice stop must be greater than or equal to start")

    new_shape_ty, new_shape_div_by = _infer_sliced_shape(array_ty, axis, const_start, const_stop)
    new_base_ptr_div_by = _infer_sliced_base_ptr_alignment(array_ty, axis, const_start)
    new_array_ty = ArrayTy(
        array_ty.dtype,
        shape=new_shape_ty,
        strides=array_ty.strides,
        elements_disjoint=array_ty.elements_disjoint,
        base_ptr_div_by=new_base_ptr_div_by,
        stride_div_by=array_ty.stride_div_by,
        shape_div_by=new_shape_div_by,
    )

    array_val = array.get_aggregate()
    assert isinstance(array_val, ArrayValue)
    static_stride = array_ty.strides[axis].maybe_value
    if static_stride == 1:
        offset = start  # skip multiplication for unit stride
    elif static_stride is not None:
        offset = binary_arithmetic("mul", start, loosely_typed_const(static_stride))
    else:
        offset = binary_arithmetic("mul", start, array_val.strides[axis])

    new_base_ptr = pointer_offset(array_val.base_ptr, astype(offset, datatype.uint64))
    axis_new_shape = astype(binary_arithmetic("sub", stop, start), array_size_type().dtype)
    new_shape = tuple(
        axis_new_shape if i == axis else s for i, s in enumerate(array_val.shape)
    )

    [ret] = unflatten_aggregates(
        (new_base_ptr,) + new_shape + array_val.strides,
        (new_array_ty,), (new_array_ty,)
    )
    return ret


@memory_effect(MemoryEffect.LOAD)
class TileLoad(TypedOperation):
    def __init__(
        self, array: Var, index: tuple[Var, ...], order: Sequence[int],
        padding_mode: PaddingMode, latency: Optional[int], allow_tma: Optional[bool],
        result_var: Var, loc: Loc
    ):
        super().__init__(
            "tile_load",
            operands={"array": array, "index": index},
            attributes={"order": tuple(order), "padding_mode": padding_mode, "latency": latency,
                        "allow_tma": allow_tma},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res, _ = tile_load_generate_bytecode(self, ctx)
        return res


def tile_load(array: Var, index: tuple[Var, ...], shape: Sequence[int], order: Sequence[int],
              padding_mode: PaddingMode, latency: Optional[int], allow_tma: Optional[bool]) -> Var:
    res_ty = make_tile_ty(array.get_type().dtype, shape)
    return add_operation(TileLoad, res_ty,
                         array=array, index=index, order=order,
                         padding_mode=padding_mode, latency=latency, allow_tma=allow_tma)


@impl(ct.load)
def tile_load_impl(array: Var, index: Var, shape: Var, order: Var,
                   padding_mode: Var, latency: Var, allow_tma: Var) -> Var:
    array_ty = require_array_type(array)
    index_ty = require_index_or_index_tuple_type(index)
    index_items = index.get_aggregate().items if isinstance(index_ty, TupleTy) else (index,)

    if array_ty.ndim != len(index_items):
        raise TileTypeError(f"Index size {len(index_items)}"
                            f" does not match the array rank {array_ty.ndim}")

    shape = require_constant_shape(shape, allow_single_int=True, expected_rank=array_ty.ndim,
                                   allow_0d_shape=True)
    broadcasted_shape = (1,) * array_ty.ndim if len(shape) == 0 else shape
    order = require_constant_axis_order(order, array_ty.ndim)
    padding_mode = require_constant_enum(padding_mode, PaddingMode)
    latency = require_optional_constant_int(latency)
    allow_tma = require_optional_constant_bool(allow_tma)
    check_load_store_hints(latency, allow_tma)
    result = tile_load(array, index_items, broadcasted_shape, order, padding_mode, latency,
                       allow_tma)
    return reshape(result, shape)


@memory_effect(MemoryEffect.LOAD)
class TileLoadTokenOrdered(TypedOperation):
    def __init__(
        self, array: Var, index: Var, order: Sequence[int],
            padding_mode: PaddingMode, token: Var, latency: Optional[int],
            allow_tma: Optional[bool],
            result_var: Var, result_token: Var, loc: Loc
    ):
        super().__init__(
            "tile_load_token_ordered",
            operands={"array": array, "index": index,
                      "token": token},
            attributes={"order": order, "padding_mode": padding_mode,
                        "latency": latency, "allow_tma": allow_tma},
            result_vars=[result_var, result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, bc.Value]:
        return tile_load_generate_bytecode(self, ctx)


@memory_effect(MemoryEffect.STORE)
class TileStore(TypedOperation):
    def __init__(self, array: Var, index: tuple[Var, ...], tile: Var, order: Sequence[int],
                 latency: Optional[int], allow_tma: Optional[bool], loc: Loc):
        super().__init__(
            "tile_store",
            operands={"array": array, "index": index, "tile": tile},
            attributes={"order": order, "latency": latency, "allow_tma": allow_tma},
            result_vars=[],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        tile_store_generate_bytecode(self, ctx)
        return ()


def tile_store(array: Var, index: tuple[Var, ...], tile: Var, order: Sequence[int],
               latency: Optional[int], allow_tma: Optional[bool]):
    array_ty = array.get_type()
    ty = require_tile_or_scalar_type(tile)
    tile = astype(tile, array_ty.dtype)
    if isinstance(ty, DType) or ty.ndim == 0:
        tile = reshape(tile, (1,) * array_ty.ndim)
    add_operation(TileStore, (), array=array, index=index, tile=tile, order=order,
                  latency=latency, allow_tma=allow_tma)


def _implicit_cast(src: Var, target_dtype: DType, error_context: str) -> Var:
    ty = require_tile_or_scalar_maybe_loose_type(src)
    try:
        check_implicit_cast(ty, target_dtype)
    except TileTypeError as e:
        raise TileTypeError(f"{error_context}: {str(e)}")
    except TileValueError as e:
        raise TileValueError(f"{error_context}: {str(e)}")
    return astype(src, target_dtype)


@impl(ct.store)
def tile_store_impl(array: Var, index: Var, tile: Var, order: Var,
                    latency: Var, allow_tma: Var):
    array_ty = require_array_type(array)
    index_ty = require_index_or_index_tuple_type(index)
    index_items = index.get_aggregate().items if isinstance(index_ty, TupleTy) else (index,)
    if array_ty.ndim != len(index_items):
        raise TileTypeError(f"Index size {len(index_items)}"
                            f" does not match the array rank {array_ty.ndim}")

    tile = _implicit_cast(tile, array_ty.dtype, "Stored tile is incompatible with array's dtype")

    order = require_constant_axis_order(order, array_ty.ndim)
    latency = require_optional_constant_int(latency)
    allow_tma = require_optional_constant_bool(allow_tma)
    check_load_store_hints(latency, allow_tma)

    tile_store(array, index_items, tile, order, latency, allow_tma)


@memory_effect(MemoryEffect.STORE)
class TileStoreTokenOrdered(TypedOperation):
    def __init__(self, array: Var, index: Var, tile: Var, order: Sequence[int], token: Var,
                 latency: Optional[int], allow_tma: Optional[bool], result_token: Var, loc: Loc):
        super().__init__(
            "tile_store_token_ordered",
            operands={"array": array, "index": index, "tile": tile, "token": token},
            attributes={"order": order, "latency": latency, "allow_tma": allow_tma},
            result_vars=[result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return tile_store_generate_bytecode(self, ctx)


@memory_effect(MemoryEffect.LOAD)
class LoadPointer(TypedOperation):
    def __init__(
        self,
        pointer: Var,
        mask: Optional[Var],
        padding_value: Optional[Var],
        latency: Optional[int],
        result_var: Var,
        loc: Loc,
    ):
        super().__init__(
            "load_pointer",
            operands={"pointer": pointer, "mask": mask, "padding_value": padding_value},
            attributes={"latency": latency},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        lowered_res, _ = load_pointer_lowering(self, ctx)
        return lowered_res


@memory_effect(MemoryEffect.LOAD)
class LoadPointerTokenOrdered(Operation):
    def __init__(
            self,
            pointer: Var,
            mask: Optional[Var],
            padding_value: Optional[Var],
            latency: Optional[int],
            token: Var,
            result_var: Var,
            result_token: Var,
            loc: Loc,
    ):
        super().__init__(
            "load_pointer_tko",
            operands={"pointer": pointer, "mask": mask, "padding_value": padding_value,
                      "token": token},
            attributes={"latency": latency},
            result_vars=[result_var, result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, bc.Value]:
        return load_pointer_lowering(self, ctx)


def load_pointer(pointer: Var, mask: Optional[Var], padding_value: Optional[Var],
                 latency: Optional[int]) -> Var:
    pointer_ty = pointer.get_type()
    shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()
    result_ty = make_tile_ty(get_dtype(pointer_ty).pointee_type, shape)
    return add_operation(LoadPointer, result_ty,
                         pointer=pointer, mask=mask, padding_value=padding_value, latency=latency)


@memory_effect(MemoryEffect.STORE)
class StorePointer(TypedOperation):
    def __init__(
            self,
            pointer: Var,
            value: Var,
            mask: Optional[Var],
            latency: Optional[int],
            loc: Loc,
    ):
        super().__init__(
            "store_pointer",
            operands={"pointer": pointer, "value": value, "mask": mask},
            attributes={"latency": latency},
            result_vars=[],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        store_pointer_lowering(self, ctx)
        return ()


@memory_effect(MemoryEffect.STORE)
class StorePointerTokenOrdered(Operation):
    def __init__(
            self,
            pointer: Var,
            value: Var,
            mask: Optional[Var],
            latency: Optional[int],
            token: Var,
            result_token: Var,
            loc: Loc,
    ):
        super().__init__(
            "store_pointer_tko",
            operands={"pointer": pointer, "value": value, "mask": mask, "token": token},
            attributes={"latency": latency},
            result_vars=[result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return store_pointer_lowering(self, ctx)


def store_pointer(pointer: Var, value: Var, mask: Optional[Var], latency: Optional[int]):
    add_operation(StorePointer, (),
                  pointer=pointer, value=value, mask=mask, latency=latency)


class PointerOffset(TypedOperation):
    def __init__(self, pointer: Var, offset: Var, result_var: Var, loc: Loc):
        super().__init__(
            "pointer_offset",
            operands={"pointer": pointer, "offset": offset},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: "BytecodeContext"):
        res_typeid = ctx.typeid_of(self.result_var)
        pointer = ctx.get_value(self.pointer)
        offset = ctx.get_value(self.offset)
        return bc.encode_OffsetOp(ctx.builder, res_typeid, pointer, offset)


def pointer_offset(pointer: Var, offset: Var) -> Var:
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    offset_ty = offset.get_type()
    offset_shape = offset_ty.shape_value if isinstance(offset_ty, TileTy) else ()

    common_shape = broadcast_shapes2(pointer_shape, offset_shape)
    pointer = broadcast_to(pointer, common_shape)
    offset = broadcast_to(offset, common_shape)
    result_ty = make_tile_ty(get_dtype(pointer_ty), common_shape)
    return add_operation(PointerOffset, result_ty, pointer=pointer, offset=offset)


@impl(ct.gather)
def gather_impl(array: Var, indices: Var, mask: Var, padding_value: Var,
                check_bounds: Var, latency: Var) -> Var:
    pointer, final_mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds, mask)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    # Handle the padding value
    padding_ty = require_tile_or_scalar_type(padding_value)
    padding_shape = padding_ty.shape_value if isinstance(padding_ty, TileTy) else ()
    if not is_shape_broadcastable_to(padding_shape, pointer_shape):
        raise TileTypeError(f"Padding shape {padding_shape} is not broadcastable to the"
                            f" index shape {pointer_ty}")
    array_dtype = array.get_type().dtype

    padding_value = _implicit_cast(padding_value, array_dtype, "Invalid padding value")
    padding_value = broadcast_to(padding_value, pointer_shape)

    # Handle the latency hint
    latency = require_optional_constant_int(latency)
    check_load_store_hints(latency)
    return load_pointer(pointer, final_mask, padding_value, latency)


@impl(ct.scatter)
def scatter_impl(array: Var, indices: Var, value: Var, mask: Var,
                 check_bounds: Var, latency: Var):
    pointer, final_mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds, mask)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    # Handle the `value`
    array_dtype = array.get_type().dtype
    value = _get_scatter_value(value, pointer_shape, array_dtype, "Value")

    # Handle the latency hint
    latency = require_optional_constant_int(latency)
    check_load_store_hints(latency)

    store_pointer(pointer, value, final_mask, latency)


def _get_scatter_value(value: Var, pointer_shape: Tuple[int, ...], array_dtype: DType,
                       value_name: str, cast_dtype: bool = True) -> Var:
    value_ty = require_tile_or_scalar_type(value)
    value_shape = value_ty.shape_value if isinstance(value_ty, TileTy) else ()

    if not is_shape_broadcastable_to(value_shape, pointer_shape):
        raise TileTypeError(f"{value_name} shape {value_shape} is not broadcastable"
                            f" to the index shape {pointer_shape}")

    if cast_dtype:
        value = _implicit_cast(value, array_dtype,
                               "Stored value is incompatible with array's dtype")
    return broadcast_to(value, pointer_shape)


def _process_custom_mask(mask: Optional[Var], bounds_mask: Optional[Var],
                         pointer_shape: Tuple[int, ...]) -> Optional[Var]:
    """
    Process and validate the custom mask parameter for gather/scatter operations.

    Args:
        mask: The user-provided mask (can be Python None or Var containing None)
        bounds_mask: The generated bounds-checking mask based on indices (or None)
        pointer_shape: The target shape that the mask should be broadcast to

    Returns:
        The final mask to use (custom AND bounds, or just one of them, or None)
    """
    # Check if mask is None (either Python None or Var containing None)
    if mask is None or (mask.is_constant() and mask.get_constant() is None):
        # No custom mask provided, return the bounds mask
        return bounds_mask

    # Validate the mask type
    mask_ty = require_tile_or_scalar_type(mask)
    mask_dtype = get_dtype(mask_ty)

    if not is_boolean(mask_dtype):
        raise TileTypeError(f"Custom mask must have boolean dtype, but got {mask_dtype}")

    # Check that mask shape is broadcastable
    mask_shape = mask_ty.shape_value if isinstance(mask_ty, TileTy) else ()
    if not is_shape_broadcastable_to(mask_shape, pointer_shape):
        raise TileTypeError(f"Custom mask shape {mask_shape} is not broadcastable"
                            f" to the index shape {pointer_shape}")

    # Broadcast the mask to the pointer shape
    mask = broadcast_to(mask, pointer_shape)

    # Combine with bounds mask if both exist
    if bounds_mask is None:
        return mask
    else:
        return binary_bitwise("and_", bounds_mask, mask)


def _gather_scatter_pointer_and_mask(
        array: Var,
        indices: Var,
        check_bounds: Var,
        custom_mask: Optional[Var] = None) -> Tuple[Var, Optional[Var]]:
    check_bounds = require_constant_bool(check_bounds)
    array_ty = require_array_type(array)
    indices_ty = require_index_or_index_tuple_type(indices,
                                                   allow_nd_tiles=True, allow_unsigned=True)
    if isinstance(indices_ty, TupleTy):
        index_types = indices_ty.value_types
    else:
        index_types = indices_ty,

    if len(index_types) != array_ty.ndim:
        msg = (f"For array of rank {array_ty.ndim}, `indices` must be a tuple of length"
               f" {array_ty.ndim}")
        if array_ty.ndim == 1:
            msg += ", or a single scalar/tile"
        msg += f". However, `indices` has type {indices_ty}."
        raise TileTypeError(msg)

    # Check that indices are ints
    for dim, indty in enumerate(index_types):
        ind_dtype = get_dtype(indty)
        if not is_integral(ind_dtype):
            for_dim = f"for dimension {dim} " if len(index_types) > 1 else ""
            raise TileTypeError(f"Index {for_dim}has non-integer data type {ind_dtype}")

    # Calculate the common index shape
    index_shapes = [indty.shape_value if isinstance(indty, TileTy) else ()
                    for indty in index_types]
    common_shape = ()
    for shape in index_shapes:
        try:
            common_shape = broadcast_shapes2(common_shape, shape)
        except BroadcastError:
            all_shapes = ", ".join(str(s) for s in index_shapes)
            raise TileTypeError(f"Index shapes {all_shapes}"
                                f" are not broadcastable to a common shape")

    # Calculate offset from indices (and the mask, if check_bounds is True)
    array_val = array.get_aggregate()
    assert isinstance(array_val, ArrayValue)
    offset = None
    mask = None
    for dim in range(len(index_types)):
        if isinstance(indices_ty, TupleTy):
            ind = tuple_item(indices.get_aggregate(), dim)
        else:
            ind = indices

        ind = astype(ind, datatype.uint64)
        ind = broadcast_to(ind, common_shape)

        if check_bounds:
            array_size = array_val.shape[dim]
            array_size = astype(array_size, datatype.uint64)
            dim_mask = comparison("lt", ind, array_size)
            if mask is None:
                mask = dim_mask
            else:
                mask = binary_bitwise("and_", mask, dim_mask)

        static_stride = array_ty.strides[dim].maybe_value
        if static_stride == 1:
            offset_delta = ind
        else:
            if static_stride is None:
                stride = astype(array_val.strides[dim], datatype.uint64)
            else:
                stride = loosely_typed_const(static_stride)
            offset_delta = binary_arithmetic("mul", ind, stride)

        if offset is None:
            offset = offset_delta
        else:
            offset = binary_arithmetic("add", offset, offset_delta)

    # Offset the base pointer
    if offset is None:
        # 0-D array case
        pointer = array_val.base_ptr
        pointer_shape = ()
    else:
        pointer = pointer_offset(array_val.base_ptr, offset)
        pointer_shape = common_shape

    # Process custom mask and combine with bounds mask
    final_mask = _process_custom_mask(custom_mask, mask, pointer_shape)
    return pointer, final_mask


@memory_effect(MemoryEffect.STORE)
class TileAtomicCAS(TypedOperation):
    def __init__(self, pointer: Var, expected: Var, desired: Var,
                 mask: Optional[Var], memory_order: MemoryOrder, memory_scope: MemoryScope,
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_atomic_cas",
            operands={"pointer": pointer, "expected": expected, "desired": desired, "mask": mask},
            attributes={"memory_order": memory_order, "memory_scope": memory_scope},
            result_vars=[result_var],
            loc=loc,
        )

    def generate_bytecode(self, ctx: BytecodeContext):
        result_value, _ = _atomic_cas_generate_bytecode(self, ctx)
        return result_value


@memory_effect(MemoryEffect.STORE)
class TileAtomicCASTokenOrdered(TypedOperation):
    def __init__(self, pointer: Var, expected: Var, desired: Var,
                 mask: Optional[Var], memory_order: Var, memory_scope: Var, token: Var,
                 result_var: Var, result_token: Var, loc: Loc):
        super().__init__(
            "tile_atomic_cas_token_ordered",
            operands={"pointer": pointer, "expected": expected, "desired": desired,
                      "mask": mask, "token": token},
            attributes={"memory_order": memory_order, "memory_scope": memory_scope},
            result_vars=[result_var, result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        return _atomic_cas_generate_bytecode(self, ctx)


def _atomic_cas_generate_bytecode(op: Union[TileAtomicCAS, TileAtomicCASTokenOrdered],
                                  ctx: BytecodeContext) -> tuple[bc.Value, bc.Value]:
    return bc.encode_AtomicCASTkoOp(
        ctx.builder,
        result_type=ctx.typeid_of(op.result_vars[0]),
        result_token_type=ctx.type_table.Token,
        pointers=ctx.get_value(op.pointer),
        cmp=ctx.get_value(op.expected),
        val=ctx.get_value(op.desired),
        mask=None if op.mask is None else ctx.get_value(op.mask),
        token=ctx.get_value(op.token) if isinstance(op, TileAtomicCASTokenOrdered) else None,
        memory_ordering_semantics=memory_order_to_bytecode[op.memory_order],
        memory_scope=memory_scope_to_bytecode[op.memory_scope],
    )


@impl(ct.atomic_cas)
def atomic_cas_impl(array: Var, indices: Var, expected: Var, desired: Var, check_bounds: Var,
                    memory_order: Var, memory_scope: Var) -> Var:
    array_dtype = array.get_type().dtype
    if array_dtype not in int_float_32_64_dtypes:
        raise TileTypeError(f"Unsupported array dtype: {array_dtype}")

    pointer, mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    # Handle the `expected` and `desired` values
    expected = _get_scatter_value(expected, pointer_shape, array_dtype, "Expected value")
    desired = _get_scatter_value(desired, pointer_shape, array_dtype, "Desired value")

    # Handle `memory_order` and `memory_scope`
    memory_order = require_constant_enum(memory_order, MemoryOrder)
    memory_scope = require_constant_enum(memory_scope, MemoryScope)

    result_ty = make_tile_ty(array_dtype, pointer_shape)
    return add_operation(TileAtomicCAS, result_ty,
                         pointer=pointer, expected=expected, desired=desired, mask=mask,
                         memory_order=memory_order, memory_scope=memory_scope)


class AtomicRMWMode(enum.Enum):
    BITWISE_AND = bc.AtomicRMWMode.AND
    BITWISE_OR = bc.AtomicRMWMode.OR
    BITWISE_XOR = bc.AtomicRMWMode.XOR
    ADD_INT = bc.AtomicRMWMode.ADD
    ADD_FLOAT = bc.AtomicRMWMode.ADDF
    MAX_SIGNED_INT = bc.AtomicRMWMode.MAX
    MIN_SIGNED_INT = bc.AtomicRMWMode.MIN
    MAX_UNSIGNED_INT = bc.AtomicRMWMode.UMAX
    MIN_UNSIGNED_INT = bc.AtomicRMWMode.UMIN
    EXCHANGE = bc.AtomicRMWMode.XCHG


@memory_effect(MemoryEffect.STORE)
class TileAtomicRMW(TypedOperation):
    def __init__(self, mode: AtomicRMWMode, pointer: Var, update: Var,
                 mask: Optional[Var], memory_order: MemoryOrder, memory_scope: MemoryScope,
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_atomic_rmw",
            attributes={"mode": mode, "memory_order": memory_order, "memory_scope": memory_scope},
            operands={"pointer": pointer, "update": update, "mask": mask, },
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        result_value, _ = _atomic_rmw_generate_bytecode(self, ctx)
        return result_value


@memory_effect(MemoryEffect.STORE)
class TileAtomicRMWTokenOrdered(TypedOperation):
    def __init__(self, mode: AtomicRMWMode, pointer: Var, update: Var,
                 mask: Optional[Var], memory_order: MemoryOrder, memory_scope: MemoryScope,
                 token: Var, result_var: Var, result_token: Var, loc: Loc):
        super().__init__(
            "tile_atomic_rmw_token_ordered",
            attributes={"mode": mode, "memory_order": memory_order, "memory_scope": memory_scope},
            operands={"pointer": pointer, "update": update, "mask": mask, "token": token},
            result_vars=[result_var, result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        return _atomic_rmw_generate_bytecode(self, ctx)


def _atomic_rmw_generate_bytecode(op: Union[TileAtomicRMW, TileAtomicRMWTokenOrdered],
                                  ctx: BytecodeContext) -> Tuple[bc.Value, bc.Value]:
    return bc.encode_AtomicRMWTkoOp(
        ctx.builder,
        result_type=ctx.typeid_of(op.result_vars[0]),
        result_token_type=ctx.type_table.Token,
        pointers=ctx.get_value(op.pointer),
        arg=ctx.get_value(op.update),
        mask=None if op.mask is None else ctx.get_value(op.mask),
        token=ctx.get_value(op.token) if isinstance(op, TileAtomicRMWTokenOrdered) else None,
        memory_ordering_semantics=memory_order_to_bytecode[op.memory_order],
        memory_scope=memory_scope_to_bytecode[op.memory_scope],
        mode=op.mode._value_
    )


int_32_64_dtypes = (datatype.int32, datatype.int64, datatype.uint32, datatype.uint64)
int_float_32_64_dtypes = (*int_32_64_dtypes, datatype.float32, datatype.float64)


@impl(ct.atomic_xchg, fixed_args=[
    AtomicRMWMode.EXCHANGE, AtomicRMWMode.EXCHANGE, AtomicRMWMode.EXCHANGE,
    False, int_float_32_64_dtypes])
@impl(ct.atomic_add, fixed_args=[
    AtomicRMWMode.ADD_INT, AtomicRMWMode.ADD_INT, AtomicRMWMode.ADD_FLOAT,
    False, (*int_float_32_64_dtypes, datatype.float16)])
@impl(ct.atomic_min, fixed_args=[
    AtomicRMWMode.MIN_SIGNED_INT, AtomicRMWMode.MIN_UNSIGNED_INT, None,
    False, int_32_64_dtypes])
@impl(ct.atomic_max, fixed_args=[
    AtomicRMWMode.MAX_SIGNED_INT, AtomicRMWMode.MAX_UNSIGNED_INT, None,
    False, int_32_64_dtypes])
@impl(ct.atomic_and, fixed_args=[
    AtomicRMWMode.BITWISE_AND, AtomicRMWMode.BITWISE_AND, None,
    True, int_32_64_dtypes])
@impl(ct.atomic_or, fixed_args=[
    AtomicRMWMode.BITWISE_OR, AtomicRMWMode.BITWISE_OR, None,
    True, int_32_64_dtypes])
@impl(ct.atomic_xor, fixed_args=[
    AtomicRMWMode.BITWISE_XOR, AtomicRMWMode.BITWISE_XOR, None,
    True, int_32_64_dtypes])
def atomic_rmw_impl(int_mode: Optional[AtomicRMWMode],
                    uint_mode: Optional[AtomicRMWMode],
                    float_mode: Optional[AtomicRMWMode],
                    bitwise: bool,
                    supported_dtypes: Sequence[DType],
                    # --- end of fixed args ---
                    array: Var, indices: Var, update: Var,
                    check_bounds: Var, memory_order: Var, memory_scope: Var):
    array_dtype = array.get_type().dtype
    if array_dtype not in supported_dtypes:
        raise TileTypeError(f"Unsupported array dtype: {array_dtype}")

    if is_float(array_dtype):
        mode = float_mode
    elif is_integral(array_dtype):
        mode = int_mode if is_signed(array_dtype) else uint_mode
    else:
        mode = None
    assert mode is not None

    pointer, mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    update = _get_scatter_value(update, pointer_shape, array_dtype, "Update",
                                cast_dtype=not bitwise)
    if bitwise:
        update_dtype = get_dtype(update.get_type())
        if update_dtype != array_dtype:
            raise TileTypeError("Bitwise atomic read-modify-write operations require"
                                f" that the update dtype ({update_dtype}) exactly matches"
                                f" the array dtype ({array_dtype})")

    memory_order = require_constant_enum(memory_order, MemoryOrder)
    memory_scope = require_constant_enum(memory_scope, MemoryScope)

    result_ty = make_tile_ty(array_dtype, pointer_shape)
    return add_operation(TileAtomicRMW, result_ty, mode=mode, pointer=pointer, update=update,
                         mask=mask, memory_order=memory_order, memory_scope=memory_scope)


class MakeToken(Operation):
    def __init__(self, result_var: Var, loc: Loc):
        super().__init__(
            "make_token",
            operands={},
            result_vars=[result_var],
            loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return bc.encode_MakeTokenOp(ctx.builder, ctx.type_table.Token)


def make_token(*, block: Block, res: Var, loc: Loc) -> None:
    make_token_op = MakeToken(res, loc)
    block.append(make_token_op)


class JoinTokens(Operation):
    def __init__(self, tokens: Tuple[Var, ...], result_var: Var, loc: Loc):
        super().__init__(
            "join_tokens",
            operands={"tokens": tokens},
            result_vars=[result_var],
            loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        tokens = tuple(ctx.get_value(x) for x in self.tokens)
        return bc.encode_JoinTokensOp(ctx.builder, ctx.type_table.Token, tokens)


def join_tokens(tokens: Tuple[Var, ...], *, block: Block, res: Var, loc: Loc) -> None:
    join_tokens_op = JoinTokens(tokens, res, loc)
    block.append(join_tokens_op)


class NumTiles(TypedOperation):
    def __init__(
        self, array: Var, axis: int, shape: Sequence[int], order: Sequence[int],
            result_var: Var, loc: Loc
    ):
        super().__init__(
            "num_tiles",
            operands={"array": array},
            attributes={"axis": axis, "shape": tuple(shape), "order": tuple(order)},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        pv = ctx.make_partition_view(self.array, self.order, self.shape,
                                     padding_mode=PaddingMode.UNDETERMINED)
        result_types = [ctx.type_table.tile(ctx.type_table.I32, ())] * len(self.shape)
        values = bc.encode_GetIndexSpaceShapeOp(ctx.builder, result_types, pv)
        return values[self.axis]


def num_tiles(array: Var, axis: int, shape: Sequence[int], order: Sequence[int]) -> Var:
    return add_operation(NumTiles, datatype.default_int_type, array=array,
                         axis=axis, shape=shape, order=order)


@impl(ct.num_tiles)
def num_tiles_impl(array: Var, axis: Var, shape: Var, order: Var) -> Var:
    array_ty = require_array_type(array)
    axis = require_constant_int(axis)
    axis = normalize_axis(axis, array_ty.ndim)
    shape = require_constant_shape(shape, allow_single_int=True, expected_rank=array_ty.ndim,
                                   allow_0d_shape=True)
    if len(shape) == 0:
        shape = (1,) * array_ty.ndim
    order = require_constant_axis_order(order, array_ty.ndim)
    return num_tiles(array, axis, shape, order)


def full_const(shape: Sequence[int], fill_value: int | float, dtype: DType) -> Var:
    res_ty = make_tile_ty(dtype, shape)
    return strictly_typed_const(fill_value, res_ty)


def full(shape: Sequence[int], fill_value: Var, dtype: DType) -> Var:
    if fill_value.is_constant():
        return full_const(shape, fill_value.get_constant(), dtype)
    fill_value = astype(fill_value, dtype)
    return broadcast_to(fill_value, shape)


@impl(ct.full)
def full_impl(shape: Var, fill_value: Var, dtype: Var) -> Var:
    require_scalar_or_0d_tile_type(fill_value)
    shape = require_constant_shape(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    return full(shape, fill_value, dtype)


@impl(ct.ones)
def ones_impl(shape: Var, dtype: Var) -> Var:
    shape = require_constant_shape(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    return full_const(shape, 1, dtype)


@impl(ct.zeros)
def zeros_impl(shape: Var, dtype: Var) -> Var:
    shape = require_constant_shape(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    return full_const(shape, 0, dtype)


def _matmul_broadcast_shape(x_shape: TupleTy, y_shape: TupleTy) -> \
        Tuple[TupleTy, TupleTy, TupleTy, TupleTy]:
    x_orig_ndim = len(x_shape)
    y_orig_ndim = len(y_shape)

    # Promote 1D tensors to 2D for matmul
    if x_orig_ndim == 1:
        x_shape = TupleTy((SizeTy(1),) + x_shape.value_types)

    if y_orig_ndim == 1:
        y_shape = TupleTy(y_shape.value_types + (SizeTy(1),))

    if x_shape[-1] != y_shape[-2]:
        raise TileTypeError(f"Incompatible shapes for matrix mul on tiles: {x_shape}, {y_shape}.")

    # Compute result matrix shape
    try:
        batch_shape = datatype.broadcast_shapes(x_shape[:-2], y_shape[:-2])
    except TypeError:
        raise TileTypeError(f"Incompatible shapes for matrix mul on tiles: {x_shape}, {y_shape}.")

    x_shape = TupleTy(batch_shape.value_types + x_shape.value_types[-2:])
    y_shape = TupleTy(batch_shape.value_types + y_shape.value_types[-2:])
    acc_shape = TupleTy(batch_shape.value_types + (x_shape[-2],) + (y_shape[-1],))

    output_shape = acc_shape
    # If x was 1D, squeeze the leading dim
    if x_orig_ndim == 1:
        output_shape = TupleTy(output_shape.value_types[:-2]
                               + output_shape.value_types[-1:])
    # If y was 1D, squeeze the trailing dim
    if y_orig_ndim == 1:
        output_shape = TupleTy(output_shape.value_types[:-1])

    return (x_shape, y_shape, acc_shape, output_shape)


class TileMma(TypedOperation):
    def __init__(self, x: Var, y: Var, acc: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_mma", operands={"x": x, "y": y, "acc": acc}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        y_value = ctx.get_value(self.y)
        acc_value = ctx.get_value(self.acc)
        res_typeid = ctx.typeid_of(self.result_var)

        x_type = ctx.typeof(self.x)
        y_type = ctx.typeof(self.y)
        if datatype.is_integral(x_type.dtype):
            signedness_lhs = datatype.get_signedness(x_type.dtype)
            signedness_rhs = datatype.get_signedness(y_type.dtype)
            return bc.encode_MmaIOp(ctx.builder, res_typeid, x_value, y_value,
                                    acc_value, signedness_lhs, signedness_rhs)
        else:
            return bc.encode_MmaFOp(ctx.builder, res_typeid, x_value, y_value,
                                    acc_value)


@impl(ct.mma)
def mma(x: Var, y: Var, acc: Var) -> Var:
    x_tile_type = require_tile_type(x)
    y_tile_type = require_tile_type(y)
    acc_tile_type = require_tile_type(acc)
    x_shape_orig = x_tile_type.shape
    y_shape_orig = y_tile_type.shape
    acc_shape_orig = acc_tile_type.shape
    if len(x_shape_orig) < 2:
        raise TileTypeError(f'Expect shape of `x` to be at least 2D, got {x_shape_orig}')
    if len(y_shape_orig) < 2:
        raise TileTypeError(f'Expect shape of `y` to be at least 2D, got {y_shape_orig}')
    x_shape, y_shape, _, output_shape = _matmul_broadcast_shape(x_shape_orig, y_shape_orig)
    if acc_shape_orig != output_shape:
        raise TileTypeError(f'Expect acc shape to be {output_shape}, got {acc_shape_orig}')
    datatype._resolve_mma_supported_dtype(x_tile_type.dtype, y_tile_type.dtype, acc_tile_type.dtype)
    x = _promote_and_broadcast_to(x, TileTy(x_tile_type.dtype, x_shape))
    y = _promote_and_broadcast_to(y, TileTy(y_tile_type.dtype, y_shape))
    return add_operation(TileMma, acc_tile_type, x=x, y=y, acc=acc)


@impl(ct.matmul)
@impl(operator.matmul)
def matmul(x: Var, y: Var) -> Var:
    x_tile_type = require_tile_type(x)
    y_tile_type = require_tile_type(y)
    x_shape_orig = x_tile_type.shape
    y_shape_orig = y_tile_type.shape
    x_shape, y_shape, acc_shape, output_shape = _matmul_broadcast_shape(x_shape_orig, y_shape_orig)
    common_dtype = promote_dtypes(x_tile_type.dtype, y_tile_type.dtype)
    acc_dtype = datatype._resolve_mma_supported_dtype(common_dtype, common_dtype, None)
    x = _promote_and_broadcast_to(x, TileTy(common_dtype, x_shape))
    if len(y_shape_orig) == 1:
        # When y is 1d, we cannot directly use cast for reshape + broadcast
        # because y is first reshaped to 2d by appending 1.
        # Therefore, we need to first reshape y from (k,) to (k, 1) and then
        # apply the reshape+broadcast rule for batch dims
        y_shape_2d = (y_shape_orig[0].value, 1)
        y = reshape(y, y_shape_2d)
    y = _promote_and_broadcast_to(y, TileTy(common_dtype, y_shape))
    acc_ty = TileTy(acc_dtype, acc_shape)
    acc_value = strictly_typed_const(0, acc_ty)
    matmul_result = add_operation(TileMma, acc_ty, x=x, y=y, acc=acc_value)
    matmul_result = astype(matmul_result, common_dtype)
    ret = reshape(matmul_result, tuple(dim.value for dim in output_shape))
    return ret


class TileReduce(TypedOperation):
    def __init__(self, xs: tuple[Var, ...], identities: tuple[bool | int | float, ...], axis: int,
                 body: Block, result_vars: tuple[Var, ...], loc: Loc):
        super().__init__(
            "tile_reduce",
            operands={"xs": xs},
            attributes={"identities": identities, "axis": axis},
            nested_blocks=[body],
            result_vars=result_vars,
            loc=loc,
        )

    @property
    def body(self):
        return self.nested_blocks[0]

    @property
    def lhs(self):
        params = self.body.params
        assert len(params) == len(self.xs) * 2
        return params[:len(self.xs)]

    @property
    def rhs(self):
        params = self.body.params
        assert len(params) == len(self.xs) * 2
        return params[len(self.xs):]

    @override
    def _to_string_block_prefixes(self) -> List[str]:
        return ["do"]

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, ...]:
        xs = tuple(ctx.get_value(x) for x in self.xs)
        res_typeids = tuple(ctx.typeid_of(v) for v in self.result_vars)

        identities = []
        param_type_ids = []
        for id_val, x in zip(self.identities, self.xs, strict=True):
            x_dtype = get_dtype(x.get_type())
            x_dtype_id = typeid(ctx.type_table, x_dtype, wrap_scalars=False)
            if datatype.is_float(x_dtype):
                x_dtype_bc = x_dtype._bytecode_type
                attr = bc.Float(float(id_val), x_dtype_bc, ctx.type_table)
            elif datatype.is_boolean(x_dtype):
                attr = bc.Bool(bool(id_val))
            else:
                assert datatype.is_integral(x_dtype)
                attr = bc.Integer(x_dtype_id, x_dtype.bitwidth, int(id_val))
            identities.append(attr)

            x_tile_typeid = ctx.type_table.tile(x_dtype_id, ())
            param_type_ids.append(x_tile_typeid)
            param_type_ids.append(x_tile_typeid)

        nested_builder = bc.encode_ReduceOp(
            ctx.builder,
            result_types=res_typeids,
            operands=xs,
            dim=self.axis,
            identities=identities
        )

        with nested_builder.new_block(param_type_ids) as block_args:
            for var, value in zip(self.body.params, block_args, strict=True):
                ctx.set_value(var, value)
            generate_bytecode_for_block(ctx, self.body)

        return nested_builder.done()


async def raw_reduce(xs: tuple[Var, ...], identities: tuple[bool | int | float], axis: int,
                     body: Callable) -> tuple[Var, ...]:
    builder = Builder.get_current()
    if builder.reduction_body:
        raise TileSyntaxError("Nested reduction is not supported")

    block_params = []
    lhs_vars = []
    rhs_vars = []
    input_shape = ()
    for i, x in enumerate(xs):
        x_ty = x.get_type()
        assert isinstance(x_ty, TileTy)
        if i == 0:
            input_shape = x_ty.shape_value
        else:
            assert input_shape == x_ty.shape_value
        tile_0d_ty = make_tile_ty(x_ty.dtype, ())
        for _ in range(2):
            var = builder.ir_ctx.make_temp(builder.loc)
            var.set_type(tile_0d_ty)
            block_params.append(var)
        lhs_vars.append(block_params[-2])
        rhs_vars.append(block_params[-1])

    assert 0 <= axis < len(input_shape)
    result_shape = input_shape[:axis] + input_shape[axis+1:]
    result_types = tuple(make_tile_ty(x.get_type().dtype, result_shape) for x in xs)

    assert len(xs) == len(identities)

    with nested_block(builder.loc, reduction_body=True) as body_block:
        body_block.params = tuple(block_params)
        body_results = await body(tuple(lhs_vars), tuple(rhs_vars))
        for body_res, x in zip(body_results, xs, strict=True):
            body_res_ty = body_res.get_type()
            if isinstance(body_res_ty, TileTy):
                assert body_res_ty.shape_value == ()
                assert body_res_ty.dtype == x.get_type().dtype
            else:
                assert isinstance(body_res_ty, DType)
                assert body_res_ty == x.get_type().dtype

        add_operation(EndBranch, (), outputs=body_results)

    return add_operation(TileReduce, result_types, xs=xs, identities=identities, axis=axis,
                         body=body_block)


async def reduce(xs: tuple[Var, ...], identities: tuple[bool | int | float, ...],
                 axis: int | None | Iterable[int], keepdims: bool,
                 body: Callable) -> tuple[Var, ...]:
    if len(xs) == 0:
        raise TileTypeError("Need at least one input value to reduce")

    if len(xs) != len(identities):
        raise TileTypeError(f"Number of input values ({len(xs)}) doesn't match the"
                            f" number of identities ({len(identities)})")

    common_input_shape = ()

    x_types = tuple(require_tile_type(x) for x in xs)
    for x_ty in x_types:
        try:
            common_input_shape = broadcast_shapes2(common_input_shape, x_ty.shape_value)
        except BroadcastError:
            all_shapes = ", ".join(str(ty.shape_value) for ty in x_types)
            raise TileTypeError(f"Input shapes {all_shapes}"
                                f" are not broadcastable to a common shape")

    if axis is None:
        axis = tuple(range(len(common_input_shape)))
    else:
        if isinstance(axis, int):
            axis = (axis,)
        axis = sorted(normalize_axis(a, len(common_input_shape)) for a in axis)
        for a1, a2 in zip(axis, axis[1:]):
            if a1 == a2:
                raise TileTypeError(f"Repeated reduction axis {a1}")

    xs = tuple(broadcast_to(x, common_input_shape) for x in xs)
    for i, a in enumerate(axis):
        xs = await raw_reduce(xs, identities, a - i, body)

    result_shape = _get_reduction_shape(common_input_shape, axis, keepdims)
    return tuple(reshape(x, result_shape) for x in xs)


@impl(ct.reduce)
async def reduce_impl(x: Var, axis: Var, func: Var, identity: Var, keepdims: Var) -> Var:
    x_ty = require_tile_or_tile_tuple_type(x)

    # Decide if we have a tuple and unpack the items of `x`
    tuple_mode = isinstance(x_ty, TupleTy)
    if tuple_mode:
        tup_val = x.get_aggregate()
        assert isinstance(tup_val, TupleValue)
        xs = tup_val.items
    else:
        xs = (x,)

    # Parse axis & func
    axis = require_constant_int(axis)
    require_callable_type(func)

    # Parse the identity
    if tuple_mode:
        id_values = require_constant_scalar_tuple(identity)
        if len(id_values) != len(xs):
            raise TileTypeError(f"Number of identity values ({len(id_values)}) must match"
                                f" the number of input tiles ({len(xs)})")
    else:
        id_values = (require_constant_scalar(identity),)

    # Parse keepdims
    keepdims = require_constant_bool(keepdims)

    async def body(lhs, rhs):
        from .._passes.hir2ir import call
        res = await call(func, (*lhs, *rhs), {})
        assert isinstance(res, Var)
        res_ty = res.get_type()
        if tuple_mode:
            if not isinstance(res_ty, TupleTy):
                raise TileTypeError(f"Reduction function returns a value of type"
                                    f" {res_ty}, but a tuple was expected", res.loc)
            if len(res_ty.value_types) != len(xs):
                raise TileTypeError(f"Reduction function must return a tuple of {len(xs)} values"
                                    f" to match the number of inputs, but a tuple of length"
                                    f" {len(res_ty.value_types)} was found.")
            res_tupval = res.get_aggregate()
            assert isinstance(res_tupval, TupleValue)
            results = res_tupval.items
        else:
            results = (res,)

        cast_results = []
        for i, (xi, r) in enumerate(zip(xs, results, strict=True)):
            r_ty = r.get_type()
            extra_ctx = f" at position #{i}" if tuple_mode else ""
            if not isinstance(r_ty, TileTy | DType):
                raise TileTypeError(f"Reduction function returned"
                                    f" a value of non-tile type {r_ty}{extra_ctx}")
            if isinstance(r_ty, TileTy) and r_ty.ndim > 0:
                raise TileTypeError(f"Reduction function returned"
                                    f" a tile of non-scalar shape {r_ty.shape_value}{extra_ctx}")
            error_ctx = f"Reduction function returned a tile of unexpected dtype{extra_ctx}"
            cast_results.append(_implicit_cast(r, xi.get_type().dtype, error_ctx))

        return tuple(cast_results)

    reduced_tiles = await reduce(xs, id_values, axis, keepdims, body)
    if tuple_mode:
        return build_tuple(reduced_tiles)
    else:
        [ret] = reduced_tiles
        return ret


def _get_reduction_shape(shape: Tuple[int, ...],
                         normalized_axis: Tuple[int, ...],
                         keepdims: bool) -> Tuple[int, ...]:
    ret = []
    for i, size in enumerate(shape):
        if i in normalized_axis:
            if keepdims:
                ret.append(1)
        else:
            ret.append(size)
    return tuple(ret)


async def reduce_simple(fn: str, x: Var, axis: int | None | tuple[int, ...], keepdims: bool,
                        rounding_mode: Optional[RoundingMode] = None,
                        flush_to_zero: bool = False) -> Var:
    x_type = require_tile_type(x)
    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, x_type.dtype)

    if datatype.is_boolean(x_type.dtype):
        x = astype(x, datatype.default_int_type)

    match fn:
        case "add": id_val = 0
        case "mul": id_val = 1
        case "min": id_val = _get_min_max(x_type.dtype)[1]
        case "max": id_val = _get_min_max(x_type.dtype)[0]
        case _: assert False

    async def body(lhs: tuple[Var], rhs: tuple[Var]) -> tuple[Var]:
        [lhs], [rhs] = lhs, rhs
        ret = raw_binary_arithmetic(fn, lhs, rhs,
                                    rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)
        return (ret,)

    [ret] = await reduce((x,), (id_val,), axis, keepdims, body)
    return ret


Limits = Tuple[float, float] | Tuple[int, int]


def _get_min_max(dtype: datatype.DType) -> Limits:
    use_float = datatype.is_float(dtype)
    if use_float:
        if dtype in [datatype.float16, datatype.bfloat16, datatype.float32, datatype.float64]:
            return -float("inf"), float("inf")
        else:
            raise NotImplementedError(f"Unsupported float dtype: {dtype}")
    elif datatype.is_signed(dtype):
        return -(1 << (dtype.bitwidth-1)), (1 << (dtype.bitwidth-1)) - 1
    else:
        return 0, (1 << dtype.bitwidth) - 1


def _parse_reduce_axis(axis: Var) -> Optional[tuple[int, ...]]:
    if isinstance(axis.get_type(), TupleTy):
        axis = require_constant_int_tuple(axis)
    else:
        axis = require_optional_constant_int(axis)
        if axis is not None:
            axis = (axis, )
    return axis


@impl(ct.sum, fixed_args=["add"])
@impl(ct.prod, fixed_args=["mul"])
async def reduce_impl_with_rd_and_ftz(fn: str, x: Var, axis: Var, keepdims: Var, rounding_mode: Var,
                                      flush_to_zero: Var) -> Var:
    axis = _parse_reduce_axis(axis)
    keepdims = require_constant_bool(keepdims)
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return await reduce_simple(fn, x, axis, keepdims,
                               rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


@impl(ct.max, fixed_args=["max"])
@impl(ct.min, fixed_args=["min"])
async def reduce_impl_with_ftz(fn: str, x: Var, axis: Var, keepdims: Var,
                               flush_to_zero: Var) -> Var:
    axis = _parse_reduce_axis(axis)
    keepdims = require_constant_bool(keepdims)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return await reduce_simple(fn, x, axis, keepdims, flush_to_zero=flush_to_zero)


async def argmax_argmin(fn: str, x: Var, axis: Optional[int], keepdims: bool) -> Var:
    require_tile_type(x)
    final_shape = None
    if axis is None:
        if keepdims:
            final_shape = (1,) * x.get_type().ndim
            keepdims = False
        x = reshape(x, (-1,))
        axis = 0
    else:
        axis = normalize_axis(axis, x.get_type().ndim)

    if datatype.is_boolean(x.get_type().dtype):
        x = astype(x, datatype.default_int_type)

    x_type = x.get_type()
    indices = arange(x_type.shape_value[axis], datatype.default_int_type)
    indices = reshape(indices, tuple(-1 if i == axis else 1 for i in range(x_type.ndim)))

    match fn:
        case "argmin":
            id_val = _get_min_max(x_type.dtype)[1]
            cmp = "lt"
        case "argmax":
            id_val = _get_min_max(x_type.dtype)[0]
            cmp = "gt"
        case _: assert False

    async def body(lhs: tuple[Var, Var], rhs: tuple[Var, Var]) -> tuple[Var, Var]:
        lhs_val, lhs_idx = lhs
        rhs_val, rhs_idx = rhs
        val_strict = raw_comparison(cmp, lhs_val, rhs_val)
        val_equal = raw_comparison("eq", lhs_val, rhs_val)
        index_lt = raw_comparison("lt", lhs_idx, rhs_idx)
        val_equal_and_index_lt = raw_binary_bitwise("and_", val_equal, index_lt)
        cond = raw_binary_bitwise("or_", val_strict, val_equal_and_index_lt)
        res = raw_where(cond, lhs_val, rhs_val)
        idx = raw_where(cond, lhs_idx, rhs_idx)
        return res, idx

    [_, ret] = await reduce((x, indices), (id_val, 0), axis, keepdims, body)

    if final_shape is not None:
        ret = reshape(ret, final_shape)

    return ret


@impl(ct.argmax, fixed_args=["argmax"])
@impl(ct.argmin, fixed_args=["argmin"])
async def argmax_argmin_impl(fn: str, x: Var, axis: Var, keepdims: Var) -> Var:
    axis = require_optional_constant_int(axis)
    keepdims = require_constant_bool(keepdims)
    return await argmax_argmin(fn, x, axis, keepdims)


class TileScan(TypedOperation):
    def __init__(self, fn: str, x: Var, axis: int, reverse: bool,
                 rounding_mode: Optional[RoundingMode], flush_to_zero: bool,
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_scan",
            operands={"x": x},
            attributes={
                "fn": fn, "axis": axis, "reverse": reverse,
                "rounding_mode": rounding_mode, "flush_to_zero": flush_to_zero,
            },
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        x_type = ctx.typeof(self.x)
        x_value = ctx.get_value(self.x)
        res_type = ctx.typeof(self.result_var)
        return lower_scan(ctx, x_value, x_type, self.axis, self.reverse,
                          res_type, self.fn, self.rounding_mode, self.flush_to_zero)


def scan(fn: str, x: Var, axis: int, reverse: bool, rounding_mode: Optional[RoundingMode] = None,
         flush_to_zero: bool = False) -> Var:
    x_type = require_tile_type(x)
    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, x_type.dtype)
    x_shape = x_type.shape
    axis = normalize_axis(axis, len(x_shape))
    x_dtype = datatype.default_int_type if datatype.is_boolean(x_type.dtype) else x_type.dtype
    x = _promote_and_broadcast_to(x, TileTy(x_dtype, x_shape))
    return add_operation(
        TileScan, TileTy(x_dtype, x_shape),
        fn=fn, x=x, axis=axis, reverse=reverse,
        rounding_mode=rounding_mode, flush_to_zero=flush_to_zero
    )


@impl(ct.cumsum, fixed_args=["add"])
@impl(ct.cumprod, fixed_args=["mul"])
def scan_impl_with_rd_and_ftz(fn: str, x: Var, axis: Var, reverse: Var,
                              rounding_mode: Var, flush_to_zero: Var) -> Var:
    axis = require_constant_int(axis)
    reverse = require_constant_bool(reverse)
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return scan(fn, x, axis, reverse, rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


def expand_dims(x: Var, axis: int) -> Var:
    x_ty = require_tile_type(x)
    axis = normalize_axis(axis, x_ty.ndim + 1)
    old_shape = x_ty.shape_value
    new_shape = (*old_shape[:axis], 1, *old_shape[axis:])
    res_type = make_tile_ty(x_ty.dtype, new_shape)
    return add_operation(TileReshape, res_type, x=x)


@impl(ct.expand_dims)
def expand_dims_impl(x: Var, axis: Var) -> Var:
    axis = require_constant_int(axis)
    return expand_dims(x, axis)


class TileCat(TypedOperation):
    def __init__(self, x: Var, y: Var, axis: int, result_var: Var, loc: Loc):
        super().__init__(
            "tile_cat",
            operands={"x": x, "y": y},
            attributes={"axis": axis},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return_type_id = ctx.typeid_of(self.result_var)
        x_value, y_value = ctx.get_value(self.x), ctx.get_value(self.y)
        return bc.encode_CatOp(ctx.builder, return_type_id, x_value, y_value, self.axis)


def cat(tiles: Var, axis: int) -> Var:
    tuple_ty = require_tuple_type(tiles)
    items = tiles.get_aggregate().items
    if len(items) == 1:
        return items[0]

    if len(tuple_ty) == 0:
        raise TileTypeError("cat() received an empty tuple")
    elif len(items) == 1:
        return items[0]
    elif len(tuple_ty) > 2:
        raise TileTypeError(f"cat() supports at most 2 tiles, got {len(tuple_ty)}")

    x_tile, y_tile = items

    if not isinstance(first_tile := tuple_ty.value_types[0], TileTy):
        raise TileTypeError(f"Expected tuple of Tile, got a {first_tile}")

    dtype = first_tile.dtype
    rank = first_tile.ndim
    shape_value = list(first_tile.shape_value)
    axis = normalize_axis(axis, rank)
    for tile_ty in tuple_ty.value_types[1:]:
        if not isinstance(tile_ty, TileTy):
            raise TileTypeError(f"Expected tuple of Tile, got a {tile_ty}")
        if tile_ty.ndim != rank:
            raise TileTypeError(f"Expected tiles to have the same rank: {rank} != {tile_ty.ndim}")
        if tile_ty.dtype != dtype:
            raise TileTypeError(f"Expected tiles to have the same dtype: {dtype} != {tile_ty.dtype}")  # noqa: E501
        for i, (x, y) in enumerate(zip(shape_value, tile_ty.shape_value, strict=True)):
            if i != axis and x != y:
                raise TileTypeError("Expected tiles to have the same shape "
                                    "for non axis dimensions, "
                                    f"got {tuple(shape_value)} and {tile_ty.shape_value}")
        shape_value[axis] += tile_ty.shape_value[axis]

    if not all(_is_power_of_2(x) for x in shape_value):
        raise TileTypeError(f"Result tile shape must be power of 2, got: {shape_value}")

    res_ty = make_tile_ty(dtype, shape_value)
    return add_operation(TileCat, res_ty, x=x_tile, y=y_tile, axis=axis)


def _is_power_of_2(x: int):
    if x <= 0:
        return False
    return x & (x - 1) == 0


@impl(ct.cat)
def cat_impl(tiles: Var, axis: Var) -> Var:
    const_axis = require_constant_int(axis)
    return cat(tiles, const_axis)


# Does not support broadcasting or type promotion
class RawWhereOperation(TypedOperation):
    def __init__(self, cond, x, y, result_var: Var, loc: Loc):
        super().__init__(
            "raw_where", operands={"cond": cond, "x": x, "y": y}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_typeid = ctx.typeid_of(self.result_var)
        cond = ctx.get_value(self.cond)
        x = ctx.get_value(self.x)
        y = ctx.get_value(self.y)
        return bc.encode_SelectOp(ctx.builder, res_typeid, cond, x, y)


def raw_where(cond: Var, x: Var, y: Var) -> Var:
    ty = x.get_type()
    assert ty == y.get_type()
    assert change_dtype(cond.get_type(), get_dtype(ty)) == ty
    return add_operation(RawWhereOperation, ty, cond=cond, x=x, y=y)


@impl(ct.where)
def where(cond, x, y) -> Var:
    cond_ty = require_tile_or_scalar_maybe_loose_type(cond)
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    xy_ty = promote_types(x_ty, y_ty)
    dtype = get_dtype(xy_ty)
    cond_like_ty = change_dtype(cond_ty, dtype)
    res_ty = promote_types(cond_like_ty, xy_ty)

    cond = _promote_and_broadcast_to(cond, change_dtype(res_ty, datatype.bool_))
    x = _promote_and_broadcast_to(x, res_ty)
    y = _promote_and_broadcast_to(y, res_ty)
    return raw_where(cond, x, y)


@memory_effect(MemoryEffect.STORE)
class TilePrintf(TypedOperation):
    def __init__(self, format: str, args: Tuple[Var, ...], loc: Loc):
        super().__init__("tile_printf",
                         operands={"args": args},
                         attributes={"format": format},
                         result_vars=[],
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        arg_vars = [ctx.get_value(arg) for arg in self.args]
        with tile_mutex("print_mutex", ctx):
            result_typeid = (ctx.type_table.Token
                             if ctx.builder.version >= bc.BytecodeVersion.V_13_2 else None)
            bc.encode_PrintTkoOp(ctx.builder, result_typeid, arg_vars, None, self.format)
        return []


@impl(ct.printf)
def printf_impl(format: Var, args: Tuple[Var, ...]) -> None:
    format_str = require_constant_str(format)
    arg_types = tuple(require_tile_or_scalar_type(x) for x in args)
    parsed_format = PrintfValidator.parse_format(format_str, arg_types)
    add_operation(TilePrintf, (), format=parsed_format, args=args)


@memory_effect(MemoryEffect.STORE)
class TileAssert(TypedOperation):
    def __init__(self, cond: Var, message: str, loc: Loc):
        super().__init__("assert",
                         operands={"cond": cond},
                         attributes={"message": message},
                         result_vars=[], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        bc.encode_AssertOp(ctx.builder, ctx.get_value(self.cond), self.message)
        return []


@impl(ct.assert_)
def assert_impl(cond: Var, message: Var) -> None:
    ty = require_tile_or_scalar_type(cond)
    if get_dtype(ty) != datatype.bool_:
        raise TileTypeError(f"Type of condition must be bool, got {ty}")
    msg_str = require_optional_constant_str(message)
    msg_str = "" if msg_str is None else msg_str
    add_operation(TileAssert, (), cond=cond, message=msg_str)


# Covert a scalar to 0d tile
class ScalarToTile(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__("scalar_to_tile", operands={"x": x}, result_vars=[result_var], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return ctx.get_value(self.x)


def scalar_to_tile(x: Var) -> Var:
    ty = require_scalar_or_0d_tile_type(x)
    if isinstance(ty, TileTy):
        return x
    else:
        res_ty = make_tile_ty(ty, ())
        return add_operation(ScalarToTile, res_ty, x=x)


class TileBroadcast(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__("tile_broadcast", operands={"x": x}, result_vars=[result_var], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        res_typeid = ctx.typeid_of(self.result_var)
        return bc.encode_BroadcastOp(ctx.builder, res_typeid, x_value)


def broadcast_to(x: Var, shape: Sequence[int], *, keep_scalar: bool = False):
    x_ty = require_tile_or_scalar_type(x)
    if isinstance(x_ty, TileTy):
        old_shape = x_ty.shape_value
    else:
        old_shape = ()

    if not is_shape_broadcastable_to(old_shape, shape):
        raise TileTypeError(f"Shape {old_shape} is not broadcastable to {tuple(shape)}")

    if len(shape) > len(old_shape):
        extra_ones = (1,) * (len(shape) - len(old_shape))
        old_shape = extra_ones + old_shape
        x = reshape(x, old_shape)

    if old_shape == shape:
        if len(shape) == 0 and not keep_scalar:
            return scalar_to_tile(x)
        else:
            return x
    else:
        result_ty = make_tile_ty(get_dtype(x_ty), shape)
        return add_operation(TileBroadcast, result_ty, x=x)


@impl(ct.broadcast_to)
def broadcast_to_impl(x: Var, shape: Var) -> Var:
    require_tile_type(x)
    shape = require_constant_shape(shape)
    return broadcast_to(x, shape)


class TileAsType(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_astype", operands={"x": x}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        value = ctx.get_value(self.x)
        return convert_dtype(ctx, value, ctx.typeof(self.x), ctx.typeof(self.result_var))


def astype(x: Var, dtype: DType) -> Var:
    x_ty = require_tile_or_scalar_type(x)
    from_dtype = get_dtype(x_ty)
    if from_dtype == dtype:
        return x

    if x.is_constant():
        val = dtype._py_type(x.get_constant())
        return strictly_typed_const(val, dtype)

    result_ty = dtype if isinstance(x_ty, DType) else make_tile_ty(dtype, x_ty.shape_value)
    return add_operation(TileAsType, result_ty, x=x)


@impl(ct.astype)
def astype_impl(x: Var, dtype: Var) -> Var:
    dtype = require_dtype_spec(dtype)
    return astype(x, dtype)


def tile_astype(x: Var, dtype: Var,
                block: Block, loc: Loc, res: Var) -> None:
    tile_astype_op = TileAsType(x, dtype, res, loc)
    block.append(tile_astype_op)


class TileBitCast(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_bitcast",
            operands={"x": x},
            attributes={},
            result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        value = ctx.get_value(self.x)
        return ctx.bitcast(value, ctx.typeof(self.x), ctx.typeof(self.result_var))


def bitcast(x: Var, dtype: DType) -> Var:
    tile_ty = require_tile_type(x)
    x_dtype = tile_ty.dtype
    if x_dtype.bitwidth != dtype.bitwidth:
        raise TileTypeError(f"Cannot bitcast from {x_dtype} to {dtype}: "
                            f"bit width is different ({x_dtype.bitwidth} vs. {dtype.bitwidth}")
    res_ty = make_tile_ty(dtype, tile_ty.shape_value)
    return add_operation(TileBitCast, res_ty, x=x)


@impl(ct.bitcast)
def bitcast_impl(x: Var, dtype: Var) -> Var:
    dtype_val = require_dtype_spec(dtype)
    return bitcast(x, dtype_val)


class TileArange(TypedOperation):
    def __init__(self, result_var: Var, loc: Loc):
        super().__init__(
            "tile_arange",
            operands={},
            attributes={},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_type = ctx.typeid_of(self.result_var)
        return bc.encode_IotaOp(ctx.builder, res_type)


def arange(size: int, dtype: DType) -> Var:
    if datatype.is_integral(dtype):
        res_ty = make_tile_ty(dtype, (size,))
    else:
        res_ty = make_tile_ty(datatype.default_int_type, (size,))
    res = add_operation(TileArange, res_ty)
    return astype(res, dtype)


@impl(ct.arange)
def arange_impl(size: Var, dtype: Var) -> Var:
    size_val = require_constant_int(size)
    dtype_val = require_dtype_spec(dtype)
    if not _is_power_of_2(size_val):
        raise TileTypeError(f"Result tile shape must be power of 2, got {size_val}")
    return arange(size_val, dtype_val)


class TileReshape(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_reshape", operands={"x": x}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        res_type_id = ctx.typeid_of(self.result_var)
        return bc.encode_ReshapeOp(ctx.builder, res_type_id, x_value)


def reshape(x: Var, new_shape: Tuple[int, ...]) -> Var:
    x_ty = require_tile_or_scalar_type(x)
    x_shape = x_ty.shape_value if isinstance(x_ty, TileTy) else ()
    numel = math.prod(x_shape)

    negative_one_index = None
    numel2 = 1
    for i, dim_value in enumerate(new_shape):
        if dim_value < 0:
            if dim_value < -1:
                raise TileTypeError(f"Dimension can only be -1 or non-negative, got {dim_value}")
            if negative_one_index is not None:
                raise TileTypeError(f"Only one dimension can be -1, got {new_shape}")
            negative_one_index = i
        else:
            numel2 *= dim_value

    if negative_one_index is not None:
        if numel2 == 0 or numel % numel2 != 0:
            raise TileTypeError(f"Cannot reshape {x_shape} to {new_shape}")
        new_shape = list(new_shape)
        new_shape[negative_one_index] = numel // numel2
        new_shape = tuple(new_shape)
    elif numel != numel2:
        raise TileTypeError(f"Cannot reshape {x_shape} to {new_shape}")

    if new_shape == x_shape:
        return x
    else:
        res_type = make_tile_ty(get_dtype(x_ty), new_shape)
        return add_operation(TileReshape, res_type, x=x)


@impl(ct.reshape)
def reshape_impl(x: Var, shape: Var) -> Var:
    require_tile_type(x)
    new_shape = require_constant_int_tuple(shape)
    return reshape(x, new_shape)


class TilePermute(TypedOperation):
    def __init__(self, x: Var, axes: Sequence[int], result_var: Var, loc: Loc):
        super().__init__(
            "tile_permute",
            operands={"x": x},
            attributes={"axes": axes},
            result_vars=[result_var],
            loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        ret_ty_id = ctx.typeid_of(self.result_var)
        x_value = ctx.get_value(self.x)
        return bc.encode_PermuteOp(ctx.builder, ret_ty_id, x_value, self.axes)


def permute(x: Var, axes: Sequence[int]) -> Var:
    ty = require_tile_type(x)
    axes = tuple(normalize_axis(ax, ty.ndim) for ax in axes)
    shape = tuple(ty.shape_value[i] for i in axes)
    result_ty = make_tile_ty(ty.dtype, shape)
    return add_operation(TilePermute, result_ty, x=x, axes=axes)


@impl(ct.permute)
def permute_impl(x: Var, axes: Var) -> Var:
    ty = require_tile_type(x)
    axes_value = require_constant_int_tuple(axes)
    if len(axes_value) != ty.ndim:
        raise TileTypeError(f"Num axes must match input's rank: {len(axes_value)} vs {ty.ndim}")
    seen_axes = set()
    for i, axis in enumerate(axes_value):
        if axis in seen_axes:
            raise TileTypeError(f"Repeated axis #{i}: {axis}")
        seen_axes.add(axis)
    return permute(x, axes_value)


def transpose(x: Var, a0: int, a1: int) -> Var:
    ty = require_tile_type(x)
    axes = list(range(ty.ndim))
    axes[a0], axes[a1] = axes[a1], axes[a0]
    return permute(x, axes)


@impl(ct.transpose)
def transpose_impl(x: Var, axis0: Var, axis1: Var) -> Var:
    ty = require_tile_type(x)
    if ty.ndim < 2:
        raise TileTypeError("Cannot transpose a tile with fewer than 2 dimensions")
    a0 = require_optional_constant_int(axis0)
    a1 = require_optional_constant_int(axis1)

    if (a0 is not None) and (a1 is not None):
        a0 = normalize_axis(a0, ty.ndim)
        a1 = normalize_axis(a1, ty.ndim)
    elif (a0 is None) and (a1 is None):
        if ty.ndim != 2:
            raise TileTypeError("`axes` must be specified for tile with more than 2 dimensions")
        a0 = ty.ndim - 1
        a1 = ty.ndim - 2
    else:
        raise TileTypeError(f"transpose axes must either both be specified or both be None, "
                            f"got axis0={a0}, axis1={a1}")
    return transpose(x, a0, a1)


class TileExtract(TypedOperation):
    def __init__(self, x: Var, index: tuple[Var, ...], shape: Sequence[int],
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_extract",
            operands={"x": x, "index": index},
            attributes={"shape": tuple(shape)},
            result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        index = tuple(ctx.get_value(idx) for idx in self.index)
        res_type_id = ctx.typeid_of(self.result_var)
        return bc.encode_ExtractOp(ctx.builder, res_type_id, x_value, index)


def extract(x: Var, index: tuple[Var, ...], shape: Sequence[int]) -> Var:
    dtype = get_dtype(x.get_type())
    res_ty = make_tile_ty(dtype, shape)
    return add_operation(TileExtract, res_ty, x=x, index=index, shape=shape)


@impl(ct.extract)
def extract_impl(x: Var, index: Var, shape: Var) -> Var:
    x_ty = require_tile_type(x)
    shape = require_constant_shape(shape, expected_rank=x_ty.ndim, allow_single_int=True,
                                   allow_0d_shape=True)
    orig_shape = shape
    if len(shape) == 0:
        shape = (1,) * x_ty.ndim

    index_ty = require_index_or_index_tuple_type(index)
    index_items = index.get_aggregate().items if isinstance(index_ty, TupleTy) else (index,)
    if x_ty.ndim != len(index_items):
        raise TileTypeError(f"Index size {len(index_items)}"
                            f" does not match the tile rank {x_ty.ndim}")

    for i, (s1, s2) in enumerate(zip(x_ty.shape_value, shape, strict=True)):
        if s2 == 0:
            raise TileTypeError(f"Zero shape at dimension #{i}: {shape}")
        if s1 % s2 != 0:
            raise TileTypeError(f"Input shape {x_ty.shape_value} is not divisible by"
                                f" result shape {shape} at dimension #{i}")
    result = extract(x, index_items, shape)
    return reshape(result, orig_shape)


class TileItem(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_item", operands={"x": x},
            result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        x_value = ctx.get_value(self.x)
        x_type_id = ctx.typeid_of(self.x)
        res_type_id = ctx.typeid_of(self.result_var)
        if res_type_id == x_type_id:
            return x_value
        else:
            return bc.encode_ReshapeOp(ctx.builder, res_type_id, x_value)


@impl(ct._m_tile_item)
def tile_item(tile: Var) -> Var:
    x_ty = require_tile_type(tile)
    if x_ty.numel != 1:
        raise TileTypeError(f"Expected a tile of size 1 to get scalar item, "
                            f"got a tile of size {x_ty.numel}")
    return add_operation(TileItem, x_ty.dtype, x=tile)


def store_var(local_idx: int, value: Var, loc: Loc | None = None):
    scope = Scope.get_current()
    new_var = scope.local.redefine(local_idx, loc or Builder.get_current().loc)
    assign(value, new_var)


def store_undefined(local_idx: int, ty: Type, loc: Loc | None = None):
    scope = Scope.get_current()
    new_var = scope.local.redefine(local_idx, loc or Builder.get_current().loc)
    new_var.set_undefined()
    new_var.set_type(ty)


@impl(hir.store_var)
def store_var_impl(name: Var, value: Var):
    name = require_constant_str(name)
    scope = Scope.get_current()
    index = scope.get_local_index(name)
    store_var(index, value)


@impl(hir.load_var)
def load_var_impl(name):
    name = require_constant_str(name)
    scope = Scope.get_current()
    rn: ResolvedName = scope.func_hir.used_names[name]
    if rn.depth >= 0:
        ret = scope.local_scopes[rn.depth][rn.index]
        ret.get_type()  # Trigger an InvalidType check
        if ret.is_undefined():
            raise TileSyntaxError(f"Undefined variable {name} used")
        return ret
    elif rn.index >= 0:
        val = scope.func_hir.frozen_global_values[rn.index]
        val = get_constant_value(val)
        return loosely_typed_const(val)
    else:
        raise TileSyntaxError(f"Undefined variable {name} used")


@impl(hir.make_closure)
def make_closure_impl(func_hir: hir.Function, default_values: tuple[Var, ...]):
    default_value_types = tuple(v.get_type() for v in default_values)

    frozen_captures_by_depth = []
    frozen_capture_types_by_depth = []
    captured_scopes = []

    builder = Builder.get_current()
    scope = Scope.get_current()
    for depth, (local_scope, captured_indices) in (enumerate(
                zip(scope.local_scopes, func_hir.captures_by_depth, strict=True))):
        if local_scope.frozen:
            frozen_vars = tuple(local_scope.get(idx, builder.loc) for idx in captured_indices)
            frozen_captures_by_depth.append(frozen_vars)
            frozen_types = tuple(v.get_type_allow_invalid() for v in frozen_vars)
            frozen_capture_types_by_depth.append(frozen_types)
        else:
            captured_scopes.append(LiveCapturedScope(depth, local_scope))
            frozen_captures_by_depth.append(None)
            frozen_capture_types_by_depth.append(None)

    closure_ty = ClosureTy(func_hir, default_value_types, tuple(captured_scopes),
                           tuple(frozen_capture_types_by_depth))
    closure_val = ClosureValue(default_values, tuple(frozen_captures_by_depth))
    return make_aggregate(closure_val, closure_ty)
