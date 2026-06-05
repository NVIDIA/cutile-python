# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Sequence

from typing_extensions import override

from cuda.tile._datatype import TileTypeError, int64, is_pointer_dtype
from cuda.tile._ir import hir_stubs, hir
from cuda.tile._ir.aggregate_support import flatten_block_parameters, expand_aggregate_var, \
    flatten_aggregate_types, flatten_aggregates, unflatten_aggregates
from cuda.tile._ir.core_ops import store_var, store_invalid
from cuda.tile._ir.ir import Operation, Var, operand, Block, nested_block, Builder, LoopVarState, \
    PhiState, ConstantState, enter_nested_block, add_operation_variadic, add_operation
from cuda.tile._ir.op_impl import ImplRegistry, require_optional_range_type, \
    require_bool_scalar_type
from cuda.tile._ir.scope import Scope, ControlFlowInfo, JumpInfo
from cuda.tile._ir.type import InvalidType, TupleValue, RangeValue, NONE, TokenTy, Type
from cuda.tile._ir2bytecode import BytecodeContext, typeid, generate_bytecode_for_block
import cuda.tile._bytecode as bc


_registry = ImplRegistry()
impl = _registry.impl


def control_flow_impl_registry() -> ImplRegistry:
    return _registry


@dataclass(eq=False)
class Loop(Operation, opcode="loop"):
    start: Var | None = operand()
    stop: Var | None = operand()
    step: Var | None = operand()
    initial_values: tuple[Var, ...] = operand()
    body: Block = nested_block()

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
        initial_values = [ctx.get_value(input_var)
                          for input_var in self.initial_values]
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
    def _to_string_block_prefixes(self) -> list[str]:
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


@impl(hir_stubs.loop)
async def loop_impl(body: hir.Block, iterable: Var):
    from .._passes.hir2ir import dispatch_hir_block, retarget_loc

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
    stored_locals = tuple(sorted(body.stored_indices))
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
        induction_var.set_type(builder.ir_ctx.typing_hooks.get_tensor_like_type(range_ty.dtype, ()))
        scope.hir2ir_varmap[body.params[0].id] = induction_var
        body_params.append(induction_var)

    # Process the loop body
    loop_info = ControlFlowInfo(stored_locals)
    body_loc = retarget_loc(body.loc, scope)
    with enter_nested_block(body_loc) as new_body, scope.change_loop_info(loop_info), \
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
        if not was_valid and is_valid and ty.is_aggregate():
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
    flat_var_types = flatten_aggregate_types(valid_var_types)

    # Update Continue/Break statements
    for jump_info in loop_info.jumps:
        values = tuple(out
                       for out, is_valid in zip(jump_info.outputs, mask, strict=True)
                       if is_valid)
        flat_values = flatten_aggregates(values, valid_var_types)
        # For undefined break/continue value, add a MakeDummy op as its producer
        flat_values = _add_dummy_op_to_invalid_vars(flat_values, flat_var_types)
        assert len(flat_values) == len(all_flattened_body_vars)
        jump_info.jump_op.values = flat_values

    # Create the loop Operation
    valid_initial_values = tuple(v for v, is_valid
                                 in zip(initial_values, mask, strict=True)
                                 if is_valid)
    flat_initial_values = flatten_aggregates(valid_initial_values, valid_var_types)
    # For undefined initial value, add a MakeDummy op as its producer
    flat_initial_values = _add_dummy_op_to_invalid_vars(flat_initial_values, flat_var_types)
    assert len(flat_initial_values) == len(all_flattened_body_vars)

    if range_ty is None:
        start = stop = step = None
    else:
        range_val = iterable.get_aggregate()
        assert isinstance(range_val, RangeValue)
        start, stop, step = range_val.start, range_val.stop, range_val.step
    flat_result_vars = add_operation_variadic(Loop, flat_var_types,
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
            invalid_type = state.result_phi.ty
            if invalid_type is None or not isinstance(invalid_type, InvalidType):
                invalid_type = body_var.get_type_allow_invalid()
            store_invalid(local_idx, invalid_type, state.result_phi.last_loc)

    # Do this check at the end because this may be an automatically inserted loop
    # around the helper function's body.
    if builder.block_restriction is not None:
        builder.block_restriction.validate_operation(Loop)


def _have_nested_jump(calls: Sequence[hir.Call]) -> bool:
    return any(
        block.jump != hir.Jump.END_BRANCH or _have_nested_jump(block.calls)
        for c in calls
        if c.callee is hir_stubs.if_else
        for block in c.args[1:]
    )


@dataclass(eq=False)
class IfElse(Operation, opcode="ifelse"):
    cond: Var = operand()
    then_block: Block = nested_block()
    else_block: Block = nested_block()

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
    def _to_string_block_prefixes(self) -> list[str]:
        return ["then", "else"]

    @override
    def _to_string_rhs(self) -> str:
        return f"if(cond={self.cond})"


@impl(hir_stubs.if_else)
async def if_else_impl(cond: Var, then_block: hir.Block, else_block: hir.Block) -> Var | None:
    from .._passes.hir2ir import dispatch_hir_block, retarget_loc

    require_bool_scalar_type(cond)
    if cond.is_constant():
        branch_taken = then_block if cond.get_constant() else else_block
        return await _flatten_branch(branch_taken)

    builder = Builder.get_current()
    if builder.block_restriction is not None:
        builder.block_restriction.validate_operation(IfElse)

    # Get the total number of results by adding the number of stored variables.
    # Note: we sort the stored variable indices to make the order deterministic.
    scope = Scope.get_current()
    stored_locals = tuple(sorted(then_block.stored_indices | else_block.stored_indices))

    # Convert the "then" branch from HIR to IR
    info = ControlFlowInfo(stored_locals)
    then_loc = retarget_loc(then_block.loc, scope)
    with enter_nested_block(then_loc) as new_then_block, scope.change_if_else_info(info), \
            scope.local.enter_branch():
        await dispatch_hir_block(then_block)

    # If "then" branch doesn't yield, transform our if-else into the following:
    #    if cond:
    #        <then_block>
    #    else:
    #        EndBranch
    #    <else_block>
    # This is to avoid the situation where none of the branches yield.
    else_loc = retarget_loc(else_block.loc, scope)
    if len(info.jumps) == 0:
        info = ControlFlowInfo(())
        with enter_nested_block(else_loc) as new_else_block, scope.change_if_else_info(info), \
                scope.local.enter_branch():
            end_branch(None)
        add_operation_variadic(IfElse, (),
                               cond=cond, then_block=new_then_block, else_block=new_else_block)
        return await _flatten_branch(else_block)

    # Convert the "else" branch from HIR to IR
    with enter_nested_block(else_loc) as new_else_block, scope.change_if_else_info(info), \
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
        jump_info.jump_op.outputs = flat_outputs

    # Generate an IfElse op
    flat_types = flatten_aggregate_types(valid_result_types)
    flat_results = add_operation_variadic(IfElse, flat_types, cond=cond,
                                          then_block=new_then_block, else_block=new_else_block)
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
            assert isinstance(result_phis[0].ty, InvalidType)
            ret = builder.ir_ctx.make_temp(builder.loc)
            ret.set_type(result_phis[0].ty)

    # Update the scope for stored named
    for res_var, phi, local_idx in zip(all_results[num_explicit_results:],
                                       result_phis[num_explicit_results:],
                                       stored_locals, strict=True):
        if res_var is None:
            store_invalid(local_idx, phi.ty, phi.last_loc)
        else:
            store_var(local_idx, res_var, phi.last_loc)

    return ret


@impl(hir_stubs.tuple_comp_if)
async def tuple_comp_if_impl(cond: Var, then_block: hir.Block) -> None:
    require_bool_scalar_type(cond)
    if not cond.is_constant():
        raise TileTypeError("Tuple comprehension if-conditions must be statically known;"
                            " if the condition can be evaluated at compile time,"
                            " wrap it with ct.static_eval()")
    if cond.get_constant():
        await _flatten_branch(then_block)


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


# Maps to ContinueOp in TileIR
@dataclass(eq=False)
class Continue(Operation, opcode="continue", terminator=True):
    values: tuple[Var, ...] = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        next_values = [ctx.get_value(var) for var in self.values]
        bc.encode_ContinueOp(ctx.builder, next_values)
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return f"continue {', '.join([x.name for x in self.values])}"


async def continue_():
    scope = Scope.get_current()
    info = scope.loop_info
    assert info is not None
    assert not info.flatten

    # Why checkpoint? Exit callbacks of context managers may mess with the scope.
    # In particular, the generator-based context manager may rewrite the locals to freeze
    # any closures that reference the context manager's scope (see exit_callback() in
    # enter_context_generator_impl()). So we need to restore the scope back to the original
    # state.
    with scope.checkpoint():
        for ctx_state in reversed(scope.context_stack[scope.loop_context_stack_depth:]):
            await ctx_state.call_exit_callback()

        builder = Builder.get_current()
        builder.add_operation_variadic(Continue, (), dict(values=()))
        op = builder.ops[-1]
        next_values = tuple(scope.local.get(local_idx, builder.loc)
                            for local_idx in info.stored_locals)
        info.jumps.append(JumpInfo(op, next_values))


# Maps to BreakOp
@dataclass(eq=False)
class Break(Operation, opcode="break", terminator=True):
    values: tuple[Var, ...] = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        output_values = [ctx.get_value(var) for var in self.values]
        bc.encode_BreakOp(ctx.builder, output_values)
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return f"break {', '.join([x.name for x in self.values])}"


async def break_():
    scope = Scope.get_current()
    info = scope.loop_info
    assert info is not None

    if info.flatten:
        return

    # See continue_() for the explanation of why checkpoint() is necessary.
    with scope.checkpoint():
        for ctx_state in reversed(scope.context_stack[scope.loop_context_stack_depth:]):
            await ctx_state.call_exit_callback()

        builder = Builder.get_current()
        builder.add_operation_variadic(Break, (), dict(values=()))
        op = builder.ops[-1]
        outputs = tuple(scope.local.get(local_idx, builder.loc) for local_idx in info.stored_locals)
        info.jumps.append(JumpInfo(op, outputs))


# Maps to YieldOp
@dataclass(eq=False)
class EndBranch(Operation, opcode="end_branch", terminator=True):
    outputs: tuple[Var, ...] = operand()

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
        builder.add_operation_variadic(EndBranch, (), dict(outputs=()))
        op = builder.ops[-1]
        outputs += tuple(scope.local.get(local_idx, builder.loc)
                         for local_idx in info.stored_locals)
    info.jumps.append(JumpInfo(op, outputs))


@dataclass(eq=False)
class Return(Operation, opcode="return", terminator=True):

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        bc.encode_ReturnOp(ctx.builder, ())
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return "return"

    @property
    def has_observable_effect(self) -> bool:
        return True


async def return_(value: Var | None):
    if value is not None and value.get_type() is not NONE:
        raise TileTypeError("Tile kernels cannot return values")

    scope = Scope.get_current()
    # See continue_() for the explanation of why checkpoint() is necessary.
    with scope.checkpoint():
        for ctx_state in reversed(scope.context_stack):
            await ctx_state.call_exit_callback()

    add_operation_variadic(Return, ())


@dataclass(eq=False)
class MakeDummy(Operation, opcode="make_dummy"):
    """Placeholder value inserted for undefined variable in loop.

    The use case for undefined variables is to represent loop's
    initial_values or continue/break's next_values during type inference or
    post dead code elimination.
    """

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        ty = ctx.typeof(self.result_var)
        if isinstance(ty, TokenTy):
            return bc.encode_MakeTokenOp(ctx.builder, ctx.type_table.Token)
        from cuda.tile._ir.type import TileTy
        if isinstance(ty, TileTy) and is_pointer_dtype(ty.dtype):
            int_ty = TileTy(dtype=int64, shape=ty.shape)
            const = ctx.constant(0, int_ty)
            return bc.encode_IntToPtrOp(ctx.builder, typeid(ctx.type_table, ty), const)
        return ctx.constant(0, ty)


def _add_dummy_op_to_invalid_vars(vars: Sequence[Var],
                                  actual_types: Sequence[Type]) -> tuple[Var, ...]:
    return tuple(add_operation(MakeDummy, actual)
                 if isinstance(v.get_type_allow_invalid(), InvalidType)
                 else v
                 for v, actual in zip(vars, actual_types, strict=True))
