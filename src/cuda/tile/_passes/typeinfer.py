# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
import inspect
import sys
from collections import defaultdict
from dataclasses import dataclass
from types import FunctionType
from typing import Tuple, Any, Optional, Sequence, Callable

from cuda.tile._debug import CUDA_TILE_LOGS
from cuda.tile._exception import (
    TileTypeError,
    TileInternalError,
    ConstantNotFoundError, TileSyntaxError, Loc, TileError
)
from cuda.tile._ir import ir
from cuda.tile._ir.ir import Operation, Function, Block, Var, Argument, IRContext, TypedOperation
from cuda.tile._ir.op_impl import op_implementations
from cuda.tile._ir.typing_support import get_signature
from cuda.tile._ir.type import FunctionTy, BoundMethodTy, DTypeConstructor, Type, UNDEFINED


class ConstantState(enum.Enum):
    UNSET = 0
    MAY_BE_CONSTANT = 1
    NONCONSTANT = 2


@dataclass
class PhiState:
    constant_state: ConstantState = ConstantState.UNSET
    constant_value: Any = None

    def set_nonconstant(self):
        self.constant_state = ConstantState.NONCONSTANT

    def set_branch_constant(self, value: Any):
        if self.constant_state == ConstantState.UNSET:
            self.constant_state = ConstantState.MAY_BE_CONSTANT
            self.constant_value = value
        elif self.constant_state == ConstantState.MAY_BE_CONSTANT and value != self.constant_value:
            self.constant_state = ConstantState.NONCONSTANT


class TypingContext:
    def __init__(self, ir_ctx: IRContext) -> None:
        self.ir_ctx = ir_ctx
        self.phis = defaultdict(PhiState)

    @property
    def typemap(self):
        return self.ir_ctx.typemap

    @property
    def constants(self):
        return self.ir_ctx.constants

    @property
    def range_infos(self):
        return self.ir_ctx.range_infos

    def get_constant(self, var: Var) -> Any:
        if var.name in self.constants:
            return self.constants[var.name]
        raise ConstantNotFoundError(var.name)

    def try_get_constant(self, var: Var) -> Optional[Any]:
        if var.name in self.constants:
            return self.constants[var.name]
        return None

    def is_constant(self, var: Var) -> bool:
        return var.name in self.constants

    def set_constant(self, var: Var, value: Any):
        if var.name in self.constants:
            raise KeyError(f'Attempt to overwrite existing constant of variable {var.name}')
        self.constants[var.name] = value

    def get_type(self, var: Var) -> Type:
        if var.is_undefined():
            return UNDEFINED
        if var.name in self.typemap:
            return self.typemap[var.name]
        raise KeyError(f"Type for {var.name} not found")

    def set_type(self, var: Var, typ: Type) -> None:
        self.typemap[var.name] = typ

    def phi_propagate_constant(self, src: Var, dst: Var):
        phi = self.phis[dst.name]
        if self.is_constant(src):
            phi.set_branch_constant(self.get_constant(src))
        else:
            phi.set_nonconstant()

    def phi_finalize_constant(self, dst: Var):
        phi = self.phis[dst.name]
        if phi.constant_state == ConstantState.MAY_BE_CONSTANT:
            self.set_constant(dst, phi.constant_value)


def propagate_type(src: Var, dst: Var) -> None:
    # Propagate the type of the variable to the destination variable.
    # Undefined types are propagated to the destination variable.
    existing_ty = dst.try_get_type()
    src_ty = src.get_type()
    if existing_ty is None or src_ty is UNDEFINED:
        dst.set_type(src_ty, force=True)
        dst.set_loose_type(src.get_loose_type(), force=True)
    elif existing_ty is UNDEFINED:
        pass
    elif existing_ty != src.get_type():
        # TODO: better error message to show the variable location.
        raise TileTypeError(f"Types mismatch for {dst.name} in propagation: "
                            f"{existing_ty} != {src.get_type()}")

    # If the loose types don't match exactly, "unify" them to the concrete type
    if dst.get_loose_type() != src.get_loose_type():
        dst.set_loose_type(existing_ty, force=True)


def infer_type(op: Operation, context: TypingContext) -> None:
    try:
        res_types = op.infer_type(context)
    except TypeError as e:
        raise TileTypeError(str(e), op.loc) from e
    if isinstance(res_types, Type):
        res_types = [res_types]
    if len(res_types) != len(op.result_vars):
        raise TileInternalError(f"Number of results mismatch: "
                                f"{len(res_types)} != {len(op.result_vars)}", op.loc)
    for result_var, res_type in zip(op.result_vars, res_types):
        context.set_type(result_var, res_type)


def _flatten_if_else(block: Block, idx: int):
    from cuda.tile._ir.ops import EndBranch, assign_untyped, Continue, Break
    op = block[idx]
    branch_taken = op.then_block if op.cond.get_constant() else op.else_block
    old_ops = branch_taken.detach_all()
    with ir.Builder(block.ctx, op.loc) as ir_builder:
        early_stop = False
        for inner_op in old_ops:
            with ir_builder.change_loc(inner_op.loc):
                if isinstance(inner_op, EndBranch):
                    for result_var, var in zip(op.result_vars, inner_op.outputs):
                        assign_untyped(var, result_var)
                else:
                    if isinstance(inner_op, (Continue, Break)):
                        early_stop = True
                    ir_builder.append_verbatim(inner_op)
    if early_stop:
        del block[idx+1:]
    block[idx:idx+1] = ir_builder.ops


def _flatten_loop(block: Block, idx: int) -> bool:
    from cuda.tile._ir.ops import Loop, Break, assign, assign_untyped

    loop = block[idx]
    if (not isinstance(loop, Loop)
            or loop.for_loop is not None
            or not isinstance(loop.body[-1], Break)
            or _have_break_or_continue(loop.body[:-1])):
        return False

    with ir.Builder(block.ctx, loop.loc) as ir_builder:
        for init_var, body_var in zip(loop.carried_vars.initial, loop.carried_vars.body,
                                      strict=True):
            assign(init_var, body_var)

        *body_ops, brek = loop.body.detach_all()
        ir_builder.extend_verbatim(body_ops)

        for break_res, loop_res in zip(brek.output_vars, loop.carried_vars.results, strict=True):
            assign_untyped(break_res, loop_res)

    block[idx:idx+1] = ir_builder.ops
    return True


def _have_break_or_continue(ops):
    from cuda.tile._ir.ops import Loop, Break, Continue
    return any(
        isinstance(op, (Break, Continue))
        or (not isinstance(op, Loop)
            and any(_have_break_or_continue(block.operations) for block in op.nested_blocks))
        for op in ops
    )


def _bind_args(sig_func, args, kwargs) -> Sequence[Var]:
    from cuda.tile._ir.ops import loosely_typed_const

    sig = get_signature(sig_func)
    try:
        bound_args = sig.bind(*args, **kwargs)
    except TypeError as e:
        raise TileTypeError(f"{sig_func.__name__}(): {e}")
    ret = []
    for name, param in sig.parameters.items():
        if name in bound_args.arguments:
            ret.append(bound_args.arguments[name])
        elif param.kind == param.VAR_POSITIONAL:
            ret.append(())
        else:
            assert param.default is not param.empty
            ret.append(loosely_typed_const(param.default))
    return ret


def _check_recursive_call(call_loc: Loc, callee: Callable):
    while call_loc is not None:
        if call_loc.function is callee:
            raise TileTypeError("Recursive function call detected")
        call_loc = call_loc.call_site


def _replace_call_or_const(block: Block, idx: int):
    from cuda.tile._ir.ops import assign, loosely_typed_const, get_bound_self, Const

    op = block[idx]

    remap_result = True
    with ir.Builder(block.ctx, op.loc) as ir_builder:
        if isinstance(op, Const):
            result = loosely_typed_const(op.value)
        else:
            args = []
            ty = op.func.get_type()
            if isinstance(ty, FunctionTy):
                callee = ty.func
            elif isinstance(ty, BoundMethodTy):
                callee = ty.func
                args.append(get_bound_self(op.func))
            elif isinstance(ty, DTypeConstructor):
                callee = ty.dtype
            else:
                raise TileTypeError(f"Cannot call an object of type {ty}")

            args.extend(op.args)
            arg_list = _bind_args(callee, args, op.kwarg_dict())

            if callee in op_implementations:
                result = op_implementations[callee](*arg_list)
                if result is None:
                    result = loosely_typed_const(None)
                assert isinstance(result, Var)

                if all(result.name != r.name
                       for new_op in ir_builder.ops for r in new_op.result_vars):
                    # If the returned result variable is not produced by any of
                    # the newly created operations, insert an Assign op.
                    #
                    # This mainly happens when an operation implementation reduces to a no-op
                    # by returning its input. For example, reshape(x, new_shape) may return `x`
                    # when the new shape is the same as the old one. So we need to replace
                    # `y = reshape(x, new_shape)` with `y = assign(x)` to make sure `y` is defined.
                    assign(result, op.result_var)
                    remap_result = False
            else:
                # Callee is a user-defined function.
                from cuda.tile._ast2ir import get_function_ir
                _check_recursive_call(op.loc, callee)

                sig = get_signature(callee)
                for param_name, param in sig.parameters.items():
                    if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                                      inspect.Parameter.VAR_KEYWORD):
                        raise TileSyntaxError("Variadic parameters in user-defined"
                                              " functions are not supported")
                callee_function_ir = get_function_ir(callee, block.ctx, call_site=op.loc)
                for arg, param in zip(arg_list, callee_function_ir.parameters):
                    assign(arg, param)
                ir_builder.extend_verbatim(callee_function_ir.root_block.detach_all())
                result = callee_function_ir.return_value

    new_ops = ir_builder.ops
    if remap_result:
        mapper = ir.Mapper(block.ctx, preserve_vars=True)
        mapper.set_var(result, op.result_var)
        block.ctx.copy_type_information(result, op.result_var)
        new_ops = [new_op.clone(mapper) for new_op in new_ops]
    block[idx:idx+1] = new_ops


def infer_types_for_op(context: TypingContext, block: Block, i: int) -> int:
    from cuda.tile._ir.ops import IfElse, Const, Call

    op = block[i]

    if _flatten_loop(block, i):
        return 0

    if isinstance(op, IfElse) and op.cond.is_constant():
        _flatten_if_else(block, i)
        return 0

    if isinstance(op, Call | Const):
        _replace_call_or_const(block, i)
        return 0

    infer_type(op, context)
    return 1


def infer_types_in_block(context: TypingContext, block: Block) -> None:
    i = 0
    while i < len(block):
        op = block[i]
        if isinstance(op, TypedOperation):
            i += 1
            continue

        with op.loc:
            try:
                i += infer_types_for_op(context, block, i)
            except TileError:
                raise
            except Exception as e:
                raise TileInternalError(str(e)) from e


def infer_types_in_func(context: TypingContext, func: Function, args: Tuple[Argument, ...]) -> None:
    if len(args) != len(func.parameters):
        msg = f"Expected {len(func.parameters)} arguments, got {len(args)}"
        raise TileTypeError(msg, func.loc)

    # Initialize the typemap and const map with input args
    for var, arg in zip(func.parameters, args):
        var.set_type(arg.type)
        var.set_loose_type(arg.loose_type)
        if arg.is_const:
            var.set_constant(arg.const_value)

    infer_types_in_block(context, func.root_block)


def infer_types_pass(func: Function, args: Tuple[Argument, ...], pyfunc: FunctionType):
    context = TypingContext(func.root_block.ctx)
    try:
        infer_types_in_func(context, func, args)
    except Exception as e:
        if 'CUTILEIR' in CUDA_TILE_LOGS:
            highlight_loc = e.loc if hasattr(e, 'loc') else None
            code = (f"====Partial CuTile IR for {func}==== \n\n"
                    f"{func.to_string(highlight_loc=highlight_loc)}\n\n")
            print(f'\n{code}', file=sys.stderr)
        raise
