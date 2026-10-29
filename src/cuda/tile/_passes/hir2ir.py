# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import sys
from contextlib import contextmanager
import dataclasses

from enum import Enum
from types import BuiltinFunctionType
from typing import Sequence, Mapping, Callable

from .ast2hir import get_function_hir, HirMode
from .. import TileTypeError
from .._dispatch_mode import StaticEvalMode
from .._exception import Loc, FunctionDesc, TileError, TileRecursionError, \
    TileValueError, UnsupportedCallError, UnsupportedSyntaxError, \
    exception_type_and_str, StaticException
from .._execution import is_function_allowed_in, is_stub, is_static_def, stub_aliases
from .._ir.hir import StaticEvalKind
from .._ir import hir, ir
from .._ir.ir import Var, IRContext, Builder
from .._ir.op_impl import ImplRegistry
from .._ir.core_ops import (
    loosely_typed_const, build_tuple, sym2var, store_var, build_dict, dataclass_new
)
from .._ir.arithmetic_ops import dtype_constructor
from .._ir.scope import Scope, LocalScope
from .._ir.type import FunctionTy, BoundMethodTy, DTypeConstructor, ClosureTy, \
    ClosureDefaultPlaceholder, TypeTy, BoundMethodValue, ClosureValue, var2sym, DataclassTy
from .._ir.typing_support import get_signature, get_dataclass_info, find_method
from cuda.tile._cext import hir_run, HirFrame, HirLabel, HirGeneratorYielded, hir_eval_block

MAX_RECURSION_DEPTH = 1000


def hir2ir(func_hir: hir.Function,
           param_aggregate_vars: Sequence[ir.Var],
           ir_ctx: IRContext):
    # Run as a coroutine using a software stack, so that we don't exceed Python's recursion limit.
    hir_run(_hir2ir_coroutine(func_hir, param_aggregate_vars, ir_ctx), ir_ctx.log_ir_on_error)


async def _hir2ir_coroutine(
    func_hir: hir.Function, param_aggregate_vars: Sequence[ir.Var], ir_ctx: IRContext
):
    scope = _create_scope(func_hir, ir_ctx, call_site=None, parent_scopes=(),
                          concrete_func_desc=func_hir.desc)
    for local_idx, var, param_loc in zip(func_hir.param_local_indices, param_aggregate_vars,
                                         func_hir.param_locs, strict=True):
        scope.local.store_local_variable(local_idx, var, param_loc)

    ir_builder = ir.Builder.get_current()
    with scope.make_current():
        try:
            await hir_eval_block(scope.hir_frame, func_hir.program.start_label())
        except Exception as e:
            if ir_ctx.log_ir_on_error:
                highlight_loc = e.loc if hasattr(e, 'loc') else None
                ir_str = "\n".join(op.to_string(highlight_loc=highlight_loc)
                                   for op in ir_builder.ops)
                print(f"==== Partial cuTile IR ====\n\n{ir_str}\n\n", file=sys.stderr)
            raise


def _create_scope(func_hir: hir.Function, ir_ctx: IRContext, call_site: Loc | None,
                  parent_scopes: tuple[LocalScope, ...],
                  concrete_func_desc: FunctionDesc) -> Scope:
    local_scope = LocalScope(func_hir.local_names, ir_ctx)
    hir_frame = HirFrame(func_hir.program, call_site, concrete_func_desc)
    return Scope(parent_scopes + (local_scope,), None, None, hir_frame, func_hir)


def _create_scope_for_user_defined_helper(callee_hir: hir.Function, builder: ir.Builder,
                                          parent_scopes: tuple[LocalScope, ...] = ()):
    # Create a fresh Scope. Each inlining gets its own concretized
    # FunctionDesc so that DI never merges two specializations whose generated
    # IR might differ.
    return _create_scope(callee_hir, builder.ir_ctx, call_site=builder.loc,
                         parent_scopes=parent_scopes,
                         concrete_func_desc=_concretize_func_desc(callee_hir, builder.ir_ctx))


def _concretize_func_desc(func_hir: hir.Function, ir_ctx: IRContext) -> FunctionDesc:
    # Mint a fresh FunctionDesc whose `specialization_id` makes its synthesized
    # linkage name unique across every inlining.
    return dataclasses.replace(func_hir.desc,
                               specialization_id=ir_ctx.next_function_specialization_id())


def retarget_loc(loc: Loc, scope: Scope) -> Loc:
    # Splice in the scope's call site and, if this loc belongs to the function
    # currently being inlined, swap in the per-specialization FunctionDesc so
    # emitted ops carry the right debug info.
    new_function = loc.function
    if loc.function is scope.func_hir.desc:
        new_function = scope.hir_frame.function_desc
    return loc.retarget(scope.hir_frame.call_site_loc, new_function)


# Optionally, pass the block parameter (e.g. the loop induction variable)
async def dispatch_hir_block(label: HirLabel, *block_parameter):
    scope = Scope.get_current()
    # If the block ends with a jump (e.g. `with cm(): continue`), the jump calls the exit callbacks
    # of the context managers but leaves them on the stack, since they may still be active
    # on other paths. Such context managers are discarded when we restore the stack here.
    old_context_stack = scope.context_stack
    scope.context_stack = list(old_context_stack)
    try:
        return await hir_eval_block(scope.hir_frame, label, *block_parameter)
    finally:
        scope.context_stack = old_context_stack


# Runs the generator function's body up to its `yield` statement.
# Returns the generator's scope and the HirGeneratorYielded, which holds the yielded value
# and can resume the rest of the body (see exit_generator_context_manager()).
async def enter_generator_context_manager(generator_hir: hir.Function,
                                          args: tuple[Var, ...],
                                          kwargs: dict[str, Var]
                                          ) -> tuple[Scope, HirGeneratorYielded]:
    arg_list = _bind_args(generator_hir.signature, generator_hir.desc.name, args, kwargs)
    builder = Builder.get_current()
    _check_recursive_call(builder.loc)
    new_scope = _create_scope_for_user_defined_helper(generator_hir, builder)
    with new_scope.make_current():
        _store_args(generator_hir, arg_list)
        result = await hir_eval_block(new_scope.hir_frame, generator_hir.program.start_label())

    if not isinstance(result, HirGeneratorYielded):
        raise TileTypeError("Expected a generator function,"
                            " but no reachable `yield` statement has been found")
    return new_scope, result


# May be called multiple times, once for each path that exits the context manager
async def exit_generator_context_manager(scope: Scope, yielded: HirGeneratorYielded):
    with scope.make_current():
        await yielded.resume()


# Calls store_var() to bind arguments to parameters
def _store_args(callee_hir: hir.Function, arg_list: list[Var | tuple[Var, ...]]):
    for arg, local_idx, param_loc in zip(arg_list, callee_hir.param_local_indices,
                                         callee_hir.param_locs, strict=True):
        if isinstance(arg, tuple):
            # Handle the *vararg parameter
            arg = build_tuple(arg)
        elif isinstance(arg, dict):
            arg = build_dict(tuple(arg.keys()), tuple(arg.values()))
        store_var(local_idx, arg, param_loc)


async def _call_user_defined(callee_hir: hir.Function,
                             arg_list: list[Var | tuple[Var, ...]],
                             builder: ir.Builder,
                             parent_scopes: tuple[LocalScope, ...] = ()) -> Var:
    _check_recursive_call(builder.loc)

    new_scope = _create_scope_for_user_defined_helper(callee_hir, builder, parent_scopes)
    with new_scope.make_current():
        _store_args(callee_hir, arg_list)
        # The function body ends with a FunctionResult opcode (see _ast2hir())
        retval = await hir_eval_block(new_scope.hir_frame, callee_hir.program.start_label())

    ret = freeze_closures_on_scope_exit(retval, new_scope.local, builder)
    new_scope.local.mark_dead()
    return ret


async def call_function(callee: Callable, *args: Var, **kwargs: Var):
    return await _call_function(callee, args, kwargs, ir.Builder.get_current())


async def _call_function(callee: Callable,
                         args: Sequence[Var],
                         kwargs: Mapping[str, Var],
                         builder: ir.Builder):
    callee = stub_aliases.get(callee, callee)

    execution_space = builder.ir_ctx.execution_space
    if not is_function_allowed_in(
        callee, execution_space,
        device_execution_space=builder.ir_ctx.device_execution_space,
    ):
        code_space = (execution_space if execution_space == "host"
                      else builder.ir_ctx.device_execution_space)
        raise UnsupportedCallError(
            f"{callee.__name__}() is not supported in {code_space} code"
        )

    impl = _try_find_function_impl(callee)
    if impl is not None or is_stub(callee) or isinstance(callee, BuiltinFunctionType):
        if impl is None:
            raise UnsupportedCallError(
                f"{callee.__name__}() is not supported in {execution_space} code"
            )
        return await _call_builtin(callee, impl, args, kwargs, builder)
    elif is_static_def(callee):
        return _call_static_def_function(callee, args, kwargs)
    else:
        try:
            callee_hir = get_function_hir(callee, mode=HirMode.HELPER_FUNCTION)
        except UnsupportedSyntaxError as e:
            e.loc = e.loc.with_call_site(builder.loc)
            raise
        sig = get_signature(callee)
        arg_list = _bind_args(sig, callee.__name__, args, kwargs)
        return await _call_user_defined(callee_hir, arg_list, builder)


def _call_static_def_function(callee, args, kwargs):
    with StaticEvalMode(StaticEvalKind.STATIC_DEF).as_current():
        args_sym = tuple(var2sym(x) for x in args)
        kwargs_sym = {k: var2sym(v) for k, v in kwargs.items()}
        try:
            res_sym = callee(*args_sym, **kwargs_sym)
        except TileError:
            raise
        except Exception as e:
            msg = f"Exception was raised inside @static_def function ({exception_type_and_str(e)})"
            static_exc = StaticException(msg)
            static_exc._where = "inside @static_def"
            raise static_exc from e

        return sym2var(res_sym)


async def _call_builtin(callee, impl, args, kwargs, builder: ir.Builder):
    sig = get_signature(callee)
    arg_list = _bind_args(sig, callee.__name__, args, kwargs)

    result = impl(*arg_list)
    if impl._is_coroutine:
        result = await result

    # Map the result variable
    if result is None:
        result = loosely_typed_const(None)
    assert isinstance(result, Var), result
    return result


def _try_find_function_impl(callee):
    impl_registry = ImplRegistry.get_current()
    try:
        return impl_registry.op_implementations[callee]
    except KeyError:
        custom_handler = getattr(callee, "_cutile_custom_implementation_handler", None)
        if custom_handler is None:
            return None

        if custom_handler._is_coroutine:
            async def custom_impl_wrapper(*args):
                return await custom_handler(callee, *args)
        else:
            def custom_impl_wrapper(*args):
                return custom_handler(callee, *args)

        custom_impl_wrapper._is_coroutine = custom_handler._is_coroutine
        return custom_impl_wrapper


_DTYPE_CONSTRUCTOR_SIGNATURE = inspect.signature(lambda x=0, /: None)


# Implementation of GenericCall HIR opcode
# Sets the builder's location to the given HIR location, retargeted to the current scope.
# Used by the Python implementations of HIR opcodes.
@contextmanager
def change_hir_loc(loc: Loc):
    builder = Builder.get_current()
    with builder.change_loc(retarget_loc(loc, Scope.get_current())):
        yield


async def generic_call(callee_var: Var, args, kwargs, loc: Loc) -> Var | None:
    with change_hir_loc(loc):
        return await call(callee_var, args, kwargs)


async def call(callee_var: Var, args, kwargs) -> Var | None:
    builder = ir.Builder.get_current()
    callee_ty = callee_var.get_type()
    if isinstance(callee_ty, FunctionTy):
        return await _call_function(callee_ty.func, args, kwargs, builder)
    elif isinstance(callee_ty, BoundMethodTy):
        bound_method = callee_var.get_aggregate()
        assert isinstance(bound_method, BoundMethodValue)
        return await _call_function(callee_ty.func, (bound_method.bound_self, *args), kwargs,
                                    builder)
    elif isinstance(callee_ty, DTypeConstructor):
        arg_list = _bind_args(_DTYPE_CONSTRUCTOR_SIGNATURE, callee_ty.dtype.name, args, kwargs)
        [x] = arg_list
        return dtype_constructor(callee_ty.dtype, x)
    elif isinstance(callee_ty, TypeTy):
        return await _call_constructor(callee_ty.ty, args, kwargs, builder)
    elif isinstance(callee_ty, ClosureTy):
        func_name = callee_ty.func_hir.desc.name
        if func_name is None:
            func_name = callee_ty.func_hir.desc.short_str()
        arg_list = _bind_args(callee_ty.func_hir.signature, func_name, args, kwargs,
                              callee_var.get_aggregate().default_values)
        parent_scopes = _get_closure_parent_scopes(callee_ty, callee_var.get_aggregate(),
                                                   builder.ir_ctx)
        return await _call_user_defined(callee_ty.func_hir, arg_list, builder,
                                        parent_scopes=parent_scopes)
    elif (isinstance(callee_ty, DataclassTy)
          and (call_dunder := find_method(callee_ty.cls, "__call__")) is not NotImplemented):
        return await _call_function(call_dunder, (callee_var, *args), kwargs, builder)
    else:
        raise TileTypeError(f"Cannot call an object of type {callee_ty}")


async def _call_constructor(ty, args, kwargs, builder):
    if dataclasses.is_dataclass(ty):
        dataclass_info = get_dataclass_info(ty)
        return await dataclass_new(dataclass_info, args, kwargs)
    elif issubclass(ty, Enum):
        if len(args) != 1 or kwargs:
            raise TileTypeError("Enum constructor takes exactly one positional argument")
        arg = args[0]
        if not arg.is_constant():
            raise TileTypeError("Enum constructor argument must be a constant")
        val = arg.get_constant()
        try:
            return loosely_typed_const(ty(val))
        except ValueError:
            raise TileValueError(f"{val!r} is not a valid {ty.__name__}")
    else:
        impl = _try_find_function_impl(ty)
        if impl is None:
            raise UnsupportedCallError(f'Creating instances of type "{ty.__name__}"'
                                       f' is not supported in '
                                       f'{builder.ir_ctx.execution_space} code')
        return await _call_builtin(ty, impl, args, kwargs, builder)


def _get_closure_parent_scopes(ty: ClosureTy, val: ClosureValue,
                               ir_ctx: IRContext) -> tuple[LocalScope, ...]:
    ret: list[LocalScope | None] = [None for _ in ty.frozen_capture_types_by_depth]
    for live_scope in ty.captured_scopes:
        ret[live_scope.depth] = live_scope.local_scope

    for depth, (local_names, frozen_local_indices, frozen_vars) in enumerate(
                zip(ty.func_hir.enclosing_local_names,
                    ty.func_hir.captures_by_depth,
                    val.frozen_captures_by_depth,
                    strict=True)):
        # Scope at this depth is either live or frozen (mutually exclusive)
        assert (frozen_vars is None) != (ret[depth] is None)
        if frozen_vars is not None:
            ret[depth] = LocalScope.create_frozen(local_names, frozen_local_indices,
                                                  frozen_vars, ir_ctx)
    return tuple(ret)


def freeze_closures_on_scope_exit(retval: Var, dead_scope: LocalScope, builder: ir.Builder) -> Var:
    ty = retval.get_type_allow_invalid()
    if not ty.is_aggregate():
        return retval

    if isinstance(ty, ClosureTy):
        retval = _freeze_closure(retval, dead_scope, builder)
        ty = retval.get_type()

    old_items = retval.get_aggregate().as_tuple()
    new_items = tuple(freeze_closures_on_scope_exit(x, dead_scope, builder) for x in old_items)
    if any(old is not new for old, new in zip(old_items, new_items, strict=True)):
        new_agg_val = ty.make_aggregate_value(new_items)
        retval = builder.make_aggregate(new_agg_val, ty)

    return retval


def _freeze_closure(retval: Var, dead_scope: LocalScope, builder: ir.Builder) -> Var:
    ty = retval.get_type_allow_invalid()
    assert isinstance(ty, ClosureTy)

    if len(ty.captured_scopes) == 0 or ty.captured_scopes[-1].local_scope is not dead_scope:
        # For example:
        #
        #    def kernel():
        #        def f1():
        #            ...
        #        def f2(x):
        #            return x  # <--at this return
        #        f2(f1)
        #
        # Note that for f1, `ty.captured_scopes[-1].local_scope` is the live scope of `kernel()`.
        # But when we return from `f2()`, the `callee_scope` is the scope of `f2`, so there
        # is nothing to freeze in this case.
        return retval

    closure_val = retval.get_aggregate()
    assert isinstance(closure_val, ClosureValue)

    depth = ty.captured_scopes[-1].depth
    frozen_locals = ty.func_hir.captures_by_depth[depth]
    frozen_captures = tuple(dead_scope.get(idx, builder.loc).var for idx in frozen_locals)
    frozen_capture_types = tuple(v.get_type_allow_invalid() for v in frozen_captures)

    new_closure_val = ClosureValue(
        default_values=closure_val.default_values,
        frozen_captures_by_depth=_replace_tuple_item(
            closure_val.frozen_captures_by_depth, depth, frozen_captures)
    )
    new_ty = ClosureTy(
        func_hir=ty.func_hir,
        default_value_types=ty.default_value_types,
        captured_scopes=ty.captured_scopes[:-1],
        frozen_capture_types_by_depth=_replace_tuple_item(
            ty.frozen_capture_types_by_depth, depth, frozen_capture_types),
    )
    return builder.make_aggregate(new_closure_val, new_ty)


def _replace_tuple_item(tup, idx, val):
    return tup[:idx] + (val,) + tup[idx+1:]


def _check_recursive_call(call_loc: Loc):
    depth = 1
    while call_loc is not None:
        depth += 1
        call_loc = call_loc.call_site
    if depth > MAX_RECURSION_DEPTH:
        raise TileRecursionError(f"Maximum recursion depth ({MAX_RECURSION_DEPTH}) reached"
                                 f" while inlining a function call")


def _bind_args(sig: inspect.Signature, func_name: str, args, kwargs,
               closure_defaults: Sequence[Var] | None = None) -> list[Var | tuple[Var, ...]]:
    try:
        bound_args = sig.bind(*args, **kwargs)
    except TypeError as e:
        raise TileTypeError(f"{func_name}(): {e}")
    ret = []
    for name, param in sig.parameters.items():
        if name in bound_args.arguments:
            ret.append(bound_args.arguments[name])
        elif param.kind == param.VAR_POSITIONAL:
            ret.append(())
        elif param.kind == param.VAR_KEYWORD:
            ret.append({})
        else:
            assert param.default is not param.empty
            if isinstance(param.default, ClosureDefaultPlaceholder):
                assert closure_defaults is not None
                default = closure_defaults[param.default.default_value_index]
            else:
                default = sym2var(param.default, constant_only=True)
            ret.append(default)
    return ret
