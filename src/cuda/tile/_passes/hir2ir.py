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

from .ast2hir import get_function_hir
from .. import TileTypeError
from .._coroutine_util import resume_after, run_coroutine
from .._dispatch_mode import StaticEvalMode
from .._exception import Loc, FunctionDesc, TileInternalError, TileError, TileRecursionError, \
    TileValueError, UnsupportedCallError, TypeCheckingError
from .._execution import is_stub, is_static_def
from .._ir import hir, ir
from .._ir.hir import StaticEvalKind
from .._ir.ir import Var, IRContext
from .._ir.op_impl import ImplRegistry
from .._ir.control_flow_ops import end_branch, return_, continue_, break_
from .._ir.core_ops import (
    loosely_typed_const, build_dataclass_instance, build_tuple, sym2var, store_var, build_dict
)
from .._ir.arithmetic_ops import dtype_constructor
from .._ir.scope import Scope, LocalScope, IntMap
from .._ir.type import FunctionTy, BoundMethodTy, DTypeConstructor, ClosureTy, \
    ClosureDefaultPlaceholder, StringFormat, TypeTy, TupleTy, BoundMethodValue, TupleValue, \
    ClosureValue, DictTy, DictValue, var2sym, DataclassTy
from .._ir.typing_support import get_signature, get_dataclass_info, find_method

MAX_RECURSION_DEPTH = 1000


def hir2ir(func_hir: hir.Function,
           param_aggregate_vars: Sequence[ir.Var],
           ir_ctx: IRContext):
    # Run as a coroutine using a software stack, so that we don't exceed Python's recursion limit.
    run_coroutine(_hir2ir_coroutine(func_hir, param_aggregate_vars, ir_ctx))


async def _hir2ir_coroutine(
    func_hir: hir.Function, param_aggregate_vars: Sequence[ir.Var], ir_ctx: IRContext
):
    scope = _create_scope(func_hir, ir_ctx, call_site=None, parent_scopes=(),
                          concrete_func_desc=func_hir.desc)
    for local_idx, var in zip(func_hir.param_local_indices, param_aggregate_vars, strict=True):
        scope.local[local_idx] = var

    ir_builder = ir.Builder.get_current()
    with scope.make_current():
        try:
            await _dispatch_hir_block_inner(func_hir.body, ir_builder)
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
    return Scope(parent_scopes + (local_scope,), None, None, call_site, IntMap(), func_hir,
                 concrete_func_desc=concrete_func_desc)


def _concretize_func_desc(func_hir: hir.Function, ir_ctx: IRContext) -> FunctionDesc:
    # Mint a fresh FunctionDesc whose `specialization_id` makes its synthesized
    # linkage name unique across every inlining.
    return dataclasses.replace(func_hir.desc,
                               specialization_id=ir_ctx.next_function_specialization_id())


def retarget_loc(loc: Loc, scope: Scope) -> Loc:
    if loc.is_unknown():
        return loc

    # Splice in the scope's call site and, if this loc belongs to the function
    # currently being inlined, swap in the per-specialization FunctionDesc so
    # emitted ops carry the right debug info.
    new_function = loc.function
    if loc.function is scope.func_hir.desc:
        new_function = scope.concrete_func_desc
    return dataclasses.replace(loc, function=new_function, call_site=scope.call_site)


async def dispatch_hir_block(block: hir.Block, cur_builder: ir.Builder | None = None):
    if cur_builder is None:
        cur_builder = ir.Builder.get_current()
    await _dispatch_hir_block_inner(block, cur_builder)


async def _dispatch_hir_block_inner(block: hir.Block, builder: ir.Builder):
    cursor = 0  # Pre-initialize to guarantee it's defined in the `except` block
    try:
        scope = Scope.get_current()
        for cursor, call in enumerate(block.calls):
            loc = retarget_loc(call.loc, scope)
            with _wrap_exceptions(loc), builder.change_loc(loc):
                await _dispatch_call(call, scope)
            if builder.is_terminated:
                # The current block has been terminated, e.g. by flattening an if-else
                # with a constant condition (`if True: break`).
                return
        cursor = len(block.calls)

        loc = retarget_loc(block.jump_loc, scope)
        with _wrap_exceptions(loc), builder.change_loc(loc):
            _dispatch_hir_jump(block, scope)
    except Exception:
        if builder.ir_ctx.log_ir_on_error:
            hir_params = ", ".join(str(p) for p in block.params)
            hir_lines = [str(c) for c in block.calls]
            if block.jump is not None:
                hir_lines.append(block.jump_str())
            hir_str = "\n".join("{}{}".format("--> " if i == cursor else "    ", c)
                                for i, c in enumerate(hir_lines))
            print(f"==== HIR for ^{block.block_id}({hir_params}) ====\n{hir_str}\n",
                  file=sys.stderr)
        raise


def _dispatch_hir_jump(block: hir.Block, scope: Scope):
    match block.jump:
        case hir.Jump.END_BRANCH:
            end_branch(_resolve_operand(block.result, scope) if block.have_result else None)
        case hir.Jump.CONTINUE:
            assert not block.have_result
            continue_()
        case hir.Jump.BREAK:
            assert not block.have_result
            break_()
        case hir.Jump.RETURN:
            return_(_resolve_operand(block.result, scope) if block.have_result else None)
        case None: pass
        case _: assert False


@contextmanager
def _wrap_exceptions(loc: Loc):
    with loc:
        try:
            yield
        except TileError:
            raise
        except Exception as e:
            raise TileInternalError(str(e)) from e


async def _dispatch_call(hir_call: hir.Call, scope: Scope):
    callee_var = _resolve_operand(hir_call.callee, scope)
    args = []
    for x in hir_call.args:
        if isinstance(x, hir.Starred):
            tup_var = _resolve_operand(x.value, scope)
            assert isinstance(tup_var, Var)
            tup_ty = tup_var.get_type()
            if not isinstance(tup_ty, TupleTy):
                raise TileTypeError(f"Expected a tuple after *, got {tup_ty}")
            tup_value = tup_var.get_aggregate()
            assert isinstance(tup_value, TupleValue)
            args.extend(tup_value.items)
        else:
            args.append(_resolve_operand(x, scope))
    kwargs = {}
    for k, v in hir_call.kwargs:
        resolved_val = _resolve_operand(v, scope)
        if k is None:
            assert isinstance(resolved_val, Var)
            dict_ty = resolved_val.get_type()
            if not isinstance(dict_ty, DictTy):
                raise TileTypeError(f"Expected a dictionary after **, got {dict_ty}")
            dict_value = resolved_val.get_aggregate()
            assert isinstance(dict_value, DictValue)
            for item_key, item_value in zip(dict_ty.keys, dict_value.values, strict=True):
                kwargs[item_key] = item_value
        else:
            kwargs[k] = resolved_val
    retval = await call(callee_var, args, kwargs)
    if hir_call.result is not None and retval is not None:
        scope.hir2ir_varmap[hir_call.result.id] = retval


async def _call_user_defined(callee_hir: hir.Function,
                             arg_list: list[Var | tuple[Var, ...]],
                             builder: ir.Builder,
                             parent_scopes: tuple[LocalScope, ...] = ()):
    _check_recursive_call(builder.loc)

    # Activate a fresh Scope. Each inlining gets its own concretized
    # FunctionDesc so that DI never merges two specializations whose generated
    # IR might differ.
    new_scope = _create_scope(callee_hir, builder.ir_ctx, call_site=builder.loc,
                              parent_scopes=parent_scopes,
                              concrete_func_desc=_concretize_func_desc(callee_hir,
                                                                       builder.ir_ctx))
    with new_scope.make_current():
        # Call store_var() to bind arguments to parameters.
        for arg, local_idx, param_loc in zip(arg_list, callee_hir.param_local_indices,
                                             callee_hir.param_locs, strict=True):
            if isinstance(arg, tuple):
                # Handle the *vararg parameter
                arg = build_tuple(arg)
            elif isinstance(arg, dict):
                arg = build_dict(tuple(arg.keys()), tuple(arg.values()))
            store_var(local_idx, arg, param_loc)

        # Dispatch the function body. Use resume_after() to break the call stack
        # and make sure we stay within the Python's recursion limit.
        await resume_after(dispatch_hir_block(callee_hir.body, builder))

    assert callee_hir.body.have_result
    ret = _process_return_value(
            new_scope.hir2ir_varmap[callee_hir.body.result.id], new_scope.local, builder)
    new_scope.local.mark_dead()
    return ret


async def call_function(callee: Callable, *args: Var, **kwargs: Var):
    return await _call_function(callee, args, kwargs, ir.Builder.get_current())


async def _call_function(callee: Callable,
                         args: Sequence[Var],
                         kwargs: Mapping[str, Var],
                         builder: ir.Builder):
    impl = _try_find_function_impl(callee)
    if impl is not None or is_stub(callee) or isinstance(callee, BuiltinFunctionType):
        if impl is None:
            raise UnsupportedCallError(f"{callee.__name__}() is not supported in device code")
        return await _call_builtin(callee, impl, args, kwargs, builder)
    elif is_static_def(callee):
        return _call_static_def_function(callee, args, kwargs)
    else:
        callee_hir = get_function_hir(callee, entry_point=False)
        sig = get_signature(callee)
        arg_list = _bind_args(sig, callee.__name__, args, kwargs)
        return await _call_user_defined(callee_hir, arg_list, builder)


def _call_static_def_function(callee, args, kwargs):
    with StaticEvalMode(StaticEvalKind.STATIC_DEF).as_current():
        args_sym = tuple(var2sym(x) for x in args)
        kwargs_sym = {k: var2sym(v) for k, v in kwargs.items()}
        res_sym = callee(*args_sym, **kwargs_sym)
        return sym2var(res_sym)


async def _call_builtin(callee, impl, args, kwargs, builder: ir.Builder):
    sig = get_signature(callee)
    arg_list = _bind_args(sig, callee.__name__, args, kwargs)

    result = impl(*arg_list)
    if impl._is_coroutine:
        result = await result

    if builder.is_terminated:
        # The current block has been terminated, e.g. by flattening an if-else
        # with a constant condition (`if True: break`). Ignore the `result` in this case.
        return None

    # Map the result variable
    if result is None:
        result = loosely_typed_const(None)
    assert isinstance(result, Var)
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
        return await _call_user_defined(callee_ty.func_hir, arg_list, builder, parent_scopes)
    elif (isinstance(callee_ty, DataclassTy)
          and (call_dunder := find_method(callee_ty.cls, "__call__")) is not NotImplemented):
        return await _call_function(call_dunder, (callee_var, *args), kwargs, builder)
    else:
        raise TileTypeError(f"Cannot call an object of type {callee_ty}")


async def _call_constructor(ty, args, kwargs, builder):
    if dataclasses.is_dataclass(ty):
        dataclass_info = get_dataclass_info(ty)
        if dataclass_info.init_signature is None:
            if is_static_def(ty.__init__):
                return _call_static_def_function(ty, args, kwargs)

            raise TypeCheckingError("Dataclass instance creation is only supported for dataclasses"
                                    " with a default generated __init__() method.")

        param_names = tuple(dataclass_info.init_signature.parameters)
        # Add an extra `None` to args for the `self` parameter
        arg_list = _bind_args(dataclass_info.init_signature, ty.__name__, (None, *args), kwargs)
        assert len(dataclass_info.field_names) + 1 == len(arg_list)
        items = tuple(arg_list[param_names.index(name)] for name in dataclass_info.field_names)
        return build_dataclass_instance(items, dataclass_info)
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
                                       f' is not supported in device code')
        return await _call_builtin(ty, impl, args, kwargs, builder)


def _get_closure_parent_scopes(ty: ClosureTy, val: ClosureValue,
                               ir_ctx: IRContext) -> tuple[LocalScope, ...]:
    ret: list[LocalScope | None] = [None for _ in ty.frozen_capture_types_by_depth]
    for live_scope in ty.captured_scopes:
        ret[live_scope.depth] = live_scope.local_scope

    for depth, (func, frozen_local_indices, frozen_vars) in enumerate(
                zip(ty.func_hir.enclosing_funcs,
                    ty.func_hir.captures_by_depth,
                    val.frozen_captures_by_depth,
                    strict=True)):
        # Scope at this depth is either live or frozen (mutually exclusive)
        assert (frozen_vars is None) != (ret[depth] is None)
        if frozen_vars is not None:
            ret[depth] = LocalScope.create_frozen(func.local_names, frozen_local_indices,
                                                  frozen_vars, ir_ctx)
    return tuple(ret)


def _process_return_value(retval: Var, callee_scope: LocalScope, builder: ir.Builder) -> Var:
    ty = retval.get_type_allow_invalid()
    if not ty.is_aggregate():
        return retval

    if isinstance(ty, ClosureTy):
        retval = _freeze_returned_closure(retval, callee_scope, builder)
        ty = retval.get_type()

    old_items = retval.get_aggregate().as_tuple()
    new_items = tuple(_process_return_value(x, callee_scope, builder) for x in old_items)
    if any(old is not new for old, new in zip(old_items, new_items, strict=True)):
        new_agg_val = ty.make_aggregate_value(new_items)
        retval = builder.make_aggregate(new_agg_val, ty)

    return retval


def _freeze_returned_closure(retval: Var, callee_scope: LocalScope, builder: ir.Builder) -> Var:
    ty = retval.get_type_allow_invalid()
    assert isinstance(ty, ClosureTy)

    if len(ty.captured_scopes) == 0 or ty.captured_scopes[-1].local_scope is not callee_scope:
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
    frozen_captures = tuple(callee_scope.get(idx, builder.loc) for idx in frozen_locals)
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


_ResolvedOperand = (Var | hir.Block | hir.Function
                    | hir.StaticEvalExpression | StringFormat | hir.ResolvedName)


def _resolve_operand(x: hir.Operand, scope: Scope) -> _ResolvedOperand:
    if isinstance(x, hir.Value):
        return scope.hir2ir_varmap[x.id]
    elif isinstance(x, hir.Block | hir.Function | hir.StaticEvalExpression
                    | StringFormat | hir.ResolvedName):
        return x
    else:
        return sym2var(x, constant_only=True)


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
