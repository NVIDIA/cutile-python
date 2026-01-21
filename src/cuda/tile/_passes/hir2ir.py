# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import sys
from contextlib import contextmanager
from typing import Any, Sequence

from .ast2hir import get_function_hir
from .. import TileTypeError
from .._coroutine_util import resume_after, run_coroutine
from .._exception import Loc, TileSyntaxError, TileInternalError, TileError, TileRecursionError
from .._ir import hir, ir
from .._ir.ir import Var, IRContext, BoundMethodValue, ClosureValue, KernelArgument
from .._ir.op_impl import op_implementations
from .._ir.ops import loosely_typed_const, end_branch, return_, continue_, \
    break_, flatten_block_parameters, store_var
from .._ir.scope import Scope, LocalScope, IntMap
from .._ir.type import FunctionTy, BoundMethodTy, DTypeConstructor, ClosureTy, \
    ClosureDefaultPlaceholder
from .._ir.typing_support import get_signature, Closure

MAX_RECURSION_DEPTH = 1000


def hir2ir(func_hir: hir.Function, args: Sequence[KernelArgument], ir_ctx: IRContext) -> ir.Block:
    # Run as a coroutine using a software stack, so that we don't exceed Python's recursion limit.
    return run_coroutine(_hir2ir_coroutine(func_hir, args, ir_ctx))


async def _hir2ir_coroutine(func_hir: hir.Function, args: Sequence[KernelArgument],
                            ir_ctx: IRContext):
    scope = _create_scope(func_hir, ir_ctx, call_site=None, parent_scopes=())
    aggregate_params = [
        scope.local.redefine(local_idx, param_loc)
        for local_idx, param_loc
        in zip(func_hir.param_local_indices, func_hir.param_locs, strict=True)
    ]

    with ir.Builder(ir_ctx, func_hir.body.loc) as ir_builder, scope.make_current():
        try:
            for local_idx, var, arg in zip(func_hir.param_local_indices, aggregate_params, args,
                                           strict=True):
                var.set_type(arg.type)
                if arg.is_const:
                    var = loosely_typed_const(arg.const_value)
                    store_var(local_idx, var, var.loc)
            flat_params = flatten_block_parameters(aggregate_params)

            await _dispatch_hir_block_inner(func_hir.body, ir_builder)
        except Exception as e:
            if 'CUTILEIR' in ir_ctx.config.log_keys:
                highlight_loc = e.loc if hasattr(e, 'loc') else None
                ir_str = "\n".join(op.to_string(highlight_loc=highlight_loc)
                                   for op in ir_builder.ops)
                print(f"==== Partial cuTile IR ====\n\n{ir_str}\n\n", file=sys.stderr)
            raise

    ret = ir.Block(ir_ctx, func_hir.body.loc)
    ret.params = sum(flat_params, ())
    ret.extend(ir_builder.ops)
    return ret


def _create_scope(func_hir: hir.Function, ir_ctx: IRContext, call_site: Loc | None,
                  parent_scopes: tuple[LocalScope, ...]) -> Scope:
    local_scope = LocalScope(func_hir.local_names, ir_ctx)
    return Scope(parent_scopes + (local_scope,), None, None, call_site, IntMap(), func_hir)


async def dispatch_hir_block(block: hir.Block, cur_builder: ir.Builder | None = None):
    if cur_builder is None:
        cur_builder = ir.Builder.get_current()
    await _dispatch_hir_block_inner(block, cur_builder)


async def _dispatch_hir_block_inner(block: hir.Block, builder: ir.Builder):
    cursor = 0  # Pre-initialize to guarantee it's defined in the `except` block
    try:
        scope = Scope.get_current()
        for cursor, call in enumerate(block.calls):
            loc = call.loc.with_call_site(scope.call_site)
            with _wrap_exceptions(loc), builder.change_loc(loc):
                await _dispatch_call(call, scope)
            if builder.is_terminated:
                # The current block has been terminated, e.g. by flattening an if-else
                # with a constant condition (`if True: break`).
                return
        cursor = len(block.calls)

        loc = block.jump_loc.with_call_site(scope.call_site)
        with _wrap_exceptions(loc), builder.change_loc(loc):
            _dispatch_hir_jump(block, scope)
    except Exception:
        if 'CUTILEIR' in builder.ir_ctx.config.log_keys:
            hir_params = ", ".join(p.name for p in block.params)
            hir_lines = [str(c) for c in block.calls]
            hir_lines.append(block.jump_str())
            hir_str = "\n".join("{}{}".format("--> " if i == cursor else "    ", c)
                                for i, c in enumerate(hir_lines))
            print(f"==== HIR for ^{block.name}({hir_params}) ====\n{hir_str}\n", file=sys.stderr)
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
    args = tuple(_resolve_operand(x, scope) for x in hir_call.args)
    kwargs = {k: _resolve_operand(v, scope) for k, v in hir_call.kwargs}
    retval = await call(callee_var, args, kwargs)
    if hir_call.result is not None and retval is not None:
        scope.hir2ir_varmap[hir_call.result.id] = retval


async def call(callee_var: Var, args, kwargs) -> Var | None:
    builder = ir.Builder.get_current()
    callee, self_arg = _get_callee_and_self(callee_var)
    args = self_arg + args
    arg_list = _bind_args(callee, args, kwargs)
    if callee in op_implementations:
        impl = op_implementations[callee]
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
    else:
        # Callee is a user-defined function.
        _check_recursive_call(builder.loc)
        if isinstance(callee, Closure):
            callee_hir = callee.ty.func_hir
            parent_scopes = _get_closure_parent_scopes(callee, builder.ir_ctx)
        else:
            callee_hir = get_function_hir(callee, entry_point=False)
            parent_scopes = ()

        for param_name, param in callee_hir.signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                raise TileSyntaxError("Variadic parameters in user-defined"
                                      " functions are not supported")

        # Activate a fresh Scope.
        new_scope = _create_scope(callee_hir, builder.ir_ctx, call_site=builder.loc,
                                  parent_scopes=parent_scopes)
        with new_scope.make_current():
            # Call store_var() to bind arguments to parameters.
            for arg, local_idx, param_loc in zip(arg_list, callee_hir.param_local_indices,
                                                 callee_hir.param_locs, strict=True):
                store_var(local_idx, arg, param_loc)

            # Dispatch the function body. Use resume_after() to break the call stack
            # and make sure we stay within the Python's recursion limit.
            await resume_after(dispatch_hir_block(callee_hir.body, builder))

        assert callee_hir.body.have_result
        ret = _process_return_value(
                new_scope.hir2ir_varmap[callee_hir.body.result.id], new_scope.local, builder)
        new_scope.local.mark_dead()
        return ret


def _get_closure_parent_scopes(closure: Closure, ir_ctx: IRContext) -> tuple[LocalScope, ...]:
    ret: list[LocalScope | None] = [None for _ in closure.ty.frozen_capture_types_by_depth]
    for live_scope in closure.ty.captured_scopes:
        ret[live_scope.depth] = live_scope.local_scope

    for depth, (func, frozen_local_indices, frozen_vars) in enumerate(
                zip(closure.ty.func_hir.enclosing_funcs,
                    closure.ty.func_hir.captures_by_depth,
                    closure.val.frozen_captures_by_depth,
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
            ty.frozen_capture_types_by_depth, depth, frozen_capture_types)
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


def _get_callee_and_self(callee_var: Var) -> tuple[Any, tuple[()] | tuple[Var]]:
    callee_ty = callee_var.get_type()
    if isinstance(callee_ty, FunctionTy):
        return callee_ty.func, ()
    elif isinstance(callee_ty, BoundMethodTy):
        bound_method = callee_var.get_aggregate()
        assert isinstance(bound_method, BoundMethodValue)
        return callee_ty.func, (bound_method.bound_self,)
    elif isinstance(callee_ty, DTypeConstructor):
        return callee_ty.dtype, ()
    elif isinstance(callee_ty, ClosureTy):
        return Closure(callee_ty, callee_var.get_aggregate()), ()
    else:
        raise TileTypeError(f"Cannot call an object of type {callee_ty}")


def _resolve_operand(x: hir.Operand, scope: Scope) -> Var | hir.Block | hir.Function:
    if isinstance(x, hir.Value):
        return scope.hir2ir_varmap[x.id]
    elif isinstance(x, hir.Block | hir.Function):
        return x
    else:
        return loosely_typed_const(x)


def _bind_args(sig_func, args, kwargs) -> list[Var]:
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
            if isinstance(param.default, ClosureDefaultPlaceholder):
                assert isinstance(sig_func, Closure)
                default = sig_func.val.default_values[param.default.default_value_index]
            else:
                default = loosely_typed_const(param.default)
            ret.append(default)
    return ret
