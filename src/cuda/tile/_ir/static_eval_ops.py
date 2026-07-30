# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from cuda.tile import TileSyntaxError, TileError, TileStaticEvalError, \
    TileTypeError
from cuda.tile._datatype import is_boolean
from cuda.tile._dispatch_mode import StaticEvalMode
from cuda.tile._exception import StaticAssertionError
from cuda.tile._ir import hir_stubs, hir
from cuda.tile._ir.core_ops import loosely_typed_const, build_tuple, sym2var
from cuda.tile._ir.ir import Var
from cuda.tile._ir.op_impl import ImplRegistry

import cuda.tile._stub as ct
from cuda.tile._ir.scope import Scope, ControlFlowInfo
from cuda.tile._ir.type import var2sym, TupleValue, TensorLikeTy


_registry = ImplRegistry()
impl = _registry.impl


def static_eval_impl_registry() -> ImplRegistry:
    return _registry


@impl(ct.static_eval)
def static_eval_impl(expr: Var):
    raise TileSyntaxError("static_eval() must be used directly by name,"
                          " e.g. cuda.tile.static_eval() or ct.static_eval().")


@impl(ct.static_assert)
def static_assert_impl(condition: Var, message: Var):
    raise TileSyntaxError("static_assert() must be used directly by name,"
                          " e.g. cuda.tile.static_assert() or ct.static_assert().")


@impl(ct.static_iter)
def static_iter_impl(iterable: Var):
    raise TileSyntaxError("static_iter() must be used directly by name,"
                          " e.g. cuda.tile.static_iter() or ct.static_iter().")


@impl(hir_stubs.do_static_eval)
def do_static_eval_impl(expr: hir.StaticEvalExpression,
                        local_var_values: tuple[Var, ...]) -> Var:
    local_proxies = tuple(var2sym(x) for x in local_var_values)
    with StaticEvalMode(expr.kind).as_current():
        try:
            result = expr.compiled_expr(*local_proxies)
        except TileError:
            raise
        except Exception as e:
            where = expr.kind._value_
            msg = f"Exception was raised inside {where} ({type(e).__name__}"
            e_str = str(e)
            if len(e_str) > 0:
                msg += ": " + e_str
            msg += ")"
            raise TileStaticEvalError(msg) from e

    if expr.kind == hir.StaticEvalKind.STATIC_ASSERT_MESSAGE:
        if result is None:
            result = ""
        return loosely_typed_const(str(result))
    elif expr.kind == hir.StaticEvalKind.STATIC_ITER_ITERABLE:
        items = _drain_static_iter_iterable(result)
        return build_tuple(tuple(items))
    else:
        return sym2var(result)


_STATIC_ITER_MAX_ITERATIONS = 1000


def _drain_static_iter_iterable(iterable) -> list[Var]:
    try:
        it = iter(iterable)
    except Exception as e:
        msg = str(e)
        if len(msg) > 0:
            msg = ": " + msg
        raise TileTypeError(f"Invalid static_iter() iterable{msg}")

    items = []
    for i in range(_STATIC_ITER_MAX_ITERATIONS + 1):
        try:
            x = next(it)
        except StopIteration:
            break
        except Exception as e:
            msg = str(e)
            if len(msg) > 0:
                msg = ": " + msg
            raise TileTypeError(f"Error was raised while obtaining item #{i}"
                                f" from the static_iter() iterable{msg}")

        try:
            var = sym2var(x)
        except TileTypeError as e:
            raise TileStaticEvalError(
                f"Invalid item #{i} of static_iter() iterable: {str(e)}")

        items.append(var)
    else:
        raise TileStaticEvalError(f"Maximum number of iterations"
                                  f" ({_STATIC_ITER_MAX_ITERATIONS}) has been reached"
                                  f" while unpacking the static_iter() iterable")
    return items


@impl(hir_stubs.do_static_assert)
async def do_static_assert_impl(condition: Var, message_block: hir.Block) -> None:
    if not condition.is_constant():
        raise TileTypeError("static_assert() condition must be a compile-time constant")

    ty = condition.get_type()
    if not (isinstance(ty, TensorLikeTy) and is_boolean(ty.tensor_dtype())):
        raise TileTypeError(f"static_assert() condition must be a boolean, not {ty}")

    if condition.get_constant():
        return None

    from .._passes.hir2ir import dispatch_hir_block
    info = ControlFlowInfo((), flatten=True)
    with Scope.get_current().change_if_else_info(info):
        await dispatch_hir_block(message_block)
    [jump] = info.jumps
    assert jump.jump_op is None
    [message] = jump.outputs
    message = message.get_constant()
    assert isinstance(message, str)
    raise StaticAssertionError(message)


@impl(hir_stubs.static_foreach)
async def static_foreach_impl(body: hir.Block, items: Var):
    scope = Scope.get_current()

    tuple_val = items.get_aggregate()
    assert isinstance(tuple_val, TupleValue)

    for item in tuple_val.items:
        scope.hir2ir_varmap[body.params[0].id] = item
        from .._passes.hir2ir import dispatch_hir_block
        await dispatch_hir_block(body)


@impl(ct.ensure_constant)
def ensure_constant_impl(value: Var):
    if not value.is_constant():
        raise StaticAssertionError("Argument of ensure_constant() is not a compile-time constant")
    return value
