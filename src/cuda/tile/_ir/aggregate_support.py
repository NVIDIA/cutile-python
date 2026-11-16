# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Iterator

from cuda.tile._ir.ir import Var, Builder
from cuda.tile._ir.op_impl import ImplRegistry
from cuda.tile._ir.type import Type, InvalidType


def flatten_aggregates(vars: Sequence[Var], types: Sequence[Type]) -> tuple[Var, ...]:
    ret = []
    for x, ty in zip(vars, types, strict=True):
        item_types = tuple(ty.flatten_aggregate())
        x_ty = x.get_type_allow_invalid()
        if isinstance(x_ty, InvalidType):
            for _ in item_types:
                t = x.ctx.make_temp()
                t.set_type(x_ty)
                ret.append(t)
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


def unflatten_aggregates(flattened: tuple[Var, ...],
                         nominal: Sequence[Type], actual: Sequence[Type]) -> tuple[Var, ...]:
    it = iter(flattened)
    ret = tuple(_maybe_unflatten_aggregate(it, n, a) for n, a in zip(nominal, actual, strict=True))
    assert next(it, None) is None
    return ret


def _maybe_unflatten_aggregate(flattened_iter: Iterator[Var], nominal: Type, actual: Type) -> Var:
    if not nominal.is_aggregate():
        return next(flattened_iter)
    return _unflatten_proper_aggregate(flattened_iter, nominal, actual, result_var=None)


def expand_aggregate_var(var: Var) -> tuple[Var, ...]:
    item_types = tuple(var.get_type().flatten_aggregate())
    ret = tuple(var.ctx.make_var(f"{var.name}_{i}")
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
        builder = Builder.get_current()
        t = builder.ir_ctx.make_temp()
        t.set_type(actual)
        return t

    items = tuple(_maybe_unflatten_aggregate(flattened_iter, item_nominal, item_actual)
                  for item_nominal, item_actual
                  in zip(nominal_item_types, actual.aggregate_item_types(), strict=True))
    val = nominal.make_aggregate_value(items)

    impl = ImplRegistry.get_current().unflatten_aggregate_implementations.get(type(nominal))
    if impl is None:
        return Builder.get_current().make_aggregate(val, nominal, result_var=result_var)
    else:
        return impl(val, nominal, result_var)
