# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from cuda.tile._execution import stub


if TYPE_CHECKING:
    from cuda.tile._ir.hir import StaticEvalExpression, Function


def hir_stub(func):
    """Declare a compiler-internal helper shared by all compiled execution spaces."""
    return stub(func, host=False, tile=True, simt=True, compiled_host=True)


@hir_stub
def if_else(cond, then_block, else_block, /): ...


@hir_stub
def tuple_comp_if(cond, then_block, /): ...  # Static-only if: rejects dynamic conditions


@hir_stub
def loop(body, iterable, /): ...  # infinite if `iterable` is None


@hir_stub
def static_foreach(body, items, /): ...


@hir_stub
def build_tuple(*items): ...  # Makes a tuple (i.e. returns `items`)


@hir_stub
def build_formatted_string(format, *values): ...  # Creates a FormattedStringTy value


@hir_stub
def unpack(iterable, expected_len, /): ...


@hir_stub
def identity(x): ...   # Identity function (i.e. returns `x`)


@hir_stub
def store_var(rn, value, /): ...  # Store value into the local slot given by ResolvedName


@hir_stub
def load_var(rn, name, /): ...  # Load from the slot/global given by ResolvedName


@hir_stub
def make_closure(func_hir: "Function", /, *default_values): ...


@hir_stub
def do_static_eval(expr: "StaticEvalExpression", *local_var_values): ...


@hir_stub
def do_static_assert(condition, message_block, /): ...


@hir_stub
def enter_context(manager, /): ...


@hir_stub
def pop_context(): ...


@hir_stub
def is_contained_in(x, y, /): ...  # "return x in y"


@hir_stub
def is_not_contained_in(x, y, /): ...  # return "x not in y"


# Marker consumed by context-manager lowering; never dispatched as a call.
@hir_stub
def generator_yield(value, /): ...
