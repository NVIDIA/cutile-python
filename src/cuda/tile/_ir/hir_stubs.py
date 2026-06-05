# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from cuda.tile._execution import stub


if TYPE_CHECKING:
    from cuda.tile._ir.hir import StaticEvalExpression, Function


@stub
def if_else(cond, then_block, else_block, /): ...


@stub
def loop(body, iterable, /): ...  # infinite if `iterable` is None


@stub
def static_foreach(body, items, /): ...


@stub
def build_tuple(*items): ...  # Makes a tuple (i.e. returns `items`)


@stub
def build_formatted_string(format, *values): ...  # Creates a FormattedStringTy value


@stub
def unpack(iterable, expected_len, /): ...


@stub
def identity(x): ...   # Identity function (i.e. returns `x`)


@stub
def store_var(name, value, /): ...  # Store into a named variable


@stub
def load_var(name, /): ...  # Load from a named variable


@stub
def make_closure(func_hir: "Function", /, *default_values): ...


@stub
def do_static_eval(expr: "StaticEvalExpression", *local_var_values): ...


@stub
def do_static_assert(condition, message_block, /): ...


@stub
def enter_context(manager, /): ...


@stub
def pop_context(): ...


@stub
def is_contained_in(x, y, /): ...  # "return x in y"


@stub
def is_not_contained_in(x, y, /): ...  # return "x not in y"
