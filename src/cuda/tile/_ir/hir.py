# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# HIR stands for "High-level Intermediate Representation".
# The HIR is the initial representation that we build from the Python AST (see ast2hir.py).


import enum
import inspect
from dataclasses import dataclass
from typing import Callable, Mapping

from cuda.tile._exception import Loc, FunctionDesc
from cuda.tile._cext import HirLabel, HirProgram


@dataclass(eq=False, repr=False)
class Function:
    desc: FunctionDesc
    program: HirProgram

    # For nested functions, this signature is synthesized from AST. In this case,
    # default values of parameters are represented with ClosureDefaultPlaceholder objects.
    signature: inspect.Signature

    # Names of all local variables defined in this function.
    local_names: tuple[str, ...]

    # For each function parameter, an index into `local_names`
    param_local_indices: tuple[int, ...]

    # For each function parameter, its location (same length as `param_local_indices`)
    param_locs: tuple[Loc, ...]

    # List of all function definitions nested directly inside this function.
    nested_functions: "tuple[Function, ...]"

    # Pre-computed view of captured locals: for each non-negative `depth` strictly less than this
    # function's depth, the list of all local variable indices captured by this function.
    # Empty when this is a top-level function.
    captures_by_depth: tuple[tuple[int, ...], ...]

    # For each enclosing function definition (outermost first), its `local_names`.
    # Empty when this is a top-level function.
    enclosing_local_names: tuple[tuple[str, ...], ...]

    stored_indices_by_block: Mapping[HirLabel, set[int]]

    def __repr__(self):
        return f"<HIR for function {self.desc}>"


class StaticEvalKind(enum.Enum):
    STATIC_EVAL = "static_eval()"
    STATIC_DEF = "@static_def function"
    STATIC_ASSERT_CONDITION = "static_assert() condition"
    STATIC_ASSERT_MESSAGE = "static_assert() message"
    STATIC_ITER_ITERABLE = "static_iter() iterable"
    STATIC_EXCEPTION = "static_exception() expression"


@dataclass
class StaticEvalExpression:
    compiled_expr: Callable
    kind: StaticEvalKind
