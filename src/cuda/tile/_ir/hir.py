# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# HIR stands for "High-level Intermediate Representation".
# The HIR is the initial representation that we build from the Python AST (see ast2hir.py).
#
# Unlike the IR, it doesn't have specific Operation definitions. Instead, it uses the concept
# of a "function call" to model all operations, including structured control flow.
# For example, addition like `a + b` is represented as calling `operator.add(a, b)`.
#
# It also has a simpler representation of constants: they can be used directly as arguments
# or as functions to be called (see the `Operand` type alias below).


import enum
import inspect
import threading
from dataclasses import dataclass
from textwrap import indent
from types import CodeType
from typing import Any, Set, OrderedDict, Callable

from cuda.tile._exception import Loc, FunctionDesc


@dataclass(frozen=True)
class Value:
    id: int

    def __str__(self):
        return f"%{self.id}"


# Cache Value objects for reuse
_value_cache = []
_value_cache_lock = threading.Lock()


def make_value(id: int) -> Value:
    try:
        # Fast path
        return _value_cache[id]
    except IndexError:
        pass

    if id >= 2000:
        # Don't cache too many objects
        return Value(id)

    # Add 100 objects at a time to avoid triggering the slow path every time
    with _value_cache_lock:
        _value_cache.extend([Value(j) for j in range(len(_value_cache), id + 100)])
    return _value_cache[id]


# An "Operand" is a value that can be used as a function's argument, or as the function itself.
# There are two kinds of Operands:
#    - Using a `Value` instance as an Operand signals that this Operand is a result
#      of a previous call, or a kernel parameter.
#    - An object of any other type means that it is an immediate constant.
Operand = Value | Any


ModuleType = type(enum)


@dataclass
class Call:
    result: Value | None
    callee: Operand
    args: tuple[Operand, ...]
    kwargs: tuple[tuple[str, Operand], ...]
    loc: Loc

    def __str__(self):
        opfmt = _OperandFormatter([])
        loc_str = f"  # Line {self.loc.line}"
        if self.callee is identity:
            return f"{self.result} = {opfmt(self.args[0])}{loc_str}"
        lhs_str = "" if self.result is None else f"{self.result} = "
        callee_str = opfmt(self.callee)
        args_and_kwargs = (*(opfmt(a) for a in self.args),
                           *(f"{k}={opfmt(v)}" for k, v in self.kwargs))
        args_str = ", ".join(args_and_kwargs)
        blocks_str = "".join(indent(f"\n{b}", "    ") for b in opfmt.blocks)
        return f"{lhs_str}{callee_str}({args_str}){loc_str}{blocks_str}"


class Jump(enum.Enum):
    END_BRANCH = "end_branch"
    CONTINUE = "continue"
    BREAK = "break"
    RETURN = "return"


@dataclass
class Block:
    block_id: int
    params: tuple[Value, ...]
    calls: list[Call]
    have_result: bool
    result: Operand
    jump: Jump | None
    jump_loc: Loc
    stored_names: Set[str]
    loc: Loc

    def __str__(self):
        params_str = ", ".join(str(p) for p in self.params)
        calls_str = "".join(f"\n{c}" for c in self.calls)
        if self.jump is not None:
            calls_str += "\n" + self.jump_str()
        calls_str = indent(calls_str, "    ")
        return f"^{self.block_id}({params_str}):{calls_str}"

    def jump_str(self):
        opfmt = _OperandFormatter([])
        results_str = "" if self.result is None else opfmt(self.result)
        return f"{self.jump._value_}{results_str}  # Line {self.jump_loc.line}"


# Resolved position of a named variable.
#
# If `depth` is -1, then this is a global variable, and the `index`
# points into `hir.Function.frozen_global_names` / `hir.Function.frozen_global_values`.
# If both `depth` and `index` are -1, then this is an indication that no variable
# with the given name has been found.
#
# If `depth` is non-negative but is less than the depth of the current function,
# then this is a captured local variable of an enclosing function.
# In this case, the local variable name can be found in
# `hir.Function.enclosing_funcs[depth].local_names[index].
#
# If `depth` is equal to the depth of the current function, then this is a local variable.
# In this case, the local variable name can be found in `hir.Function.local_names[index]`.
@dataclass(frozen=True)
class ResolvedName:
    depth: int
    index: int


UNKNOWN_NAME = ResolvedName(-1, -1)


@dataclass(eq=False, repr=False)
class Function:
    desc: FunctionDesc
    body: Block

    # For nested functions, this signature is synthesized from AST. In this case,
    # default values of parameters are represented with ClosureDefaultPlaceholder objects.
    signature: inspect.Signature

    # Names of all local variables defined in this function.
    local_names: tuple[str, ...]

    # For each function parameter, an index into `local_names`
    param_local_indices: tuple[int, ...]

    # For each function parameter, its location (same length as `param_local_indices`)
    param_locs: tuple[Loc, ...]

    # Names of all global-like variables accessible by this function.
    frozen_global_names: tuple[str, ...]

    # Values of all global-like variables access by this function.
    # We assume that they don't change after compilation, hence the name "frozen".
    # Same length as `frozen_global_names`.
    frozen_global_values: tuple[Any, ...]

    # Strict upper bound on hir.Value.id inside this function definition,
    # excluding any nested functions. Can be used to pre-allocate lists etc.
    value_id_upper_bound: int

    # List of all function definitions nested directly inside this function.
    nested_functions: "tuple[Function, ...]"

    # Names of all variables that are ever loaded by this function
    loaded_names: tuple[str, ...]

    # For each variable name used in this function (i.e. loaded from or stored to),
    # its resolved position in the scope.
    used_names: OrderedDict[str, ResolvedName]

    # Pre-computed view of `used_names`: for each non-negative `depth` strictly less than this
    # function's depth, the list of all local variable indices captured by this function.
    # Empty when this is a top-level function.
    captures_by_depth: tuple[tuple[int, ...], ...]

    # Sequence of enclosing function definitions, outermost first.
    # Empty when this is a top-level function.
    enclosing_funcs: "tuple[Function, ...]"

    # Code object compiled for evaluation inside static_eval context.
    # Only set for nested functions & lambdas.
    code_object: CodeType | None

    def __repr__(self):
        return f"<HIR for function {self.desc}>"


@dataclass
class _OperandFormatter:
    blocks: list["Block"]

    def __call__(self, x: Operand) -> str:
        if isinstance(x, Value):
            return str(x)
        elif isinstance(x, ModuleType):
            return str(f"<mod:{x.__name__}>")
        elif isinstance(x, Block):
            self.blocks.append(x)
            return f"^{x.block_id}"
        elif callable(x):
            return f"<fn:{x.__name__}>"
        else:
            return f"<{repr(x)}>"


# ==================================
# Special function stubs used in HIR
# ==================================


class StaticEvalKind(enum.Enum):
    STATIC_EVAL = "static_eval()"
    STATIC_ASSERT_CONDITION = "static_assert() condition"
    STATIC_ASSERT_MESSAGE = "static_assert() message"
    STATIC_ITER_ITERABLE = "static_iter() iterable"


@dataclass
class StaticEvalExpression:
    compiled_expr: Callable
    kind: StaticEvalKind


def if_else(cond, then_block, else_block, /): ...
def loop(body, iterable, /): ...  # infinite if `iterable` is None
def static_foreach(body, items, /): ...
def build_tuple(*items): ...  # Makes a tuple (i.e. returns `items`)
def unpack(iterable, expected_len, /): ...
def identity(x): ...   # Identity function (i.e. returns `x`)
def store_var(name, value, /): ...  # Store into a named variable
def load_var(name, /): ...  # Load from a named variable
def make_closure(func_hir: Function, /, *default_values): ...
def do_static_eval(expr: StaticEvalExpression, *local_var_values): ...
def do_static_assert(condition, message_block, /): ...
