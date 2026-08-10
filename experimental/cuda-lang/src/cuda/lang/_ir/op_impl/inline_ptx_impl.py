# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from cuda.tile._datatype import is_pointer_dtype
from cuda.tile._ir.op_impl import (
    require_constant_str,
    require_dtype_spec,
    ImplRegistry,
)
from cuda.tile._ir.core_ops import (
    build_tuple,
)
from cuda.tile._ir.ir import add_operation_variadic
from cuda.lang._exception import TypeCheckingError
import cuda.lang._datatype as datatype
from ..op_defs import InlinePTX
from ..type_checking_helpers import (
    require_scalar_type,
    require_pointer_type,
)
from ..type import (
    ScalarTy,
    PointerTy,
    TupleTy,
    TupleValue,
)
from ..ir import Var
from ..._stub import core_api

_registry = ImplRegistry()
impl = _registry.impl


def inline_ptx_impl_registry() -> ImplRegistry:
    return _registry


@dataclass(eq=False, frozen=True)
class InlinePTXOperand:
    mode: InlinePTX.RMWMode
    type_code: str
    value: Var | datatype.DType


def require_inline_ptx_pair(var: Var) -> tuple[Var, Var]:
    pair_ty = var.get_type()
    if not isinstance(pair_ty, TupleTy) or len(pair_ty.value_types) != 2:
        raise TypeCheckingError(
            "Expected constraint arguments to be pairs of constraint strings and values"
        )
    pair_val = var.get_aggregate()
    assert isinstance(pair_val, TupleValue)
    return pair_val.as_tuple()


_INLINE_PTX_MODE_FROM_PREFIX = {
    "": InlinePTX.RMWMode.READ_ONLY,
    "=": InlinePTX.RMWMode.WRITE_ONLY,
    "+": InlinePTX.RMWMode.READ_WRITE,
}

_INLINE_PTX_TYPECODES = (
    "h",
    "r",
    "l",
    "f",
    "d",
    "p",
)

_INLINE_PTX_SCALAR_DTYPE_FROM_TYPECODE = {
    "h": datatype.int16,
    "r": datatype.int32,
    "l": datatype.int64,
    "f": datatype.float32,
    "d": datatype.float64,
}


def parse_inline_ptx_constraint(var: Var) -> tuple[str, InlinePTX.RMWMode, str]:
    constraint_str = require_constant_str(var)

    if len(constraint_str) not in (1, 2):
        raise TypeCheckingError(
            f"Invalid inline_ptx constraint {constraint_str}, expected length 1 or 2"
        )

    prefix = constraint_str[0:-1]
    type_char = constraint_str[-1]

    mode = _INLINE_PTX_MODE_FROM_PREFIX.get(prefix)
    if mode is None:
        raise TypeCheckingError(
            f"Unknown constraint rmw modifier {prefix!r}, expected "
            "'' (meaning readonly), '+' (meaning readwrite), or '=' (meaning writeonly)"
        )
    if type_char not in _INLINE_PTX_TYPECODES:
        expected = ", ".join(_INLINE_PTX_TYPECODES)
        raise TypeCheckingError(
            f"Unknown constraint dtype {type_char!r}, expected one of {expected}"
        )
    if mode is InlinePTX.RMWMode.READ_WRITE:
        raise TypeCheckingError("Read-write inline_ptx constraints are not supported")

    return constraint_str, mode, type_char


def validate_inline_ptx_operand(
    constraint_str: str, mode: InlinePTX.RMWMode, type_char: str, value: Var
) -> InlinePTXOperand:
    if mode is InlinePTX.RMWMode.WRITE_ONLY:
        actual_dtype = require_dtype_spec(value)
        if type_char == "p":
            if not is_pointer_dtype(actual_dtype):
                raise TypeCheckingError(
                    f"Expected a pointer dtype for constraint {constraint_str}, "
                    f"got {actual_dtype}"
                )
            return InlinePTXOperand(
                mode=mode, type_code=type_char, value=actual_dtype
            )

        expected_dtype = _INLINE_PTX_SCALAR_DTYPE_FROM_TYPECODE[type_char]
        if actual_dtype != expected_dtype:
            raise TypeCheckingError(
                f"Expected dtype {expected_dtype} for constraint "
                f"{constraint_str}, got {actual_dtype}"
            )
        return InlinePTXOperand(mode=mode, type_code=type_char, value=actual_dtype)

    if type_char == "p":
        require_pointer_type(value)
        return InlinePTXOperand(mode=mode, type_code=type_char, value=value)

    actual_dtype = require_scalar_type(value).dtype
    expected_dtype = _INLINE_PTX_SCALAR_DTYPE_FROM_TYPECODE[type_char]
    if actual_dtype != expected_dtype:
        raise TypeCheckingError(
            f"Expected value of type {expected_dtype} for "
            f"constraint {constraint_str}, got {actual_dtype}"
        )

    return InlinePTXOperand(mode=mode, type_code=type_char, value=value)


def require_constant_constraint_tuple(
    constraint_tuple: Var,
) -> InlinePTXOperand:
    constraint_var, value_var = require_inline_ptx_pair(constraint_tuple)
    constraint_str, mode, type_char = parse_inline_ptx_constraint(constraint_var)
    return validate_inline_ptx_operand(constraint_str, mode, type_char, value_var)


_INLINE_PTX_PLACEHOLDER_RE = re.compile(r"%(?P<index>[0-9]+)")


def require_inline_ptx_constraint_pairs(
    ptx_code: str, constraint_pairs: tuple
) -> tuple:
    if not isinstance(constraint_pairs, tuple):
        raise TypeCheckingError(
            f"Expected a tuple of constraint pairs, but got {type(constraint_pairs)}"
        )

    ro_args, rw_args, wo_args = [], [], []
    # need to replace e.g. %0 with {$r0}, {$rw0}, or {$w0} for all ptx
    # interpolation directives.
    ptx_interpolation_replacements = []
    arg_specs = [require_constant_constraint_tuple(pair) for pair in constraint_pairs]

    for arg_spec in arg_specs:
        match arg_spec.mode:
            case InlinePTX.RMWMode.READ_ONLY:
                ptx_interpolation_replacements.append("{$r" + str(len(ro_args)) + "}")
                assert isinstance(arg_spec.value, Var)
                ro_args.append(arg_spec.value)
            case InlinePTX.RMWMode.READ_WRITE:
                ptx_interpolation_replacements.append("{$rw" + str(len(rw_args)) + "}")
                assert isinstance(arg_spec.value, Var)
                rw_args.append(arg_spec.value)
            case InlinePTX.RMWMode.WRITE_ONLY:
                ptx_interpolation_replacements.append("{$w" + str(len(wo_args)) + "}")
                assert isinstance(arg_spec.value, datatype.DType)
                wo_args.append(arg_spec.value)

    def rewrite(match: re.Match[str]) -> str:
        index = int(match.group("index"))
        if index >= len(ptx_interpolation_replacements):
            raise TypeCheckingError(
                f"inline_ptx placeholder %{index} is out of range "
                f"for {len(ptx_interpolation_replacements)} operands"
            )

        return ptx_interpolation_replacements[index]

    ptx_fragments = []
    for fragment in ptx_code.split("%%"):
        fragment = _INLINE_PTX_PLACEHOLDER_RE.sub(rewrite, fragment)
        if "%" in fragment:
            raise TypeCheckingError(
                "Literal percent signs in inline_ptx must be escaped as '%%'"
            )
        ptx_fragments.append(fragment)
    mlir_ptx_code = "%".join(ptx_fragments)
    return (
        mlir_ptx_code,
        tuple(ro_args),
        tuple(rw_args),
        tuple(wo_args),
    )


@impl(core_api._inline_ptx)
def inline_ptx_impl(ptx_code: Var, constraint_pairs: tuple) -> Var[TupleTy]:
    ptx_code = require_constant_str(ptx_code)
    mlir_ptx_code, ro_args, rw_args, wo_args = require_inline_ptx_constraint_pairs(
        ptx_code, constraint_pairs
    )
    result_types = tuple(
        PointerTy(dtype) if is_pointer_dtype(dtype) else ScalarTy(dtype)
        for dtype in wo_args
    )
    results = add_operation_variadic(
        InlinePTX,
        result_types,
        ptx_code=mlir_ptx_code,
        read_only_operands=ro_args,
        write_only_operands=wo_args,
        read_write_operands=rw_args,
    )
    return build_tuple(results)
