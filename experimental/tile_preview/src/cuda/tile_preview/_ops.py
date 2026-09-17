# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._datatype import DType
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.core_ops import build_tuple
from cuda.tile._ir.ir import (
    Operation,
    MemoryEffect,
    Var,
    add_operation_variadic,
    attribute,
    operand,
)
from cuda.tile._ir.op_impl import (
    require_constant_int_tuple,
    require_constant_str,
    require_dtype_spec,
    require_tuple_type,
)
from cuda.tile._ir.ops import tile_impl_registry
from cuda.tile._ir.type import (
    DTypeSpec,
    TileTy,
    TokenTy,
    TupleTy,
)
from cuda.tile._ir2bytecode import BytecodeContext, dtype_typeid

from . import _attribute as preview_attribute
from cuda.tile._bytecode import encodings
from ._stub import foreign_call
from ._types import Tilelib


def _constant_attributes(ctx: BytecodeContext, values: tuple[object, ...]):
    attrs = []
    for value in values:
        if isinstance(value, bool):
            raise TileTypeError("boolean foreign-call constant parameters are not supported")
        if isinstance(value, int):
            attrs.append(preview_attribute.PreviewExprInt(value))
        elif isinstance(value, str):
            attrs.append(preview_attribute.PreviewExprConstStr(value))
        elif isinstance(value, DType):
            attrs.append(preview_attribute.PreviewConcreteType(dtype_typeid(ctx.type_table, value)))
        else:
            raise TileTypeError(
                "foreign-call constant parameters must be integers, strings, or dtypes"
            )
    return tuple(attrs)


@dataclass(eq=False)
class PreviewForeignCall(
    Operation,
    opcode="preview_foreign_call",
    memory_effect=MemoryEffect.STORE,
):
    tilelibs: tuple[tuple[Path, str], ...] = attribute()
    symbol_name: str = attribute()
    constant_params: tuple[object, ...] = attribute()
    inputs: tuple[Var, ...] = operand()
    token: Optional[Var] = operand(default=None)

    def generate_bytecode(self, ctx: BytecodeContext):
        result_types = tuple(ctx.typeid_of(var) for var in self.result_vars)
        operands = tuple(ctx.get_value(var) for var in self.inputs)
        if self.token is not None:
            operands += (ctx.get_value(self.token),)
        result = encodings.encode_PreviewCallOp(
            ctx.builder,
            result_types=result_types,
            operands=operands,
            callee=self.symbol_name,
            const_args=_constant_attributes(ctx, self.constant_params),
            arg_attrs=None,
            res_attrs=None,
        )
        if result is None:
            return ()
        return result


@tile_impl_registry.impl(foreign_call, min_version=BytecodeVersion.V_13_5)
def foreign_call_impl(
    tilelibs: Var,
    symbol_name: Var,
    constant_params: Var,
    inputs: Var,
    output_types: Var,
) -> Var:
    # Validate tilelibs
    require_tuple_type(tilelibs)
    tilelib_values = []
    for i, item in enumerate(tilelibs.get_aggregate().items):
        if not item.is_constant():
            raise TileTypeError(f"tilelibs[{i}] must be a compile-time constant")
        value = item.get_constant()
        if not isinstance(value, Tilelib):
            raise TileTypeError(
                f"tilelibs[{i}] must be a tile_preview.Tilelib instance"
            )
        tilelib_values.append((Path(value.path), value.version))
    if not tilelib_values:
        raise TileTypeError("tilelibs must contain at least one Tilelib")

    # Validate symbol_name
    symbol_value = require_constant_str(symbol_name)
    if not symbol_value:
        raise TileTypeError("symbol_name must be a non-empty constant string")

    # Validate constant_params
    constant_values = []
    require_tuple_type(constant_params)
    for i, item in enumerate(constant_params.get_aggregate().items):
        if not item.is_constant():
            raise TileTypeError(
                f"constant_params[{i}] must be a compile-time constant"
            )
        value = item.get_constant()
        if isinstance(value, bool) or not isinstance(value, (int, str, DType)):
            raise TileTypeError(
                f"constant_params[{i}] must be an integer, string, or cuda.tile dtype;"
                f" got {type(value).__name__}"
            )
        constant_values.append(value)

    # Validate inputs
    call_inputs = []
    require_tuple_type(inputs)
    for i, item in enumerate(inputs.get_aggregate().items):
        item_ty = item.get_type()
        if isinstance(item_ty, TileTy):
            call_inputs.append(item)
        else:
            raise TileTypeError(
                f"inputs[{i}] must be a tile or scalar; got {item_ty}. "
                "Convert arrays with the foreign_pointer property"
            )

    # Validate output_types
    data_result_types = []
    require_tuple_type(output_types)
    for i, descriptor in enumerate(output_types.get_aggregate().items):
        descriptor_ty = descriptor.get_type()
        if isinstance(descriptor_ty, DTypeSpec):
            dtype = require_dtype_spec(descriptor)
            shape = ()
        elif isinstance(descriptor_ty, TupleTy):
            descriptor_items = descriptor.get_aggregate().items
            if len(descriptor_items) != 2:
                raise TileTypeError(
                    f"output_types[{i}] must be a dtype or (dtype, shape)"
                )
            dtype = require_dtype_spec(descriptor_items[0])
            shape = require_constant_int_tuple(descriptor_items[1])
            if any(dim < 0 for dim in shape):
                raise TileTypeError(f"output_types[{i}].shape must be non-negative")
        else:
            raise TileTypeError(
                f"output_types[{i}] must be a dtype or (dtype, shape)"
            )
        result_ty = TileTy(dtype, shape)
        data_result_types.append(result_ty)

    # Create the operation
    results = add_operation_variadic(
        PreviewForeignCall,
        result_types=(*data_result_types, TokenTy()),
        tilelibs=tuple(tilelib_values),
        symbol_name=symbol_value,
        constant_params=tuple(constant_values),
        inputs=tuple(call_inputs),
        token=None,
    )
    return build_tuple(results[:-1])
