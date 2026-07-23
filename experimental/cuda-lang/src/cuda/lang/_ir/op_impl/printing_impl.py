# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.core_ops import build_formatted_string as tile_build_formatted_string
from cuda.tile._ir.type import FormattedPiece
from cuda.tile._ir.type import StringFormat
from cuda.tile._ir import hir_stubs
import builtins

import cuda.lang._datatype as datatype
from cuda.tile._ir.core_ops import (
    FormattedStringBuilder,
    loosely_typed_const,
    repr_tensorlike_impl as tile_repr_tensorlike_impl,
)
from cuda.tile._ir.op_impl import ImplRegistry
from cuda.tile._ir.type import TensorLikeTy
from cuda.tile._passes.hir2ir import call_function

from ..ir import Var, add_operation
from ..op_defs import BitCast
from ..type import PointerTy, ScalarTy, VectorTy, type_bitwidth
from .core_api_impl import bitcast
from .vector_impl import vector_getitem


_registry = ImplRegistry()
impl = _registry.impl


def printing_impl_registry() -> ImplRegistry:
    return _registry


async def repr_vector(x: Var[VectorTy]):
    builder = FormattedStringBuilder()
    vector_type = x.get_type()
    builder.append_literal_piece("<")
    separator = ""
    for index in range(vector_type.length):
        builder.append_literal_piece(separator)
        item = vector_getitem(x, loosely_typed_const(index))
        item_repr = await call_function(builtins.repr, item)
        builder.append_string_var(item_repr)
        separator = ", "
    builder.append_literal_piece(">")
    return builder.build()


def to_uint32_words(x: Var[ScalarTy | PointerTy]) -> tuple[Var[ScalarTy], ...]:
    bitwidth = type_bitwidth(x.get_type())
    if isinstance(x.get_type(), PointerTy):
        integer_dtype = datatype.uint32 if bitwidth == 32 else datatype.uint64
        x = bitcast(x, integer_dtype)

    if bitwidth == 32:
        return (x,)

    assert bitwidth % 32 == 0
    word_count = bitwidth // 32
    words = add_operation(
        BitCast,
        VectorTy(datatype.uint32, word_count),
        x=x,
    )
    return tuple(
        vector_getitem(words, loosely_typed_const(index)) for index in range(word_count)
    )


def format_hex_le(x: Var[ScalarTy | PointerTy]):
    builder = FormattedStringBuilder()
    builder.append_literal_piece("0x")
    words = to_uint32_words(x)
    for word in reversed(words):
        builder.append_formatted_piece(word, "08x")
    return builder.build()


# FIXME: overload=(VectorTy,) does not supercede overload=(TensorLikeTy,) so
# we need to do our own dispatching.
@impl(builtins.repr, overload=(TensorLikeTy,))
async def repr_tensorlike_impl(x: Var[TensorLikeTy]):
    value_type = x.get_type()
    if isinstance(value_type, VectorTy):
        return await repr_vector(x)
    if isinstance(value_type, PointerTy):
        return format_hex_le(x)
    if isinstance(value_type, ScalarTy) and value_type.dtype is datatype.mbarrier:
        return format_hex_le(x)
    return tile_repr_tensorlike_impl(x)


@impl(str, overload=(VectorTy,))
async def str_vector_impl(x: Var[VectorTy]):
    return await repr_vector(x)


@impl(hir_stubs.build_formatted_string)
async def build_formatted_string_impl(
    format: StringFormat,
    values: tuple[Var, ...],
) -> Var:
    formatted_values = list(values)

    for piece in format.pieces:
        if not isinstance(piece, FormattedPiece):
            continue
        if piece.format_spec is not None:
            continue

        value = formatted_values[piece.value_idx]
        value_type = value.get_type()

        if isinstance(value_type, VectorTy | PointerTy | ScalarTy):
            formatted_values[piece.value_idx] = await call_function(str, value)

    return await tile_build_formatted_string(
        format,
        tuple(formatted_values),
    )
