# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._datatype as datatype
from cuda.lang._enums import (
    MatrixLoadShape,
    MatrixLoadSourceFormat,
    MatrixStoreShape,
)
from cuda.lang._exception import InvalidValueError, TypeCheckingError
from cuda.lang._ir.ir import Var
from cuda.lang._ir.op_defs import RawNVVMIntrinsic
from cuda.lang._ir.op_impl.vector_impl import (
    vector_getitem,
    vector_undef,
    vector_with_item,
)
from cuda.lang._ir.type import MemorySpace, ScalarTy, VectorTy
from cuda.lang._ir.type_checking_helpers import (
    is_none,
    require_pointer_in_memory_space,
)
from cuda.lang._stub import load_store_matrix
from cuda.tile._datatype import is_integral
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_bool,
    require_constant_enum,
    require_constant_int,
)
from cuda.tile._ir.core_ops import strictly_typed_const
from cuda.tile._ir.ir import add_operation_variadic


_registry = ImplRegistry()
impl = _registry.impl


def matrix_impl_registry() -> ImplRegistry:
    return _registry


@impl(load_store_matrix.load_matrix)
def load_matrix_impl(
    src: Var,
    shape: Var,
    count: Var,
    transpose: Var,
    source_format: Var,
) -> Var:
    require_pointer_in_memory_space(src, (MemorySpace.SHARED,))
    shape_value = require_constant_enum(shape, MatrixLoadShape)
    count_value = require_constant_int(count)
    transpose_value = require_constant_bool(transpose)
    source_format_value = (
        None
        if is_none(source_format)
        else require_constant_enum(source_format, MatrixLoadSourceFormat)
    )
    if count_value not in (1, 2, 4):
        raise InvalidValueError("count must be 1, 2, or 4")
    register_count = count_value * (
        2 if shape_value is MatrixLoadShape.M16N16 else 1
    )
    register_type = ScalarTy(datatype.int32)

    name = f"llvm.nvvm.ldmatrix.sync.aligned.{shape_value.value}.x{count_value}"

    if transpose_value:
        name += ".trans"

    if source_format_value is MatrixLoadSourceFormat.B6X16_P32:
        name += ".b8x16.b6x16_p32"
    elif source_format_value is MatrixLoadSourceFormat.B4X16_P64:
        name += ".b8x16.b4x16_p64"
    else:
        name += ".b16" if shape_value is MatrixLoadShape.M8N8 else ".b8"

    registers = add_operation_variadic(
        RawNVVMIntrinsic,
        (register_type,) * register_count,
        intrinsic=name,
        operands_=(src,),
    )
    if register_count == 1:
        return registers[0]

    result = vector_undef(VectorTy(datatype.int32, register_count))
    for index, register in enumerate(registers):
        result = vector_with_item(result, index, register)
    return result


def _store_register_count(values: Var) -> int:
    value_type = values.get_type()
    match value_type:
        case ScalarTy() as st:
            dtype = st.dtype
            count = 1
        case VectorTy() as vt:
            dtype = vt.element_dtype
            count = vt.length
        case _:
            raise TypeCheckingError(
                "Expected a scalar or vector of 32-bit integers"
            )

    if not is_integral(dtype) or dtype.bitwidth != 32:
        # TODO: is this too restrictive? should we bitcast if the operand is
        # not integral but is 32 bits wide?
        raise TypeCheckingError(
            "Expected a scalar or vector of 32-bit integers, "
            f"but got {value_type}"
        )
    if count not in (1, 2, 4):
        raise InvalidValueError("Matrix store register count must be 1, 2, or 4")
    return count


@impl(load_store_matrix.store_matrix)
def store_matrix_impl(
    dst: Var,
    values: Var,
    shape: Var,
    transpose: Var,
) -> None:
    require_pointer_in_memory_space(dst, (MemorySpace.SHARED,))
    shape_value = require_constant_enum(shape, MatrixStoreShape)
    transpose_value = require_constant_bool(transpose)
    register_count = _store_register_count(values)
    if shape_value is MatrixStoreShape.M16N8 and not transpose_value:
        raise InvalidValueError("M16N8 requires transpose=True")

    if register_count == 1:
        registers = (values,)
    else:
        index_type = ScalarTy(datatype.int32)
        registers = tuple(
            vector_getitem(values, strictly_typed_const(index, index_type))
            for index in range(register_count)
        )

    name = f"llvm.nvvm.stmatrix.sync.aligned.{shape_value.value}.x{register_count}"
    if transpose_value:
        name += ".trans"
    name += ".b16" if shape_value is MatrixStoreShape.M8N8 else ".b8"

    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=name,
        operands_=(dst, *registers),
    )
