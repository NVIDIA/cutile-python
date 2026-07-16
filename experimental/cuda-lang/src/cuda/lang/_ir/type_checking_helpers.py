# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

import cuda.lang._datatype as datatype
from cuda.lang._enums import TMALoadMode
from cuda.lang._ir.ir import add_operation
from cuda.lang._ir.op_defs import LoadPointer, StorePointer, TensorMapAsOpaquePtr
from cuda.lang._ir.type import MemorySpace, ScalarTy, TensorMapTy, VectorTy, PointerTy
from cuda.lang._exception import TypeCheckingError
from cuda.tile import DType
from cuda.tile._memory_model import MemoryOrder
from cuda.tile._ir.ir import Var
from cuda.tile._ir.op_impl import (  # noqa: F401
    make_type_checking_error,
    require_array_type,
    require_optional_constant_enum,
    require_optional_constant_int,
    require_dtype_spec,
)
from cuda.tile._ir.ops import broadcast_to, implicit_cast
from cuda.tile._ir.ops_utils import promote_types
from cuda.tile._ir.type import PointerInfo, TupleTy, TupleValue
from cuda.tile._datatype import is_integral, is_signed
from cuda.lang._datatype import (
    clusterlaunchcontrol_token,
    is_boolean,
    is_float,
    mbarrier,
    opaque_pointer_dtype,
)


def is_none(var: Var):
    return var.is_constant() and var.get_constant() is None


def require_none(var: Var, message: str | None):
    if not is_none(var):
        raise make_type_checking_error(message, var)


def require_array_indices(array: Var, indices: Var) -> tuple[Var, ...]:
    array_ty = require_array_type(array)
    indices_ty = indices.get_type()
    if isinstance(indices_ty, TupleTy):
        tuple_value = indices.get_aggregate()
        assert isinstance(tuple_value, TupleValue)
        index_vars = tuple_value.items
    else:
        index_vars = (indices,)

    if len(index_vars) != array_ty.ndim:
        raise make_type_checking_error(f"Wrong number of indices ({len(index_vars)})"
                                       f" for an array of rank {array_ty.ndim}", indices)

    return tuple(implicit_cast(var, array_ty.index_dtype, "Invalid array index")
                 for var in index_vars)


def require_signed_int_scalar_or_tuple(var: Var) -> tuple[Var, ...]:
    ty = var.get_type()
    have_tuple = isinstance(ty, TupleTy)
    item_types = ty.value_types if have_tuple else (ty,)

    for pos, item_ty in enumerate(item_types):
        if (not isinstance(item_ty, ScalarTy)
                or not is_integral(item_ty.dtype) or not is_signed(item_ty.dtype)):
            what = f"item at position #{pos}" if have_tuple else "given value"
            raise make_type_checking_error(
                f"Expected a signed integer, but {what} has type {item_ty}")

    if have_tuple:
        tuple_value = var.get_aggregate()
        assert isinstance(tuple_value, TupleValue)
        return tuple_value.items
    return (var,)


def require_uniform_tuple_type(var: Var, predicate=None):
    items = var.get_aggregate().items
    if items == ():
        return ()

    first = items[0]
    if predicate is not None:
        predicate(first)

    for item in items[1:]:
        if predicate is not None:
            predicate(item)
        if item.get_type() != first.get_type():
            raise make_type_checking_error("Expected tuple elements to have the same type, but ")

    return items


def require_scalar_type(var: Var,
                        predicate: Callable[[DType], bool] | None = None,
                        message: str | None = None) -> ScalarTy:
    ty = var.get_type()
    if not isinstance(ty, ScalarTy):
        raise make_type_checking_error(f"Expected a scalar, but given value has type {ty}", var)
    if predicate is not None and not predicate(ty.dtype):
        if message is None:
            message = f"Predicate {predicate} failed"
        message += f", but given value has dtype {ty.dtype}"
        raise make_type_checking_error(message, var)

    return ty


def require_integral_scalar_type(var: Var, /, bitwidth: int | None = None):
    ty = require_scalar_type(var)
    if not is_integral(ty.dtype):
        raise make_type_checking_error(f"Expected scalar integral but got {ty}", var)
    if bitwidth is not None and ty.dtype.bitwidth != bitwidth:
        raise make_type_checking_error(
            f"Expected {bitwidth}-bit scalar integral but got {ty}", var
        )


def require_boolean_scalar_type(var: Var):
    ty = require_scalar_type(var)
    if not is_boolean(ty.dtype):
        raise make_type_checking_error(f"Expected scalar integral but got {ty}", var)


def require_clusterlaunchcontrol_token_type(var: Var) -> ScalarTy:
    ty = var.get_type()
    if ty != ScalarTy(clusterlaunchcontrol_token):
        raise make_type_checking_error(f"Expected a clusterlaunchcontrol_token,"
                                       f" but given value has type {ty}")
    return ty


def require_pointer_type(var: Var) -> PointerTy:
    ty = var.get_type()
    if not isinstance(ty, PointerTy):
        raise make_type_checking_error(f"Expected a pointer, got {ty}", var)
    return ty


def require_concrete_pointer_type(var: Var) -> PointerTy:
    ptr_ty = require_pointer_type(var)
    info = PointerInfo(ptr_ty.pointer_dtype)
    if info.opaque:
        raise make_type_checking_error(
            f"Expected concrete pointer type but got {ptr_ty.pointer_dtype}."
            "\nHint: you can use ``cl.bitcast(ptr, cl.pointer_dtype(dtype))``"
            " to cast the pointer to a typed pointer.",
            var,
        )
    return ptr_ty


def require_pointer_in_memory_space(ptr_value, spaces: tuple[MemorySpace, ...]) -> PointerTy:
    ptr_type = require_pointer_type(ptr_value)
    if ptr_type.memory_space not in spaces:
        expected = ' or '.join(map(str, spaces))
        raise make_type_checking_error(
            f"Expected pointer memory space to be {expected} "
            f"but got {ptr_type.memory_space}",
            ptr_value,
        )
    return ptr_type


def require_uniform_int_tuple_type(var: Var):
    return tuple(
        require_uniform_tuple_type(
            var, lambda element: require_scalar_type(element, is_integral)
        )
    )


def tensor_map_descriptor_like(var: Var):
    ty = var.get_type()
    match ty:
        case TensorMapTy():
            result_ty = PointerTy(opaque_pointer_dtype())
            return add_operation(TensorMapAsOpaquePtr, result_ty, tensor_map=var)
        case PointerTy(pointer_dtype=dtype):
            info = PointerInfo(dtype)
            if not info.opaque or info.memory_space is not MemorySpace.GENERIC:
                raise make_type_checking_error(
                    "Expected tensor map or opaque tensor map pointer in generic "
                    f"memory space but got {ty}",
                    var,
                )
            return var

    raise make_type_checking_error(
        f"Expected tensor map or tensor map pointer but got {ty}", var
    )


def require_tensor_map_ty(var: Var) -> TensorMapTy:
    ty = var.get_type()
    if not isinstance(ty, TensorMapTy):
        raise make_type_checking_error(f"Expected a tensor map, got {ty}", var)
    return ty


def validate_tensor_map_load_mode(map_ty: TensorMapTy, mode: TMALoadMode) -> None:
    """Validate constraints imposed by a TMA load mode on a typed TensorMap."""
    if mode is TMALoadMode.TILE_GATHER4 and (
        len(map_ty.tile_shape) != 2 or map_ty.tile_shape[1] != 1
    ):
        raise TypeCheckingError(
            "TILE_GATHER4 requires a rank-2 tensor map with tile_shape[1] == 1"
        )


def require_optional(var: Var, requirement_if_not_none: Callable[[Var], Any]):
    if is_none(var):
        return None
    return requirement_if_not_none(var)


def require_optional_alignment(alignment: Var) -> int | None:
    alignment = require_optional_constant_int(alignment)

    if alignment is None:
        return None

    if alignment <= 0 or alignment & (alignment - 1):
        raise make_type_checking_error("alignment must be a positive power of two")

    return alignment


def require_pointer_memory_order(
    operation: type[LoadPointer] | type[StorePointer],
    ordering_var: Var,
):
    ordering = require_optional_constant_enum(ordering_var, MemoryOrder)
    if ordering in operation.valid_orderings:
        return ordering

    formatted_expected = ", ".join(
        "None" if order is None else str(order) for order in operation.valid_orderings
    )
    operation_name = "load" if operation is LoadPointer else "store"
    raise make_type_checking_error(
        f"Invalid memory order for Pointer.{operation_name}. "
        f"Got {ordering}, expected one of {formatted_expected}"
    )


def require_mbarrier_ptr(
    mbar: Var,
    spaces: tuple[MemorySpace, ...] = (
        MemorySpace.SHARED,
        MemorySpace.SHARED_CLUSTER,
    ),
) -> PointerTy:
    mbar_ptr_type = require_pointer_in_memory_space(mbar, spaces)
    if mbar_ptr_type.opaque or mbar_ptr_type.pointee_dtype is not mbarrier:
        raise make_type_checking_error(f"Expected a pointer to an mbarrier, got {mbar}", mbar)
    return mbar_ptr_type


def require_vector_type(var: Var, length: int | None = None) -> VectorTy:
    ty = var.get_type()
    if not isinstance(ty, VectorTy):
        raise make_type_checking_error(f"Expected a vector, got {ty}", var)
    if length is not None and ty.length != length:
        raise make_type_checking_error(f"Expected a vector of length {length}, got {ty}", var)
    return ty


def require_scalar_or_vector_type(var: Var, dtype_predicate=None) -> VectorTy | ScalarTy:
    ty = var.get_type()

    match ty:
        case ScalarTy() as st:
            dtype = st.dtype
        case VectorTy() as vt:
            dtype = vt.element_dtype
        case _:
            raise make_type_checking_error(
                f"Expected a scalar or vector type but got {ty}", var
            )

    if dtype_predicate is not None and not dtype_predicate(dtype):
        raise make_type_checking_error(
            "Expected scalar or vector to satisfy constraint "
            f"{dtype_predicate.__name__} but got {ty}", var
        )

    return ty


def require_scalar_or_vector_float_type(var: Var) -> VectorTy | ScalarTy:
    return require_scalar_or_vector_type(var, is_float)


def broadcast_to_same_shape(x: Var, y: Var) -> tuple[Var, Var]:
    x_ty = require_scalar_or_vector_type(x)
    y_ty = require_scalar_or_vector_type(y)
    match x_ty, y_ty:
        case ScalarTy(), VectorTy() as vt:
            x = broadcast_to(x, vt.tensor_shape())
        case VectorTy() as vt, ScalarTy():
            y = broadcast_to(y, vt.tensor_shape())
        case VectorTy() as vt1, VectorTy() as vt2 if vt1.length != vt2.length:
            raise make_type_checking_error(
                f"Expected scalar and vector with compatible shape but got {vt1} and {vt2}",
            )
    return x, y


def common_type(x: Var, y: Var):
    x_ty = x.get_loose_type()
    y_ty = y.get_loose_type()

    if not datatype.is_arithmetic(x_ty.tensor_dtype()):
        raise make_type_checking_error(
            f"Left-hand side has non-arithmetic dtype {x_ty.tensor_dtype()}", x
        )
    if not datatype.is_arithmetic(y_ty.tensor_dtype()):
        raise make_type_checking_error(
            f"Right-hand side has non-arithmetic dtype {y_ty.tensor_dtype()}", y
        )

    return promote_types(x_ty, y_ty, x.ctx.typing_hooks)


def optional_cast(var, dtype, context: str):
    if is_none(var):
        return None
    return implicit_cast(var, dtype, context)
