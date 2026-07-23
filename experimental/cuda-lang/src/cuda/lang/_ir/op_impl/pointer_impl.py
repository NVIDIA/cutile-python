# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import operator

import cuda.lang._datatype as datatype
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.ir import Var, add_operation
from cuda.lang._ir.type import (
    ArrayTy,
    ArrayValue,
    MemorySpace,
    PointerInfoTy,
    PointerTy,
    ScalarTy,
    Type,
    VectorTy,
)
from cuda.lang._ir.op_defs import LoadPointer, StorePointer, ReinterpretPointerAsArray
from cuda.lang._ir.type_checking_helpers import (
    require_concrete_pointer_type,
    require_optional_alignment,
    require_array_indices,
    require_pointer_type,
    require_pointer_memory_order,
    require_scalar_type,
)
from cuda.tile._datatype import (
    PointerInfo,
    is_pointer_dtype,
    opaque_pointer_dtype,
    pointer_dtype,
    uint64,
)
from cuda.tile._ir.arithmetic_ops import astype, binary_arithmetic_tensorlike_raw
from cuda.tile._ir.cast_ops import address_space_cast, implicit_cast, reinterpret_pointer
from cuda.tile._ir.core_ops import bind_method, loosely_typed_const, strictly_typed_const
from cuda.tile._ir.ir import add_operation_variadic
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    WILDCARD,
    require_array_type,
    require_constant_bool,
    require_constant_enum,
    require_constant_int_tuple,
    require_constant_pointer_info,
    require_constant_str,
    require_dtype_spec,
    require_optional_constant_int,
)
from cuda.tile._ir.ops import PointerOffset
from cuda.tile._memory_model import MemoryOrder
from ..._stub import core_api
from ..._stub.core_api import Array
from ..._stub.types import Pointer


_registry = ImplRegistry()
impl = _registry.impl


def pointer_impl_registry() -> ImplRegistry:
    return _registry


def contiguous_strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


# -------------------------------------------------------------------------------------
# Pointer dtype APIs
# -------------------------------------------------------------------------------------

@impl(is_pointer_dtype)
def is_pointer_dtype_impl(dtype: Var) -> Var:
    dtype = require_dtype_spec(dtype)
    return loosely_typed_const(is_pointer_dtype(dtype))


@impl(pointer_dtype)
def pointer_dtype_impl(pointee_dtype: Var, memory_space: Var) -> Var:
    pointee_dtype = require_dtype_spec(pointee_dtype)
    memory_space = require_constant_enum(memory_space, MemorySpace)
    res = pointer_dtype(pointee_dtype, memory_space)
    return loosely_typed_const(res)


@impl(opaque_pointer_dtype)
def opaque_pointer_dtype_impl(memory_space: Var) -> Var:
    memory_space = require_constant_enum(memory_space, MemorySpace)
    res = opaque_pointer_dtype(memory_space)
    return loosely_typed_const(res)


@impl(PointerInfo)
def pointer_info_impl(dtype: Var) -> Var:
    dtype = require_dtype_spec(dtype)
    try:
        res = PointerInfo(dtype)
    except TypeError as e:
        raise TypeCheckingError(str(e))

    return loosely_typed_const(res)


@impl(getattr, overload=(PointerInfoTy, "opaque"))
def pointer_info_opaque_impl(object: Var, name: Var) -> Var:
    info = require_constant_pointer_info(object)
    return loosely_typed_const(info.opaque)


@impl(getattr, overload=(PointerInfoTy, "pointee_dtype"))
def pointer_info_pointee_dtype_impl(object: Var, name: Var) -> Var:
    info = require_constant_pointer_info(object)
    try:
        pointee_dtype = info.pointee_dtype
    except ValueError as e:
        raise TypeCheckingError(str(e))

    return loosely_typed_const(pointee_dtype)


@impl(getattr, overload=(PointerInfoTy, "memory_space"))
def pointer_info_memory_space_impl(object: Var, name: Var) -> Var:
    info = require_constant_pointer_info(object)
    return loosely_typed_const(info.memory_space)


# -------------------------------------------------------------------------------------


def array_base_pointer_type(array_ty: ArrayTy) -> PointerTy:
    return PointerTy(pointer_dtype(array_ty.dtype, array_ty.memory_space))


def _get_array_base_pointer(array: Var) -> Var:
    array_ty = require_array_type(array)
    array_val = array.get_aggregate()
    assert isinstance(array_val, ArrayValue)
    base_ptr = array_val.base_ptr
    expected_type = array_base_pointer_type(array_ty)
    if base_ptr.get_type() != expected_type:
        raise TypeCheckingError(
            "Array base pointer type does not match expected type: "
            f"{expected_type=}, got {base_ptr.get_type()}"
        )
    return base_ptr


def _array_linear_offset(array: Var, indices: tuple[Var, ...]) -> Var:
    array_val = array.get_aggregate()
    zero = strictly_typed_const(0, ScalarTy(uint64))
    offset = zero
    if len(indices) != len(array_val.strides):
        raise TypeCheckingError(
            f"Expected {len(array_val.strides)} indices but got {len(indices)}"
        )
    for index, stride in zip(indices, array_val.strides, strict=True):
        index = astype(index, datatype.uint64)
        stride = astype(stride, datatype.uint64)
        scaled = binary_arithmetic_tensorlike_raw("mul", index, stride)
        offset = binary_arithmetic_tensorlike_raw("add", offset, scaled)
    return offset


def _array_get_element_pointer(array: Var, indices: tuple[Var, ...]) -> Var:
    base_pointer = _get_array_base_pointer(array)
    offset = _array_linear_offset(array, indices)
    return add_operation(
        PointerOffset,
        base_pointer.get_type(),
        pointer=base_pointer,
        offset=offset,
    )


@impl(getattr, overload=(ArrayTy, "get_base_pointer"))
@impl(getattr, overload=(ArrayTy, "get_element_pointer"))
def getattr_array_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(Array, name)
    return bind_method(object, unbound_func)


@impl(Array.get_base_pointer)
def array_get_base_pointer_impl(self: Var) -> Var:
    return _get_array_base_pointer(self)


@impl(Array.get_element_pointer)
def array_get_element_pointer_impl(self: Var, indices: Var) -> Var:
    return _array_get_element_pointer(self, require_array_indices(self, indices))


@impl(operator.getitem, overload=(PointerTy, WILDCARD))
def pointer_getitem(object: Var[PointerTy], key: Var[Type]):
    ptr_ty = require_concrete_pointer_type(object)
    pointer = pointer_with_offset(object, key)
    return add_operation(
        LoadPointer,
        ScalarTy(ptr_ty.pointee_dtype),
        pointer=pointer,
        volatile=False,
        alignment=None,
        memory_order=None,
    )


@impl(operator.setitem, overload=(PointerTy, WILDCARD, WILDCARD))
def pointer_setitem(object: Var[PointerTy], key: Var[Type], value: Var[Type]):
    ptr_ty = require_concrete_pointer_type(object)
    pointer = pointer_with_offset(object, key)
    value = astype(value, ptr_ty.pointee_dtype)
    add_operation_variadic(
        StorePointer,
        (),
        pointer=pointer,
        value=value,
        alignment=None,
        volatile=False,
        memory_order=None,
    )


@impl(operator.getitem, overload=(ArrayTy, WILDCARD))
def array_getitem(object: Var, key: Var) -> Var:
    array_ty = require_array_type(object)
    indices = require_array_indices(object, key)
    pointer = _array_get_element_pointer(object, indices)
    return add_operation(
        LoadPointer,
        PointerTy(array_ty.dtype) if is_pointer_dtype(array_ty.dtype) else ScalarTy(array_ty.dtype),
        pointer=pointer,
        alignment=None,
        volatile=False,
    )


@impl(operator.setitem, overload=(ArrayTy, WILDCARD, WILDCARD))
def array_setitem(object: Var, key: Var, value: Var):
    array_ty = require_array_type(object)
    require_scalar_type(value)
    value = astype(value, array_ty.dtype)
    indices = require_array_indices(object, key)
    pointer = _array_get_element_pointer(object, indices)
    add_operation_variadic(
        StorePointer,
        (),
        pointer=pointer,
        value=value,
        alignment=None,
        volatile=False,
        memory_order=None,
    )


def _require_atomic_scalar_dtype(dtype: datatype.DType, operation: str) -> None:
    """Atomic accesses operate on a single value of power-of-two byte size."""
    total_bytes, remainder = divmod(dtype.bitwidth, 8)
    if remainder or (total_bytes & (total_bytes - 1)):
        raise TypeCheckingError(
            f"Type {dtype} cannot be accessed atomically; atomic {operation}s "
            f"require a pointee whose size is a power-of-two number of bytes"
        )


def pointer_load(
    pointer: Var,
    count: Var,
    alignment: Var,
    volatile: Var,
    memory_order: Var,
) -> Var:
    pointee_dtype = require_pointer_type(pointer).pointee_dtype
    count = require_optional_constant_int(count)
    volatile = require_constant_bool(volatile)
    alignment = require_optional_alignment(alignment)
    memory_order = require_pointer_memory_order(LoadPointer, memory_order)
    is_atomic = memory_order not in (None, MemoryOrder.WEAK)
    if count is None or count == 1:
        if is_atomic:
            _require_atomic_scalar_dtype(pointee_dtype, "load")
            if alignment is None:
                alignment = pointee_dtype.bitwidth // 8
        result_ty = ScalarTy(pointee_dtype)
    else:
        if is_atomic:
            raise TypeCheckingError(
                "Atomic loads must be scalar; a vector load (count > 1) cannot be atomic"
            )
        result_ty = VectorTy(pointee_dtype, count)
    return add_operation(
        LoadPointer,
        result_ty,
        pointer=pointer,
        volatile=volatile,
        alignment=alignment,
        memory_order=memory_order,
    )


def pointer_store(
    pointer: Var,
    value: Var,
    alignment: Var,
    volatile: Var,
    memory_order: Var,
) -> None:
    pointer_ty = require_pointer_type(pointer)
    volatile = require_constant_bool(volatile)
    alignment = require_optional_alignment(alignment)
    memory_order = require_pointer_memory_order(StorePointer, memory_order)

    pointee_dtype = pointer_ty.pointee_dtype
    value = implicit_cast(value, pointee_dtype,
                          "Stored value type is incompatible with pointer type")

    is_atomic = memory_order not in (None, MemoryOrder.WEAK)
    if is_atomic:
        if isinstance(value.get_type(), VectorTy):
            raise TypeCheckingError(
                "Atomic stores must be scalar; a vector value cannot be stored atomically"
            )
        _require_atomic_scalar_dtype(pointee_dtype, "store")
        if alignment is None:
            alignment = pointee_dtype.bitwidth // 8

    add_operation_variadic(
        StorePointer,
        (),
        pointer=pointer,
        value=value,
        volatile=volatile,
        alignment=alignment,
        memory_order=memory_order,
    )


def pointer_with_offset(pointer: Var, offset: Var) -> Var:
    require_pointer_type(pointer)
    ty = require_scalar_type(offset)
    if not datatype.is_integral(ty.dtype):
        raise TypeCheckingError("Only integers cna be used to take the offset of a pointer")
    return add_operation(
        PointerOffset,
        pointer.get_type(),
        pointer=pointer,
        offset=offset,
    )


@impl(getattr, overload=(PointerTy, "opaque"))
def pointer_opaque_impl(object: Var[PointerTy], name: Var):
    return loosely_typed_const(object.get_type().opaque)


@impl(getattr, overload=(PointerTy, "pointee_dtype"))
def pointer_pointee_dtype_impl(object: Var[PointerTy], name: Var):
    ty = object.get_type()
    if ty.opaque:
        raise TypeCheckingError("Opaque pointers have no pointee_dtype")
    return loosely_typed_const(ty.pointee_dtype)


@impl(getattr, overload=(PointerTy, "memory_space"))
def pointer_memory_space_impl(object: Var[PointerTy], name: Var):
    return loosely_typed_const(object.get_type().memory_space)


@impl(getattr, overload=(PointerTy, "load"))
@impl(getattr, overload=(PointerTy, "store"))
def getattr_pointer_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(Pointer, name)
    return bind_method(object, unbound_func)


@impl(Pointer.load)
def pointer_load_impl(
    self: Var,
    count: Var,
    alignment: Var,
    volatile: Var,
    memory_order: Var,
) -> Var:
    return pointer_load(self, count, alignment, volatile, memory_order)


@impl(Pointer.store)
def pointer_store_impl(
    self: Var,
    value: Var,
    alignment: Var,
    volatile: Var,
    memory_order: Var,
) -> None:
    pointer_store(self, value, alignment, volatile, memory_order)


@impl(core_api.address_space_cast)
def address_space_cast_impl(value: Var, memory_space: Var) -> Var:
    memory_space = require_constant_enum(memory_space, MemorySpace)
    return address_space_cast(value, memory_space)


@impl(core_api.reinterpret_pointer_as_array)
def reinterpret_pointer_as_array_impl(pointer: Var, dtype: Var, shape: Var, strides: Var) -> Var:
    if not strides.is_constant() or strides.get_constant() is not None:
        raise TypeCheckingError(
            "Reinterpreting a pointer as an array with "
            "non-default strides is not yet implemented."
        )
    pointer_ty = require_pointer_type(pointer)
    shape = require_constant_int_tuple(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    strides = contiguous_strides_from_shape(shape)
    memory_space = pointer_ty.memory_space

    typed_pointer_ty = PointerTy(pointer_dtype(dtype, memory_space))
    if pointer.get_type() != typed_pointer_ty:
        pointer = reinterpret_pointer(pointer, typed_pointer_ty.pointer_dtype)
    index_dtype = datatype.int32
    array_ty = ArrayTy(
        dtype,
        shape=shape,
        strides=strides,
        typing_hooks=pointer.ctx.typing_hooks,
        index_dtype=index_dtype,
        memory_space=memory_space,
    )
    result = add_operation(
        ReinterpretPointerAsArray,
        array_ty,
        pointer=pointer,
    )
    # FIXME: it seems that the index dtype should be derived from the dtype of shape/strides instead
    size_ty = ScalarTy(index_dtype)
    shape_vars = tuple(strictly_typed_const(extent, size_ty) for extent in shape)
    stride_vars = tuple(strictly_typed_const(extent, size_ty) for extent in strides)
    result.set_aggregate(ArrayValue(pointer, shape_vars, stride_vars))
    return result
