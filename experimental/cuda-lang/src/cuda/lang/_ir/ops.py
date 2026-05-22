# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import math
import re
import operator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from cuda.tile._memory_model import MemoryOrder
from cuda.tile._ir.op_impl import (
    require_optional_constant_int,
    require_tuple_type,
    require_constant_str,
    require_dtype_spec,
    require_constant_int_tuple,
    require_constant_int,
    require_constant_enum,
    require_optional_constant_enum,
    require_index_or_index_tuple_type,
    require_array_type,
    WILDCARD,
    require_tile_type, require_constant_bool, require_constant_pointer_info,
    require_scalar_pointer_type,
)
from cuda.lang._ir.type import (
    LocalArrayContextManagerTy, ContextManagerState, TensorMapTy,
    dtype_to_tensor_map_type, ArrayValue, PointerInfoTy
)
from cuda.tile._ir.ops import (
    binary_arithmetic,
    loosely_typed_const,
    tile_impl_registry,
    add_impl as tile_add_impl,
    bind_method,
    build_tuple,
    strictly_typed_const,
    astype,
    raw_binary_arithmetic,
    Return,
    return_,
    Assign,
    AssumeBounded,
    AssumeDivBy,
    MakeTensorView,
    MakeDummy,
    RawBinaryArithmeticOperation,
    RawComparisonOperation,
    RawBinaryBitwiseOperation,
    RawBitwiseShiftOperation,
    TileAsType,
    TypedConst,
    IfElse,
    RawWhereOperation,
    EndBranch,
    Loop,
    Continue,
    Break,
    PointerOffset,
    TilePrintf,
    printf_impl,
    Unary, implicit_cast, address_space_cast, reinterpret_pointer, AddrSpaceCast,
    ReinterpretPointer,
)
from cuda.tile._ir.ir import MemoryEffect, make_aggregate
from cuda.lang._exception import TileCompilerError, TileTypeError
import cuda.lang._datatype as datatype
from cuda.tile._datatype import (
    is_pointer_dtype,
    pointer_dtype,
    PointerInfo,
    opaque_pointer_dtype,
)
import cuda.lang._mlir as mlir

from .. import _stub as stub

from .type import (
    MemorySpace,
    Type,
    make_vector_ty,
    is_vector_ty,
    ArrayTy,
    TileTy,
    TupleTy,
    TupleValue
)

from .ir import (
    Operation,
    Block,
    attribute,
    operand,
    Var,
    add_operation,
    format_var,
    LocalArrayContextManagerValue,
)
from .._stub import TensorMapSwizzle, MbarrierScope
from cuda.tile._ir import hir_stubs
from cuda.tile._ir.typing_support import I32_TY, U64_TY, BOOL_TY

cuda_lang_impl_registry = tile_impl_registry.clone()
impl = cuda_lang_impl_registry.impl
overload_dispatcher = cuda_lang_impl_registry.overload_dispatcher


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
        raise TileTypeError(str(e))

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
        raise TileTypeError(str(e))

    return loosely_typed_const(pointee_dtype)


@impl(getattr, overload=(PointerInfoTy, "memory_space"))
def pointer_info_memory_space_impl(object: Var, name: Var) -> Var:
    info = require_constant_pointer_info(object)
    return loosely_typed_const(info.memory_space)


# -------------------------------------------------------------------------------------


@dataclass(eq=False)
class StorePointer(Operation, opcode="store_pointer", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    alignment: Optional[int] = attribute()
    volatile: bool = attribute(default=False)
    ordering: Optional[MemoryOrder] = attribute(default=None)

    valid_orderings = (
        None,
        MemoryOrder.WEAK,
        MemoryOrder.RELAXED,
        MemoryOrder.RELEASE,
    )


@dataclass(eq=False)
class LoadPointer(Operation, opcode="load_pointer", memory_effect=MemoryEffect.LOAD):
    pointer: Var = operand()
    alignment: Optional[int] = attribute()
    volatile: bool = attribute(default=False)
    ordering: Optional[MemoryOrder] = attribute(default=None)

    valid_orderings = (
        None,
        MemoryOrder.WEAK,
        MemoryOrder.RELAXED,
        MemoryOrder.ACQUIRE,
    )


@dataclass(eq=False)
class ArrayGetItem(Operation, opcode="array_getitem", memory_effect=MemoryEffect.LOAD):
    x: Var = operand()
    indices: tuple[Var, ...] = operand()


@dataclass(eq=False)
class ArraySetItem(Operation, opcode="array_setitem", memory_effect=MemoryEffect.STORE):
    x: Var = operand()
    indices: tuple[Var, ...] = operand()
    value: Var = operand()


@dataclass(eq=False)
class ReinterpretPointerAsArray(Operation, opcode="reinterpret_ptr_as_array"):
    pointer: Var = operand()


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
    raise TileTypeError(
        f"Invalid memory order for Pointer.{operation_name}. "
        f"Got {ordering}, expected one of {formatted_expected}"
    )


def require_array_indices(array: Var, indices: Var) -> tuple[Var, ...]:
    array_ty = require_array_type(array)
    key_ty = require_index_or_index_tuple_type(indices)
    if isinstance(key_ty, TileTy):
        if key_ty.ndim == 0:
            return (indices,)
        raise TileTypeError(
            "Cannot index an array with a tile, use a tuple of indices or a scalar instead"
        )
    if isinstance(key_ty, TupleTy):
        if len(key_ty.value_types) != array_ty.ndim:
            raise TileTypeError(
                f"Expected {array_ty.ndim} indices but got {len(key_ty.value_types)}"
            )
        tuple_value = indices.get_aggregate()
        assert isinstance(tuple_value, TupleValue)
        return tuple_value.items
    raise TileTypeError(
        f"Expected a tuple of indices or a single index, but got {indices.get_type()}"
    )


def _array_base_pointer_type(array_ty: ArrayTy) -> TileTy:
    return TileTy(pointer_dtype(array_ty.dtype, array_ty.memory_space))


def _get_array_base_pointer(array: Var) -> Var:
    array_ty = require_array_type(array)
    array_val = array.get_aggregate()
    assert isinstance(array_val, ArrayValue)
    base_ptr = array_val.base_ptr
    expected_type = _array_base_pointer_type(array_ty)
    if base_ptr.get_type() != expected_type:
        raise TileTypeError(
            "Array base pointer type does not match expected type: "
            f"{expected_type=}, got {base_ptr.get_type()}"
        )
    return base_ptr


def _array_linear_offset(array: Var, indices: tuple[Var, ...]) -> Var:
    array_val = array.get_aggregate()
    zero = strictly_typed_const(0, U64_TY)
    offset = zero
    if len(indices) != len(array_val.strides):
        raise TileTypeError(
            f"Expected {len(array_val.strides)} indices but got {len(indices)}"
        )
    for index, stride in zip(indices, array_val.strides, strict=True):
        index = astype(index, datatype.uint64)
        stride = astype(stride, datatype.uint64)
        scaled = raw_binary_arithmetic("mul", index, stride)
        offset = raw_binary_arithmetic("add", offset, scaled)
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
    unbound_func = getattr(stub.Array, name)
    return bind_method(object, unbound_func)


@impl(stub.Array.get_base_pointer)
def _m_array_get_base_pointer_impl(self: Var) -> Var:
    return _get_array_base_pointer(self)


@impl(stub.Array.get_element_pointer)
def _m_array_get_element_pointer_impl(self: Var, indices: Var) -> Var:
    return _array_get_element_pointer(self, require_array_indices(self, indices))


@impl(operator.getitem, overload=(ArrayTy, WILDCARD))
def array_getitem(object: Var, key: Var) -> Var:
    array_ty = require_array_type(object)
    indices = require_array_indices(object, key)
    pointer = _array_get_element_pointer(object, indices)
    [result] = add_operation(
        LoadPointer,
        (TileTy(array_ty.dtype),),
        pointer=pointer,
        alignment=None,
        volatile=False,
    )
    return result


@impl(operator.setitem, overload=(ArrayTy, WILDCARD, WILDCARD))
def array_setitem(object: Var, key: Var, value: Var):
    array_ty = require_array_type(object)
    require_scalar_tile_type(value)
    value = astype(value, array_ty.dtype)
    indices = require_array_indices(object, key)
    pointer = _array_get_element_pointer(object, indices)
    add_operation(
        StorePointer,
        (),
        pointer=pointer,
        value=value,
        alignment=None,
        volatile=False,
        ordering=None,
    )


class AtomicRMWKind(Enum):
    ADD = auto()
    SUB = auto()
    AND = auto()
    OR = auto()
    XOR = auto()
    MIN = auto()
    MAX = auto()
    INC = auto()
    DEC = auto()


@dataclass(eq=False)
class AtomicRMW(Operation, opcode="atomic_rmw", memory_effect=MemoryEffect.STORE):
    kind: AtomicRMWKind = attribute()
    pointer: Var = operand()
    value: Var = operand()
    memory_order: int = attribute()


@dataclass(eq=False)
class AtomicExchange(Operation, opcode="atomic_exch", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    memory_order: int = attribute()


@dataclass(eq=False)
class AtomicCAS(Operation, opcode="atomic_cas", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    compare: Var = operand()
    value: Var = operand()
    success_memory_order: int = attribute()
    failure_memory_order: int = attribute()


@impl(stub.atomic_add, fixed_args=[AtomicRMWKind.ADD])
@impl(stub.atomic_sub, fixed_args=[AtomicRMWKind.SUB])
@impl(stub.atomic_and, fixed_args=[AtomicRMWKind.AND])
@impl(stub.atomic_or, fixed_args=[AtomicRMWKind.OR])
@impl(stub.atomic_xor, fixed_args=[AtomicRMWKind.XOR])
@impl(stub.atomic_min, fixed_args=[AtomicRMWKind.MIN])
@impl(stub.atomic_max, fixed_args=[AtomicRMWKind.MAX])
@impl(stub.atomic_inc, fixed_args=[AtomicRMWKind.INC])
@impl(stub.atomic_dec, fixed_args=[AtomicRMWKind.DEC])
def atomic_rmw_dispatch_impl(kind: AtomicRMWKind, A: Var, idx: Var, val: Var) -> Var:
    array_ty = require_array_type(A)
    require_scalar_tile_type(val)
    val = astype(val, array_ty.dtype)
    indices = require_array_indices(A, idx)
    pointer = _array_get_element_pointer(A, indices)
    result_ty = TileTy(array_ty.dtype)
    memory_order = mlir.llvm.AtomicOrdering.acq_rel
    return add_operation(
        AtomicRMW,
        result_ty,
        kind=kind,
        pointer=pointer,
        value=val,
        memory_order=memory_order,
    )


@impl(stub.atomic_exch)
def atomic_exch_impl(A: Var, idx: Var, val: Var) -> Var:
    array_ty = require_array_type(A)
    require_scalar_tile_type(val)
    val = astype(val, array_ty.dtype)
    indices = require_array_indices(A, idx)
    pointer = _array_get_element_pointer(A, indices)
    result_ty = TileTy(array_ty.dtype)
    memory_order = mlir.llvm.AtomicOrdering.acq_rel
    return add_operation(
        AtomicExchange,
        result_ty,
        pointer=pointer,
        value=val,
        memory_order=memory_order,
    )


@impl(stub.atomic_cas)
def atomic_cas_impl(A: Var, idx: Var, old: Var, val: Var) -> Var:
    array_ty = require_array_type(A)
    require_scalar_tile_type(val)
    val = astype(val, array_ty.dtype)
    compare_ty = require_tile_type(old)
    if array_ty.dtype != compare_ty.dtype:
        raise TileTypeError(
            f"Expected atomic compare value of type {array_ty.dtype}, got {compare_ty.dtype}"
        )
    indices = require_array_indices(A, idx)
    pointer = _array_get_element_pointer(A, indices)
    result_ty = TileTy(array_ty.dtype)
    success_memory_order = mlir.llvm.AtomicOrdering.acq_rel
    failure_memory_order = mlir.llvm.AtomicOrdering.monotonic
    return add_operation(
        AtomicCAS,
        result_ty,
        pointer=pointer,
        compare=old,
        value=val,
        success_memory_order=success_memory_order,
        failure_memory_order=failure_memory_order,
    )


def require_vector_var(var: Var) -> TileTy:
    ty = var.get_type()
    if not is_vector_ty(ty):
        raise TileTypeError(f"Expected a vector, got {ty}")
    return ty


def _pointer_load(
    pointer: Var,
    count: Var,
    alignment: Var,
    volatile: Var,
    ordering: Var,
) -> Var:
    pointer_tile_ty = require_scalar_pointer_type(pointer)
    pointee_dtype = PointerInfo(pointer_tile_ty.dtype).pointee_dtype
    count = require_optional_constant_int(count)
    volatile = require_constant_bool(volatile)
    alignment = require_optional_alignment(alignment)
    ordering = require_pointer_memory_order(LoadPointer, ordering)
    if ordering not in (None, MemoryOrder.WEAK) and alignment is None:
        raise TileTypeError("Expected explicit alignment on atomic load")
    if count is None or count == 1:
        result_ty = TileTy(pointee_dtype)
    else:
        result_ty = make_vector_ty(pointee_dtype, count)
    [result] = add_operation(
        LoadPointer,
        (result_ty,),
        pointer=pointer,
        volatile=volatile,
        alignment=alignment,
        ordering=ordering,
    )
    return result


def _pointer_store(
    pointer: Var,
    value: Var,
    alignment: Var,
    volatile: Var,
    ordering: Var,
) -> None:
    pointer_tile_ty = require_scalar_pointer_type(pointer)
    volatile = require_constant_bool(volatile)
    alignment = require_optional_alignment(alignment)
    ordering = require_pointer_memory_order(StorePointer, ordering)
    if ordering not in (None, MemoryOrder.WEAK) and alignment is None:
        raise TileTypeError("Expected explicit alignment on atomic store")

    pointee_dtype = PointerInfo(pointer_tile_ty.dtype).pointee_dtype
    value = implicit_cast(value, pointee_dtype,
                          "Stored value type is incompatible with pointer type")

    add_operation(
        StorePointer,
        (),
        pointer=pointer,
        value=value,
        volatile=volatile,
        alignment=alignment,
        ordering=ordering,
    )


def _pointer_with_offset(pointer: Var, offset: Var) -> Var:
    require_scalar_pointer_type(pointer)
    offset = astype(offset, datatype.int64)
    return add_operation(
        PointerOffset,
        pointer.get_type(),
        pointer=pointer,
        offset=offset,
    )


def _is_pointer_type(ty):
    return isinstance(ty, TileTy) and is_pointer_dtype(ty.dtype)


@impl(operator.add)
def add_impl(x: Var, y: Var) -> Var:
    xty, yty = x.get_type(), y.get_type()
    if _is_pointer_type(yty):
        xty, yty = yty, xty
    if _is_pointer_type(xty):
        offset_dtype = require_scalar_tile_type(y).dtype
        if not datatype.is_integral(offset_dtype):
            raise TileTypeError(f"Expected integer pointer offset, got {offset_dtype}")
        return _pointer_with_offset(x, y)
    return tile_add_impl(x, y)


@impl(operator.sub)
def sub_impl(x: Var, y: Var) -> Var:
    xty, yty = x.get_type(), y.get_type()
    if _is_pointer_type(xty):
        offset_dtype = require_scalar_tile_type(y).dtype
        if not datatype.is_integral(offset_dtype):
            raise TileTypeError(f"Expected integer pointer offset, got {offset_dtype}")
        y = astype(y, datatype.int64)
        c0 = loosely_typed_const(0)
        offset = binary_arithmetic('sub', c0, y)
        return _pointer_with_offset(x, offset)
    if _is_pointer_type(yty):
        raise TileTypeError('It is invalid to subtract a pointer from an integer')
    return binary_arithmetic('sub', x, y)


@impl(stub.address_space_cast)
def address_space_cast_impl(value: Var, memory_space: Var) -> Var:
    memory_space = require_constant_enum(memory_space, MemorySpace)
    return address_space_cast(value, memory_space)


@impl(stub.reinterpret_pointer_as_array)
def reinterpret_pointer_as_array_impl(pointer: Var, dtype: Var, shape: Var, strides: Var) -> Var:
    if not strides.is_constant() or strides.get_constant() is not None:
        raise TileTypeError(
            "Reinterpreting a pointer as an array with "
            "non-default strides is not yet implemented."
        )
    pointer_tile_ty = require_scalar_pointer_type(pointer)
    shape = require_constant_int_tuple(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    strides = _contiguous_strides(shape)
    memory_space = PointerInfo(pointer_tile_ty.dtype).memory_space

    typed_pointer_ty = TileTy(pointer_dtype(dtype, memory_space))
    if pointer.get_type() != typed_pointer_ty:
        pointer = reinterpret_pointer(pointer, typed_pointer_ty.dtype)
    index_dtype = datatype.int32
    array_ty = ArrayTy(
        dtype,
        shape=shape,
        strides=strides,
        index_dtype=index_dtype,
        memory_space=memory_space,
    )
    result = add_operation(
        ReinterpretPointerAsArray,
        array_ty,
        pointer=pointer,
    )
    # FIXME: it seems that the index dtype should be derived from the dtype of shape/strides instead
    size_ty = TileTy(index_dtype)
    shape_vars = tuple(strictly_typed_const(extent, size_ty) for extent in shape)
    stride_vars = tuple(strictly_typed_const(extent, size_ty) for extent in strides)
    result.set_aggregate(ArrayValue(pointer, shape_vars, stride_vars))
    return result


@impl(getattr, overload=(TileTy, "element_count"))
def vector_element_count_impl(object: Var, name: Var):
    ty = require_vector_var(object)
    return loosely_typed_const(ty.shape[0])


@impl(getattr, overload=(TileTy, "dtype"))
def tile_dtype_impl(object: Var, name: Var):
    dtype = require_tile_type(object).dtype
    return loosely_typed_const(dtype)


@impl(getattr, overload=(TileTy, "load"))
@impl(getattr, overload=(TileTy, "store"))
def getattr_pointer_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(stub.Pointer, name)
    return bind_method(object, unbound_func)


@impl(stub.Pointer.load)
def pointer_load_impl(
    self: Var,
    count: Var,
    alignment: Var,
    volatile: Var,
    ordering: Var,
) -> Var:
    return _pointer_load(self, count, alignment, volatile, ordering)


@impl(stub.Pointer.store)
def pointer_store_impl(
    self: Var,
    value: Var,
    alignment: Var,
    volatile: Var,
    ordering: Var,
) -> None:
    _pointer_store(self, value, alignment, volatile, ordering)


@dataclass(eq=False)
class Branch(Operation, opcode="br", terminator=True):
    target: Block = attribute()
    args: tuple[Var, ...] = operand()

    def _to_string_rhs(self) -> str:
        return f"{self.op} ^{self.target._name}({', '.join(format_var(arg) for arg in self.args)})"


def branch(target: Block, args: tuple[Var, ...]) -> Branch:
    return add_operation(Branch, (), target=target, args=args)


@dataclass(eq=False)
class CondBranch(Operation, opcode="cond_br", terminator=True):
    cond: Var = operand()
    true_args: tuple[Var, ...] = operand()
    false_args: tuple[Var, ...] = operand()
    true_target: Block = attribute()
    false_target: Block = attribute()

    def _to_string_rhs(self) -> str:
        formatted = f"{self.op} {format_var(self.cond)}"

        formatted += " ^" + self.true_target._name
        formatted += f"({', '.join(format_var(arg) for arg in self.true_args)})"

        formatted += " ^" + self.false_target._name
        formatted += f"({', '.join(format_var(arg) for arg in self.false_args)})"

        return formatted


def cond_branch(
    cond: Var,
    true_args: tuple[Var, ...],
    false_args: tuple[Var, ...],
    true_target: Block,
    false_target: Block,
) -> CondBranch:
    return add_operation(
        CondBranch,
        (),
        cond=cond,
        true_args=true_args,
        false_args=false_args,
        true_target=true_target,
        false_target=false_target,
    )


@dataclass(eq=False)
class AllocLocalMemory(Operation, opcode="alloc_local_memory", memory_effect=MemoryEffect.STORE):
    count: int = attribute()
    alignment: int | None = attribute()


@dataclass(eq=False)
class DeallocLocalMemory(Operation,
                         opcode="dealloc_local_memory",
                         memory_effect=MemoryEffect.STORE):
    ptr: Var = operand()


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def _dtype_byte_width(dtype: datatype.DType) -> int:
    assert dtype.bitwidth % 8 == 0
    return dtype.bitwidth // 8


def require_optional_alignment(alignment: Var) -> int | None:
    alignment = require_optional_constant_int(alignment)

    if alignment is None:
        return None

    if alignment <= 0 or alignment & (alignment - 1):
        raise TileTypeError("alignment must be a positive power of two")

    return alignment


@impl(stub.local_array)
def local_array_impl(shape: Var, dtype: Var, alignment: Var) -> Var:
    shape = require_constant_int_tuple(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    alignment = require_optional_alignment(alignment)
    dtype_byte_width = _dtype_byte_width(dtype)
    if alignment is not None and alignment < dtype_byte_width:
        raise TileTypeError(f"Requested {alignment=} is less than {dtype_byte_width}")

    state = ContextManagerState()
    agg_ty = LocalArrayContextManagerTy(dtype, shape, alignment, state)
    agg_val = LocalArrayContextManagerValue()
    return make_aggregate(agg_val, agg_ty)


@impl(hir_stubs.enter_context, overload=(LocalArrayContextManagerTy,))
def enter_context_local_array_impl(manager: Var):
    mgr_ty = manager.get_type()
    assert isinstance(mgr_ty, LocalArrayContextManagerTy)

    dtype_byte_width = _dtype_byte_width(mgr_ty.dtype)
    if mgr_ty.alignment is not None and mgr_ty.alignment < dtype_byte_width:
        raise TileTypeError(f"Requested alignment {mgr_ty.alignment}"
                            f" is less than item size {dtype_byte_width}")
    strides = _contiguous_strides(mgr_ty.shape)
    index_dtype = datatype.int32
    array_type = ArrayTy(
        mgr_ty.dtype,
        shape=mgr_ty.shape,
        strides=strides,
        index_dtype=index_dtype,
        memory_space=MemorySpace.GENERIC,
    )
    size_ty = TileTy(index_dtype, ())
    shape_vars = tuple(strictly_typed_const(extent, size_ty) for extent in mgr_ty.shape)
    stride_vars = tuple(strictly_typed_const(extent, size_ty) for extent in strides)

    base_ptr = add_operation(
        AllocLocalMemory,
        _array_base_pointer_type(array_type),
        count=math.prod(mgr_ty.shape),
        alignment=mgr_ty.alignment,
    )

    def exit_callback():
        add_operation(DeallocLocalMemory, (), ptr=base_ptr)

    mgr_ty.state.exit_callback = exit_callback

    array_val = ArrayValue(base_ptr, shape_vars, stride_vars)
    return make_aggregate(array_val, array_type)


@dataclass(eq=False)
class GetDynSharedMemoryBasePtr(Operation, opcode="get_dyn_shared_memory_base_ptr"):
    initial_alignment = 1024


def get_dyn_shared_memory_base_ptr():
    result_ty = TileTy(pointer_dtype(datatype.uint8, MemorySpace.SHARED))
    return add_operation(GetDynSharedMemoryBasePtr, result_ty)


@dataclass(eq=False)
class AllocStaticSharedMemory(Operation, opcode="alloc_static_shared_memory",
                              memory_effect=MemoryEffect.STORE):
    count: int = attribute()
    alignment: int | None = attribute()


@dataclass(eq=False)
class AllocDynSharedMemory(Operation, opcode="alloc_dyn_shared_memory",
                           memory_effect=MemoryEffect.STORE):
    shape: tuple[Var, ...] = operand()
    alignment: int | None = attribute()


@impl(stub.shared_array)
def shared_array_impl(shape: Var, dtype: Var, dynamic: Var, alignment: Var) -> Operation:
    dynamic = require_constant_bool(dynamic)

    shape_ty = require_index_or_index_tuple_type(shape)
    if isinstance(shape_ty, TileTy):
        sizes = (shape,)
    else:
        tuple_val = shape.get_aggregate()
        assert isinstance(tuple_val, TupleValue)
        sizes = tuple_val.items

    dtype = require_dtype_spec(dtype)
    alignment = require_optional_alignment(alignment)
    dtype_byte_width = _dtype_byte_width(dtype)
    if alignment is not None and alignment < dtype_byte_width:
        raise TileTypeError(f"Requested {alignment=} is less than {dtype_byte_width=}")
    index_dtype = datatype.int32
    size_ty = TileTy(index_dtype, ())

    ty_strides = []
    ty_shape = []
    total_size = 1
    total_size_var = strictly_typed_const(total_size, size_ty)
    shape_vars = []
    stride_vars = []
    for size_var in reversed(sizes):
        if size_var.is_constant():
            size = size_var.get_constant()
            size_var = strictly_typed_const(size, size_ty)
        else:
            if size_var.get_type().dtype != index_dtype:
                # TODO: allow implicit cast?
                raise TileTypeError(f"Shared memory size must be {index_dtype},"
                                    f" got {size_var.get_type().dtype}")
            size = None
        ty_shape.append(size)
        shape_vars.append(size_var)
        ty_strides.append(total_size)
        stride_vars.append(total_size_var)

        if size is None or total_size is None:
            total_size = None
            total_size_var = raw_binary_arithmetic("mul", total_size_var, size_var)
        else:
            total_size *= size
            total_size_var = strictly_typed_const(total_size, size_ty)

    ty_shape.reverse()
    shape_vars.reverse()
    ty_strides.reverse()
    stride_vars.reverse()

    array_type = ArrayTy(
        dtype,
        shape=tuple(ty_shape),
        strides=tuple(ty_strides),
        index_dtype=index_dtype,
        memory_space=MemorySpace.SHARED,
    )

    if dynamic:
        base_ptr = add_operation(AllocDynSharedMemory,
                                 _array_base_pointer_type(array_type),
                                 shape=sizes,
                                 alignment=alignment)
    else:
        if total_size is None:
            raise TileTypeError("Shape must be constant when `dynamic` is False")

        base_ptr = add_operation(AllocStaticSharedMemory,
                                 _array_base_pointer_type(array_type),
                                 count=total_size,
                                 alignment=alignment)

    array_value = ArrayValue(base_ptr=base_ptr, shape=tuple(shape_vars), strides=tuple(stride_vars))
    return make_aggregate(array_value, array_type)


@dataclass(eq=False)
class SyncThreads(Operation, opcode="syncthreads", memory_effect=MemoryEffect.STORE):
    pass


@impl(stub.syncthreads)
def syncthreads_impl() -> Operation:
    return add_operation(SyncThreads, None,)


@impl(stub.elect_sync)
def elect_sync_impl(membermask) -> Var:
    mask = require_constant_int(membermask)
    mask = strictly_typed_const(mask & 0xffffffff, I32_TY)

    _, is_elected = add_operation(RawNVVMIntrinsic,
                                  (I32_TY, BOOL_TY),
                                  intrinsic="llvm.nvvm.elect.sync",
                                  operands_=(mask,))
    return is_elected


impl(stub.printf)(printf_impl)


@dataclass(eq=False)
class InlinePTX(Operation, opcode="inline_ptx", memory_effect=MemoryEffect.STORE):
    ptx_code: str = attribute()
    read_only_operands: tuple[Var, ...] = operand()
    write_only_operands: tuple[datatype.DType, ...] = attribute()
    read_write_operands: tuple[Var, ...] = operand()

    class RMWMode(Enum):
        READ_ONLY = auto()
        WRITE_ONLY = auto()
        READ_WRITE = auto()


@dataclass(eq=False, frozen=True)
class InlinePTXOperand:
    mode: InlinePTX.RMWMode
    type_code: str
    value: Var | datatype.DType


def require_inline_ptx_pair(var: Var) -> tuple[Var, Var]:
    pair_ty = var.get_type()
    if not isinstance(pair_ty, TupleTy) or len(pair_ty.value_types) != 2:
        raise TileTypeError(
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

_INLINE_PTX_TYPECODES = {
    "h",
    "r",
    "l",
    "f",
    "d",
    "C",
}

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
        raise TileTypeError(
            f"Invalid inline_ptx constraint {constraint_str}, expected length 1 or 2"
        )

    prefix = constraint_str[0:-1]
    type_char = constraint_str[-1]

    mode = _INLINE_PTX_MODE_FROM_PREFIX.get(prefix)
    if mode is None:
        raise TileTypeError(
            f"Unknown constraint rmw modifier {prefix!r}, expected "
            "'' (meaning readonly), '+' (meaning readwrite), or '=' (meaning writeonly)"
        )

    if type_char not in _INLINE_PTX_TYPECODES:
        expected = ", ".join(_INLINE_PTX_TYPECODES)
        raise TileTypeError(
            f"Unknown constraint dtype {type_char!r}, expected one of {expected}"
        )

    return constraint_str, mode, type_char


def validate_inline_ptx_operand(
    constraint_str: str, mode: InlinePTX.RMWMode, type_char: str, value: Var
) -> InlinePTXOperand:
    if mode is InlinePTX.RMWMode.WRITE_ONLY:
        if type_char == "C":
            # write-only arguments require specifying the output data type, but we don't
            # expose a dtype for pointers. Disallow this for now.
            raise TileTypeError("Write-only pointer outputs are not supported for inline_ptx")

        actual_dtype = require_dtype_spec(value)
        expected_dtype = _INLINE_PTX_SCALAR_DTYPE_FROM_TYPECODE[type_char]
        if actual_dtype != expected_dtype:
            raise TileTypeError(
                f"Expected dtype {expected_dtype} for constraint "
                f"{constraint_str}, got {actual_dtype}"
            )
        return InlinePTXOperand(mode=mode, type_code=type_char, value=actual_dtype)

    if type_char == "C":
        require_scalar_pointer_type(value)
        return InlinePTXOperand(mode=mode, type_code=type_char, value=value)

    value_ty = require_tile_type(value)
    if value_ty.shape != ():
        raise TileTypeError(
            f"Expected a scalar value for constraint {constraint_str!r}, but got {value_ty}"
        )

    actual_dtype = value_ty.dtype
    expected_dtype = _INLINE_PTX_SCALAR_DTYPE_FROM_TYPECODE[type_char]
    if actual_dtype != expected_dtype:
        raise TileTypeError(
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


def require_inline_ptx_constraint_pairs(ptx_code: str, constraint_pairs: tuple) -> tuple:
    if not isinstance(constraint_pairs, tuple):
        raise TileTypeError(
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
                ptx_interpolation_replacements.append('{$r' + str(len(ro_args)) + '}')
                assert isinstance(arg_spec.value, Var)
                ro_args.append(arg_spec.value)
            case InlinePTX.RMWMode.READ_WRITE:
                ptx_interpolation_replacements.append('{$rw' + str(len(rw_args)) + '}')
                assert isinstance(arg_spec.value, Var)
                rw_args.append(arg_spec.value)
            case InlinePTX.RMWMode.WRITE_ONLY:
                ptx_interpolation_replacements.append('{$w' + str(len(wo_args)) + '}')
                assert isinstance(arg_spec.value, datatype.DType)
                wo_args.append(arg_spec.value)

    def rewrite(match: re.Match[str]) -> str:
        index = int(match.group("index"))
        if index >= len(ptx_interpolation_replacements):
            raise TileTypeError(
                f"inline_ptx placeholder %{index} is out of range "
                f"for {len(ptx_interpolation_replacements)} operands"
            )

        return ptx_interpolation_replacements[index]

    mlir_ptx_code = _INLINE_PTX_PLACEHOLDER_RE.sub(rewrite, ptx_code)
    return (
        mlir_ptx_code,
        tuple(ro_args),
        tuple(rw_args),
        tuple(wo_args),
    )


@impl(stub.inline_ptx)
def inline_ptx_impl(ptx_code: Var, constraint_pairs: tuple) -> tuple:
    ptx_code = require_constant_str(ptx_code)
    mlir_ptx_code, ro_args, rw_args, wo_args = require_inline_ptx_constraint_pairs(
        ptx_code, constraint_pairs)
    result_types = tuple(TileTy(dtype) for dtype in wo_args)
    results = add_operation(
        InlinePTX,
        result_types,
        ptx_code=mlir_ptx_code,
        read_only_operands=ro_args,
        write_only_operands=wo_args,
        read_write_operands=rw_args,
    )
    return build_tuple(results)


@dataclass(eq=False)
class RawNVVMIntrinsic(Operation, opcode="nvvm.call_intrinsic",
                       memory_effect=MemoryEffect.STORE):
    intrinsic: str = attribute()
    operands_: tuple[Var, ...] = operand()


def require_scalar_tile_type(value: Var, valid_dtypes: tuple[datatype.DType, ...] = ()) -> TileTy:
    value_ty = require_tile_type(value)
    if value_ty.ndim != 0:
        raise TileTypeError(f"Expected scalar value, got {value_ty}")
    if valid_dtypes and value_ty.dtype not in valid_dtypes:
        raise TileTypeError(f"Expected type to be one of {valid_dtypes}, but got {value_ty.dtype}")
    return value_ty


def _require_nvvm_intrinsic_name(intrinsic: str) -> str:
    if not intrinsic.startswith("llvm."):
        raise TileTypeError(f"Expected intrinsic name to start with 'llvm.', but got {intrinsic!r}")
    return intrinsic


def shfl_sync_impl(mode: str, mask: Var, value: Var, lane_mask: Var, width: Var) -> Var:
    """
    Implements the instructions as the psuedocode in the NVVM IR spec.
    https://docs.nvidia.com/cuda/archive/12.3.1/nvvm-ir-spec/index.html#data-movement

    See also Clang's lowering in __clang_cuda_intrinsics.h.
    """
    value_ty = require_scalar_tile_type(
        value,
        (datatype.int32, datatype.float32),
    )
    require_scalar_tile_type(mask, (datatype.int32,))
    require_scalar_tile_type(lane_mask, (datatype.int32,))
    width = require_constant_int(width)
    if width not in (1, 2, 4, 8, 16, 32):
        raise TileTypeError(f"Expected shuffle width to be a power of two in [1, 32], got {width}")

    WARP_SIZE = 32
    clamp = 0 if mode == 'up' else 0x1F
    mask_and_clamp = strictly_typed_const(
        ((WARP_SIZE - width) << 8) | clamp,
        I32_TY,
    )

    suffix = "i32" if datatype.is_integral(value_ty.dtype) else "f32"
    intrinsic = f"llvm.nvvm.shfl.sync.{mode}.{suffix}"
    return add_operation(
        RawNVVMIntrinsic,
        value_ty,
        intrinsic=intrinsic,
        operands_=(mask, value, lane_mask, mask_and_clamp),
    )


@impl(stub.shfl_sync)
def shfl_sync_idx_impl(mask: Var, value: Var, src_lane: Var, width: Var) -> Var:
    return shfl_sync_impl("idx", mask, value, src_lane, width)


@impl(stub.shfl_up_sync)
def shfl_sync_up_impl(mask: Var, value: Var, delta: Var, width: Var) -> Var:
    return shfl_sync_impl("up", mask, value, delta, width)


@impl(stub.shfl_down_sync)
def shfl_sync_down_impl(mask: Var, value: Var, delta: Var, width: Var) -> Var:
    return shfl_sync_impl("down", mask, value, delta, width)


@impl(stub.shfl_xor_sync)
def shfl_sync_xor_impl(mask: Var, value: Var, lane_mask: Var, width: Var) -> Var:
    return shfl_sync_impl("bfly", mask, value, lane_mask, width)


@impl(getattr, overload=(TensorMapTy, "as_opaque_ptr"))
def getattr_tensor_map_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(stub.TensorMap, name)
    return bind_method(object, unbound_func)


@dataclass(eq=False)
class CreateTensorMap(Operation, opcode="create_tensor_map"):
    base_ptr: Var = operand()
    array_shape: tuple[Var, ...] = operand()
    array_strides: tuple[Var, ...] = operand()


@impl(stub.tensor_map_tiled)
def tensor_map_tiled_impl(array: Var, tile_shape: Var, swizzle: Var) -> Var:
    array_ty = require_array_type(array)
    array_val = array.get_aggregate()
    assert isinstance(array_val, ArrayValue)

    tile_shape = require_constant_int_tuple(tile_shape, allow_single_int=True)
    swizzle = require_constant_enum(swizzle, TensorMapSwizzle)
    data_type = dtype_to_tensor_map_type(array_ty.dtype)
    map_ty = TensorMapTy(data_type=data_type,
                         tile_shape=tile_shape,
                         swizzle=swizzle)
    return add_operation(CreateTensorMap, map_ty,
                         base_ptr=array_val.base_ptr,
                         array_shape=array_val.shape,
                         array_strides=array_val.strides)


@dataclass
class TensorMapAsOpaquePtr(Operation, opcode="tensor_map_as_opaque_ptr"):
    tensor_map: Var = operand()


def require_tensor_map_ty(var: Var) -> TensorMapTy:
    ty = var.get_type()
    if not isinstance(ty, TensorMapTy):
        raise TileTypeError(f"Expected a tensor map, got {ty}")
    return ty


@impl(stub.TensorMap.as_opaque_ptr)
def tensor_map_as_opaque_ptr_impl(self: Var):
    require_tensor_map_ty(self)
    result_ty = TileTy(opaque_pointer_dtype())
    return add_operation(TensorMapAsOpaquePtr, result_ty, tensor_map=self)


def require_constant_result_dtype(dtype: Var) -> Type:
    if not dtype.is_constant():
        raise TileTypeError(f"Expected a dtype constructor but got {dtype}")

    const_dtype = dtype.get_constant()
    if isinstance(const_dtype, datatype.OpaquePointerSpec):
        if const_dtype == datatype.any_opaque_ptr:
            raise TileTypeError("Result type cannot have no memory space")
        memory_space = datatype.MemorySpace(const_dtype.value)
        return TileTy(opaque_pointer_dtype(memory_space=memory_space))
    elif is_vector_ty(const_dtype):
        return const_dtype
    elif isinstance(const_dtype, datatype.DType):
        return TileTy(const_dtype)
    else:
        raise TileTypeError(f"Expected a type spec but got {dtype}")


def require_constant_result_dtypes(result_dtypes: Var) -> tuple[Type, ...]:
    require_tuple_type(result_dtypes)
    result_dtypes = result_dtypes.get_aggregate().items
    return tuple(require_constant_result_dtype(dtype) for dtype in result_dtypes)


@impl(stub.clusterlaunchcontrol_try_cancel)
def clusterlaunchcontrol_try_cancel_impl(addr: Var, mbar: Var, multicast: Var) -> None:
    addr_info = PointerInfo(require_scalar_pointer_type(addr).dtype)
    mbar_info = PointerInfo(require_scalar_pointer_type(mbar).dtype)
    multicast = require_constant_bool(multicast)

    if (
        addr_info.opaque
        or addr_info.pointee_dtype is not datatype.clusterlaunchcontrol_token
        or addr_info.memory_space is not MemorySpace.SHARED
    ):
        raise TileTypeError(
            "Expected a pointer to a cluster launch control "
            f"token in shared memory, got {addr.get_type()}"
        )

    if (
        mbar_info.opaque
        or mbar_info.pointee_dtype is not datatype.mbarrier
        or mbar_info.memory_space is not MemorySpace.SHARED
    ):
        raise TileTypeError(
            f"Expected a pointer to an mbarrier in shared memory, got {mbar.get_type()}"
        )

    intrinsic = "llvm.nvvm.clusterlaunchcontrol.try_cancel.async"
    if multicast:
        intrinsic += ".multicast"
    intrinsic += ".shared"

    add_operation(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(addr, mbar),
    )


@impl(stub.clusterlaunchcontrol_is_canceled)
def clusterlaunchcontrol_is_canceled_impl(token: Var) -> Var:
    require_scalar_tile_type(token, (datatype.clusterlaunchcontrol_token,))
    return add_operation(
        RawNVVMIntrinsic,
        TileTy(datatype.bool_),
        intrinsic="llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled",
        operands_=(token,),
    )


@impl(stub.clusterlaunchcontrol_get_first_block_idx)
def clusterlaunchcontrol_get_first_block_idx_impl(token: Var, axis: Var) -> Var:
    require_scalar_tile_type(token, (datatype.clusterlaunchcontrol_token,))
    if not axis.is_constant():
        raise TileTypeError(
            f"Expected axis to be constant int or None, but got {axis=}"
        )
    axis = axis.get_constant()
    if type(axis) not in (int, None):
        raise TileTypeError(
            f"Expected axis to be constant int or None, but got {axis=}"
        )
    cta_ids = tuple(
        add_operation(
            RawNVVMIntrinsic,
            TileTy(datatype.int32),
            intrinsic=f"llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.{dim}",
            operands_=(token,),
        )
        for dim in ("x", "y", "z")
    )
    return build_tuple(cta_ids) if axis is None else cta_ids[axis]


@dataclass(eq=False)
class ForeignFunction(Operation, opcode="foreign_function", memory_effect=MemoryEffect.STORE):
    function_name: str = attribute()
    operands_: tuple[Var, ...] = operand()


@impl(stub.foreign_function._call_foreign_function)
def _call_foreign_function_impl(func: Var, return_type: Var, parameters: Var) -> Operation:
    function_name = require_constant_str(func)
    if return_type.is_constant() and return_type.get_constant() is None:
        result_type = tuple()
    else:
        result_type = require_constant_result_dtype(return_type)
    require_tuple_type(parameters)
    parameters = parameters.get_aggregate().items
    result = add_operation(
        ForeignFunction,
        result_type,
        function_name=function_name,
        operands_=parameters,
    )
    return result if result_type else None


def require_mbarrier_ptr(
    mbar: Var,
    expected_memory_spaces: tuple[MemorySpace, ...] = (
        MemorySpace.SHARED,
        MemorySpace.SHARED_CLUSTER,
    ),
) -> PointerInfo:
    info = PointerInfo(require_scalar_pointer_type(mbar).dtype)
    if info.opaque or info.pointee_dtype is not datatype.mbarrier:
        raise TileTypeError(f"Expected a pointer to an mbarrier, got {mbar}")
    if info.memory_space not in expected_memory_spaces:
        expected = ', '.join(x.value for x in expected_memory_spaces)
        raise TileTypeError(
            f"Expected mbarrier to be allocated in one of {expected} "
            f"but got {info.memory_space}"
        )
    return info


@impl(stub.mbarrier_init)
def mbarrier_init_impl(mbar: Var, participants: Var) -> Var:
    require_mbarrier_ptr(mbar)
    participants = astype(participants, datatype.int32)
    add_operation(
        RawNVVMIntrinsic,
        tuple(),
        intrinsic="llvm.nvvm.mbarrier.init.shared",
        operands_=(mbar, participants),
    )


@impl(stub.mbarrier_invalidate)
def mbarrier_invalidate_impl(mbar: Var) -> Var:
    require_mbarrier_ptr(mbar)
    add_operation(
        RawNVVMIntrinsic,
        tuple(),
        intrinsic="llvm.nvvm.mbarrier.inval.shared",
        operands_=(mbar,),
    )


def _mbar_space_scope_suffix(scope: MbarrierScope, space: MemorySpace) -> str:
    match space:
        case MemorySpace.SHARED:
            space_str = 'cta'
        case MemorySpace.SHARED_CLUSTER:
            space_str = 'cluster'
        case _:
            raise TileCompilerError(f"Unexpected {space=}")
    return (
        ".scope."
        + scope.value
        + ".space."
        + space_str
    )


def require_mbarrier_ordering(
    ordering_var: Var,
    valid_orderings: tuple[MemoryOrder, ...],
) -> MemoryOrder:
    ordering = require_constant_enum(ordering_var, MemoryOrder)
    if ordering not in valid_orderings:
        formatted = ", ".join(str(o) for o in valid_orderings)
        raise TileTypeError(
            f"Invalid mbarrier memory order {ordering}, expected one of {formatted}"
        )
    return ordering


ARRIVE_ORDERINGS = (MemoryOrder.RELEASE, MemoryOrder.RELAXED)
WAIT_ORDERINGS = (MemoryOrder.ACQUIRE, MemoryOrder.RELAXED)


@impl(stub.mbarrier_arrive)
def mbarrier_arrive_impl(
    mbar: Var,
    count: Var,
    drop: Var,
    scope: Var,
    ordering: Var,
) -> Var | None:
    count = astype(count, datatype.int32)
    drop = require_constant_bool(drop)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, ARRIVE_ORDERINGS)
    space = require_mbarrier_ptr(mbar).memory_space
    intrinsic = "llvm.nvvm.mbarrier.arrive"
    if drop:
        intrinsic += '.drop'
    if ordering is MemoryOrder.RELAXED:
        intrinsic += '.relaxed'
    intrinsic += _mbar_space_scope_suffix(scope, space)

    return_type = (TileTy(datatype.uint64),) if space is MemorySpace.SHARED else ()
    results = add_operation(
        RawNVVMIntrinsic,
        return_type,
        intrinsic=intrinsic,
        operands_=(mbar, count),
    )
    return results[0] if return_type else None


@impl(stub.mbarrier_arrive_expect_tx)
def mbarrier_arrive_expect_tx_impl(
    mbar: Var,
    bytes: Var,
    drop: Var,
    scope: Var,
    ordering: Var,
) -> Var | None:
    bytes = astype(bytes, datatype.int32)
    drop = require_constant_bool(drop)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, ARRIVE_ORDERINGS)
    space = require_mbarrier_ptr(mbar).memory_space
    intrinsic = "llvm.nvvm.mbarrier.arrive"
    if drop:
        intrinsic += '.drop'
    intrinsic += '.expect.tx'
    if ordering is MemoryOrder.RELAXED:
        intrinsic += '.relaxed'
    intrinsic += _mbar_space_scope_suffix(scope, space)

    return_type = (TileTy(datatype.uint64),) if space is MemorySpace.SHARED else ()
    results = add_operation(
        RawNVVMIntrinsic,
        return_type,
        intrinsic=intrinsic,
        operands_=(mbar, bytes),
    )
    return results[0] if return_type else None


@impl(stub.mbarrier_expect_tx)
def mbarrier_expect_tx_impl(mbar: Var, bytes: Var, scope: Var) -> Var:
    space = require_mbarrier_ptr(mbar).memory_space
    bytes = astype(bytes, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    intrinsic = "llvm.nvvm.mbarrier.expect.tx"
    intrinsic += _mbar_space_scope_suffix(scope, space)
    add_operation(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(mbar, bytes),
    )


@impl(stub.mbarrier_complete_tx)
def mbarrier_complete_tx_impl(mbar: Var, bytes: Var, scope: Var) -> Var:
    space = require_mbarrier_ptr(mbar).memory_space
    bytes = astype(bytes, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    intrinsic = "llvm.nvvm.mbarrier.complete.tx"
    intrinsic += _mbar_space_scope_suffix(scope, space)
    add_operation(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(mbar, bytes),
    )


@impl(stub.mbarrier_test_wait)
def mbarrier_test_wait_impl(
    mbar: Var, state: Var, scope: Var, ordering: Var
) -> Var:
    scope = require_constant_enum(scope, MbarrierScope)
    state = astype(state, datatype.int64)
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    ordering = require_mbarrier_ordering(ordering, WAIT_ORDERINGS)
    intrinsic = "llvm.nvvm.mbarrier.test.wait"
    if ordering is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    results = add_operation(
        RawNVVMIntrinsic,
        (TileTy(datatype.bool_),),
        intrinsic=intrinsic,
        operands_=(mbar, state),
    )
    return results[0]


@impl(stub.mbarrier_test_wait_parity)
def mbarrier_test_wait_parity_impl(
    mbar: Var, parity: Var, scope: Var, ordering: Var
) -> Var:
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    parity = astype(parity, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, WAIT_ORDERINGS)
    intrinsic = "llvm.nvvm.mbarrier.test.wait.parity"
    if ordering is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    results = add_operation(
        RawNVVMIntrinsic,
        (TileTy(datatype.bool_),),
        intrinsic=intrinsic,
        operands_=(mbar, parity),
    )
    return results[0]


def _is_none(var: Var):
    return var.is_constant() and var.get_constant() is None


@impl(stub.mbarrier_try_wait)
def mbarrier_try_wait_impl(
    mbar: Var,
    state: Var,
    time_hint: Var,
    scope: Var,
    ordering: Var,
) -> Var:
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    state = astype(state, datatype.int64)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, WAIT_ORDERINGS)
    intrinsic = "llvm.nvvm.mbarrier.try.wait"
    args = (mbar, state)
    if not _is_none(time_hint):
        intrinsic += ".tl"
        time_hint = astype(time_hint, datatype.int32)
        args = (*args, time_hint)
    if ordering is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    results = add_operation(
        RawNVVMIntrinsic,
        (TileTy(datatype.bool_),),
        intrinsic=intrinsic,
        operands_=args,
    )
    return results[0]


@impl(stub.mbarrier_try_wait_parity)
def mbarrier_try_wait_parity_impl(
    mbar: Var,
    parity: Var,
    time_hint: Var,
    scope: Var,
    ordering: Var,
) -> Var:
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    parity = astype(parity, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, WAIT_ORDERINGS)
    intrinsic = "llvm.nvvm.mbarrier.try.wait.parity"
    args = (mbar, parity)
    if not _is_none(time_hint):
        time_hint = astype(time_hint, datatype.int32)
        args = (*args, time_hint)
        intrinsic += ".tl"
    if ordering is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    results = add_operation(
        RawNVVMIntrinsic,
        (TileTy(datatype.bool_),),
        intrinsic=intrinsic,
        operands_=args,
    )
    return results[0]


@impl(stub.map_shared_to_cluster)
def map_shared_to_cluster_impl(ptr: Var, rank: Var):
    ptr_ty = require_scalar_pointer_type(ptr).dtype
    rank = astype(rank, datatype.int32)
    info = PointerInfo(ptr_ty)
    if info.memory_space != MemorySpace.SHARED:
        raise TileTypeError(
            f"Expected pointer in shared memory, but got memory_space={info.memory_space}"
        )
    if info.opaque:
        result_scalar_ty = opaque_pointer_dtype(MemorySpace.SHARED_CLUSTER)
    else:
        result_scalar_ty = pointer_dtype(info.pointee_dtype, MemorySpace.SHARED_CLUSTER)
    result_ty = TileTy(result_scalar_ty)
    results = add_operation(
        RawNVVMIntrinsic,
        (result_ty,),
        intrinsic="llvm.nvvm.mapa.shared.cluster",
        operands_=(ptr, rank),
    )
    return results[0]


__all__ = (
    "AddrSpaceCast",
    "ArrayGetItem",
    "ArraySetItem",
    "AtomicCAS",
    "AtomicExchange",
    "AtomicRMW",
    "AtomicRMWKind",
    "Assign",
    "AssumeBounded",
    "AssumeDivBy",
    "Branch",
    "CondBranch",
    "MakeTensorView",
    "MakeDummy",
    "Return",
    "RawBinaryArithmeticOperation",
    "RawBinaryBitwiseOperation",
    "RawBitwiseShiftOperation",
    "RawComparisonOperation",
    "TileAsType",
    "TypedConst",
    "branch",
    "cond_branch",
    "return_",
    "IfElse",
    "EndBranch",
    "Loop",
    "Continue",
    "Break",
    "TilePrintf",
    "PointerOffset",
    "LoadPointer",
    "ReinterpretPointer",
    "ReinterpretPointerAsArray",
    "StorePointer",
    "RawWhereOperation",
    "Unary",
)
