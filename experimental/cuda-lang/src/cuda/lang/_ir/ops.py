# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import math
import operator
from dataclasses import dataclass
from cuda.tile._memory_model import MemoryOrder, MemoryScope
from cuda.tile._ir.op_impl import (
    require_tuple_type,
    require_constant_str,
    require_dtype_spec,
    require_constant_int_tuple,
    require_constant_int,
    require_constant_enum,
    require_array_type,
    require_constant_bool,
    require_constant_axis_order,
    ImplRegistry,
)
from cuda.tile._ir.type import TensorLikeTy
from cuda.tile._ir.core_ops import (
    TypedConst, core_impl_registry,
)
from cuda.tile._ir.arithmetic_ops import (
    binary_arithmetic_tensorlike,
    binary_arithmetic_tensorlike_raw,
    RawBinaryArithmeticOperation,
    RawComparisonOperation,
    RawBinaryBitwiseOperation,
    RawBitwiseShiftOperation,
    TileAsType,
    TileBroadcast,
    TileReshape,
    RawWhereOperation,
    Unary,
    arithmetic_impl_registry,
)
from cuda.tile._ir.static_eval_ops import static_eval_impl_registry
from cuda.tile._ir.ops import (
    AssumeBounded,
    AssumeDivBy,
    MakeTensorView,
    PointerOffset,
    TilePrintf,
    array_impl_registry,
)
from cuda.tile._ir.arithmetic_ops import astype
from cuda.tile._ir.core_ops import (
    Assign,
    bind_method,
    build_tuple,
    loosely_typed_const,
    strictly_typed_const,
)
from cuda.tile._ir.cast_ops import (
    AddrSpaceCast, ReinterpretPointer,
)
from cuda.tile._ir.control_flow_ops import (
    control_flow_impl_registry,
    IfElse,
    EndBranch,
    Loop,
    Continue,
    Break,
    Return,
    return_,
    MakeDummy,
)
from cuda.tile._ir.ir import MemoryEffect, make_aggregate, add_operation_variadic
from cuda.lang._exception import TypeCheckingError
import cuda.lang._datatype as datatype
from cuda.tile._datatype import (
    pointer_dtype,
    PointerInfo,
    opaque_pointer_dtype,
    int32,
    bool_,
)
from .atomics_support import (
    ATOMIC_CAS_DTYPES,
    ATOMIC_XCHG_DTYPES,
    AtomicRMWKind,
    atomic_rmw_op_name,
    require_atomic_dtype,
    require_atomic_memory_order_and_scope,
    require_atomic_rmw_value,
)
from .op_defs import (  # noqa: F401
    RawNVVMIntrinsic,
    RawMLIROperation,
    InlinePTX,
    ForeignFunction,
    TensorMapAsOpaquePtr,
    VectorGetItem,
    VectorReduce,
    StorePointer,
    LoadPointer,
    ReinterpretPointerAsArray,
    BitCast,
)
from .op_impl.core_api_impl import core_api_impl_registry
from .type_checking_helpers import (
    require_optional_alignment,
    require_scalar_type,
    require_pointer_type,
    require_signed_int_scalar_or_tuple,
    require_clusterlaunchcontrol_token_type,
    is_none,
    require_tensor_map_ty,
    validate_tensor_map_load_mode,
)

from .type import (
    LocalArrayContextManagerTy,
    ContextManagerState,
    TensorMapTy,
    dtype_to_tensor_map_type,
    ArrayValue,
    MemorySpace,
    Type,
    ArrayTy,
    ScalarTy,
    PointerTy,
    VectorTy,
    DTypeSpec,
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
from .._stub.cluster_launch_control import clusterlaunchcontrol_try_cancel, \
    clusterlaunchcontrol_is_canceled, clusterlaunchcontrol_get_first_block_index
from .._enums import SwizzleMode, TMALoadMode
from .._stub import (
    foreign_function,
    core_api,
    tensor_map,
)
from cuda.tile._ir import hir_stubs

from .op_impl.tcgen05_impl import tcgen05_impl_registry
from .op_impl.math_impl import math_impl_registry
from .op_impl.vector_impl import vector_impl_registry
from .op_impl.pointer_impl import (
    pointer_with_offset,
    array_base_pointer_type,
    contiguous_strides_from_shape,
    pointer_impl_registry,
)
from .op_impl.copy_async_impl import copy_async_impl_registry
from .op_impl.barrier_impl import barrier_impl_registry
from .op_impl.mbarrier_impl import mbarrier_impl_registry
from .op_impl.inline_ptx_impl import inline_ptx_impl_registry
from .op_impl.matrix_impl import matrix_impl_registry
from .op_impl.printing_impl import printing_impl_registry

cuda_lang_impl_registry = ImplRegistry()
cuda_lang_impl_registry.update(core_impl_registry())
cuda_lang_impl_registry.update(static_eval_impl_registry())
cuda_lang_impl_registry.update(arithmetic_impl_registry())
cuda_lang_impl_registry.update(control_flow_impl_registry())
cuda_lang_impl_registry.update(array_impl_registry)

cuda_lang_impl_registry.update(tcgen05_impl_registry())
cuda_lang_impl_registry.update(inline_ptx_impl_registry())
cuda_lang_impl_registry.update(core_api_impl_registry())
cuda_lang_impl_registry.update(math_impl_registry())
cuda_lang_impl_registry.update(vector_impl_registry())
cuda_lang_impl_registry.update(pointer_impl_registry())
cuda_lang_impl_registry.update(copy_async_impl_registry())
cuda_lang_impl_registry.update(barrier_impl_registry())
cuda_lang_impl_registry.update(mbarrier_impl_registry())
cuda_lang_impl_registry.update(matrix_impl_registry())
cuda_lang_impl_registry.update(printing_impl_registry())

impl = cuda_lang_impl_registry.impl


@impl(core_api.dtype_of)
def dtype_of_impl(value: Var):
    ty = value.get_type()
    if isinstance(ty, ScalarTy):
        dtype = ty.dtype
    elif isinstance(ty, PointerTy):
        dtype = ty.pointer_dtype
    else:
        raise TypeCheckingError(
            f"dtype_of() expects a scalar or a pointer as the argument, got {ty}"
        )
    return loosely_typed_const(dtype)


def _atomic_rmw_dispatch(
    kind: AtomicRMWKind,
    ptr: Var,
    val: Var,
    memory_order: Var,
    memory_scope: Var,
) -> Var:
    op_name = atomic_rmw_op_name(kind)
    ptr_ty = require_pointer_type(ptr)
    val, result_ty = require_atomic_rmw_value(kind, ptr_ty, val)
    memory_order, memory_scope = require_atomic_memory_order_and_scope(
        op_name, memory_order, memory_scope
    )
    return add_operation(
        AtomicRMW,
        result_ty,
        kind=kind,
        pointer=ptr,
        value=val,
        memory_order=memory_order,
        memory_scope=memory_scope,
    )


@dataclass(eq=False)
class AtomicRMW(Operation, opcode="atomic_rmw", memory_effect=MemoryEffect.STORE):
    kind: AtomicRMWKind = attribute()
    pointer: Var = operand()
    value: Var = operand()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()


@dataclass(eq=False)
class AtomicExchange(Operation, opcode="atomic_xchg", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()


@dataclass(eq=False)
class AtomicCAS(Operation, opcode="atomic_cas", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    compare: Var = operand()
    value: Var = operand()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()


@impl(core_api.atomic_add, fixed_args=[AtomicRMWKind.ADD])
@impl(core_api.atomic_sub, fixed_args=[AtomicRMWKind.SUB])
@impl(core_api.atomic_and, fixed_args=[AtomicRMWKind.AND])
@impl(core_api.atomic_or, fixed_args=[AtomicRMWKind.OR])
@impl(core_api.atomic_xor, fixed_args=[AtomicRMWKind.XOR])
@impl(core_api.atomic_min, fixed_args=[AtomicRMWKind.MIN])
@impl(core_api.atomic_max, fixed_args=[AtomicRMWKind.MAX])
@impl(core_api.atomic_inc, fixed_args=[AtomicRMWKind.INC])
@impl(core_api.atomic_dec, fixed_args=[AtomicRMWKind.DEC])
def atomic_rmw_dispatch_impl(
    kind: AtomicRMWKind,
    ptr: Var,
    val: Var,
    memory_order: Var,
    memory_scope: Var,
) -> Var:
    return _atomic_rmw_dispatch(kind, ptr, val, memory_order, memory_scope)


@impl(core_api.atomic_xchg)
def atomic_xchg_impl(
    ptr: Var, val: Var, memory_order: Var, memory_scope: Var
) -> Var:
    ptr_ty = require_pointer_type(ptr)
    dtype = ptr_ty.pointee_dtype
    require_atomic_dtype("atomic_xchg", dtype, ATOMIC_XCHG_DTYPES)
    require_scalar_type(val)
    val = astype(val, dtype)
    result_ty = ScalarTy(dtype)
    memory_order, memory_scope = require_atomic_memory_order_and_scope(
        "atomic_xchg", memory_order, memory_scope
    )
    return add_operation(
        AtomicExchange,
        result_ty,
        pointer=ptr,
        value=val,
        memory_order=memory_order,
        memory_scope=memory_scope,
    )


@impl(core_api.atomic_cas)
def atomic_cas_impl(
    ptr: Var, old: Var, val: Var, memory_order: Var, memory_scope: Var
) -> Var:
    ptr_ty = require_pointer_type(ptr)
    dtype = ptr_ty.pointee_dtype
    require_atomic_dtype("atomic_cas", dtype, ATOMIC_CAS_DTYPES)
    require_scalar_type(val)
    val = astype(val, dtype)
    compare_ty = require_scalar_type(old)
    if dtype != compare_ty.dtype:
        raise TypeCheckingError(
            f"Expected atomic compare value of type {dtype}, got {compare_ty.dtype}"
        )
    result_ty = ScalarTy(dtype)
    memory_order, memory_scope = require_atomic_memory_order_and_scope(
        "atomic_cas", memory_order, memory_scope
    )
    return add_operation(
        AtomicCAS,
        result_ty,
        pointer=ptr,
        compare=old,
        value=val,
        memory_order=memory_order,
        memory_scope=memory_scope,
    )


@impl(operator.add, overload=(TensorLikeTy, TensorLikeTy))
async def add_impl(x: Var, y: Var) -> Var:
    xty, yty = x.get_type(), y.get_type()
    if isinstance(yty, PointerTy):
        xty, yty = yty, xty
    if isinstance(xty, PointerTy):
        offset_dtype = require_scalar_type(y).dtype
        if not datatype.is_integral(offset_dtype):
            raise TypeCheckingError(f"Expected integer pointer offset, got {offset_dtype}")
        return pointer_with_offset(x, y)
    return binary_arithmetic_tensorlike("add", x, y)


@impl(operator.sub, overload=(TensorLikeTy, TensorLikeTy))
async def sub_impl(x: Var, y: Var) -> Var:
    xty, yty = x.get_type(), y.get_type()
    if isinstance(xty, PointerTy):
        offset_dtype = require_scalar_type(y).dtype
        if not datatype.is_integral(offset_dtype):
            raise TypeCheckingError(f"Expected integer pointer offset, got {offset_dtype}")
        y = astype(y, datatype.int64)
        c0 = loosely_typed_const(0)
        offset = binary_arithmetic_tensorlike('sub', c0, y)
        return pointer_with_offset(x, offset)
    if isinstance(yty, PointerTy):
        raise TypeCheckingError('It is invalid to subtract a pointer from an integer')
    return binary_arithmetic_tensorlike("sub", x, y)


@impl(getattr, overload=(VectorTy, "element_count"))
def vector_element_count_impl(object: Var[VectorTy], name: Var):
    return loosely_typed_const(object.get_type().length)


@impl(getattr, overload=(VectorTy, "dtype"))
def vector_dtype_impl(object: Var[VectorTy], name: Var):
    return loosely_typed_const(object.get_type().element_dtype)


@dataclass(eq=False)
class Branch(Operation, opcode="br", terminator=True):
    target: Block = attribute()
    args: tuple[Var, ...] = operand()

    def _to_string_rhs(self) -> str:
        return f"{self.op} ^{self.target._name}({', '.join(format_var(arg) for arg in self.args)})"


def branch(target: Block, args: tuple[Var, ...]) -> None:
    add_operation_variadic(Branch, (), target=target, args=args)


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
) -> None:
    add_operation_variadic(
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


def _dtype_byte_width(dtype: datatype.DType) -> int:
    assert dtype.bitwidth % 8 == 0
    return dtype.bitwidth // 8


@impl(core_api.local_array)
def local_array_impl(shape: Var, dtype: Var, alignment: Var) -> Var:
    shape = require_constant_int_tuple(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    alignment = require_optional_alignment(alignment)
    dtype_byte_width = _dtype_byte_width(dtype)
    if alignment is not None and alignment < dtype_byte_width:
        raise TypeCheckingError(f"Requested {alignment=} is less than {dtype_byte_width}")

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
        raise TypeCheckingError(
            f"Requested alignment {mgr_ty.alignment}"
            f" is less than item size {dtype_byte_width}"
        )
    strides = contiguous_strides_from_shape(mgr_ty.shape)
    index_dtype = datatype.int32
    array_type = ArrayTy(
        mgr_ty.dtype,
        shape=mgr_ty.shape,
        strides=strides,
        typing_hooks=manager.ctx.typing_hooks,
        index_dtype=index_dtype,
        memory_space=MemorySpace.GENERIC,
    )
    size_ty = ScalarTy(index_dtype)
    shape_vars = tuple(strictly_typed_const(extent, size_ty) for extent in mgr_ty.shape)
    stride_vars = tuple(strictly_typed_const(extent, size_ty) for extent in strides)

    base_ptr = add_operation(
        AllocLocalMemory,
        array_base_pointer_type(array_type),
        count=math.prod(mgr_ty.shape),
        alignment=mgr_ty.alignment,
    )

    def exit_callback():
        add_operation_variadic(DeallocLocalMemory, (), ptr=base_ptr)

    mgr_ty.state.exit_callback = exit_callback

    array_val = ArrayValue(base_ptr, shape_vars, stride_vars)
    return make_aggregate(array_val, array_type)


@dataclass(eq=False)
class GetDynSharedMemoryBasePtr(Operation, opcode="get_dyn_shared_memory_base_ptr"):
    initial_alignment = 1024


def get_dyn_shared_memory_base_ptr():
    result_ty = PointerTy(pointer_dtype(datatype.uint8, MemorySpace.SHARED))
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


@impl(core_api.shared_array)
def shared_array_impl(shape: Var, dtype: Var, dynamic: Var, alignment: Var) -> Operation:
    dynamic = require_constant_bool(dynamic)

    sizes = require_signed_int_scalar_or_tuple(shape)

    dtype = require_dtype_spec(dtype)
    alignment = require_optional_alignment(alignment)
    dtype_byte_width = _dtype_byte_width(dtype)
    if alignment is not None and alignment < dtype_byte_width:
        raise TypeCheckingError(f"Requested {alignment=} is less than {dtype_byte_width=}")
    index_dtype = datatype.int32
    size_ty = ScalarTy(index_dtype)

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
                raise TypeCheckingError(
                    f"Shared memory size must be {index_dtype},"
                    f" got {size_var.get_type().dtype}"
                )
            size = None
        ty_shape.append(size)
        shape_vars.append(size_var)
        ty_strides.append(total_size)
        stride_vars.append(total_size_var)

        if size is None or total_size is None:
            total_size = None
            total_size_var = binary_arithmetic_tensorlike_raw("mul", total_size_var, size_var)
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
        typing_hooks=shape.ctx.typing_hooks,
        index_dtype=index_dtype,
        memory_space=MemorySpace.SHARED,
    )

    if dynamic:
        base_ptr = add_operation(AllocDynSharedMemory,
                                 array_base_pointer_type(array_type),
                                 shape=sizes,
                                 alignment=alignment)
    else:
        if total_size is None:
            raise TypeCheckingError("Shape must be constant when `dynamic` is False")

        base_ptr = add_operation(AllocStaticSharedMemory,
                                 array_base_pointer_type(array_type),
                                 count=total_size,
                                 alignment=alignment)

    array_value = ArrayValue(base_ptr=base_ptr, shape=tuple(shape_vars), strides=tuple(stride_vars))
    return make_aggregate(array_value, array_type)


@impl(core_api.elect_sync)
def elect_sync_impl(membermask) -> Var:
    mask = require_constant_int(membermask)
    mask = strictly_typed_const(mask & 0xffffffff, ScalarTy(int32))

    _, is_elected = add_operation_variadic(RawNVVMIntrinsic,
                                           (ScalarTy(int32), ScalarTy(bool_)),
                                           intrinsic="llvm.nvvm.elect.sync",
                                           operands_=(mask,))
    return is_elected


def shfl_sync_impl(mode: str, mask: Var, value: Var, operand: Var, width: Var) -> Var:
    """
    Implements the instructions as the psuedocode in the NVVM IR spec.
    https://docs.nvidia.com/cuda/archive/12.3.1/nvvm-ir-spec/index.html#data-movement

    See also Clang's lowering in __clang_cuda_intrinsics.h.
    """
    valid_value_dtypes = (datatype.int32, datatype.uint32, datatype.float32)
    value_ty = require_scalar_type(
        value,
        lambda dtype: dtype in valid_value_dtypes,
        f"Expected shuffle value dtype to be one of {valid_value_dtypes}",
    )
    require_scalar_type(
        mask,
        datatype.is_integral,
        "Expected shuffle mask dtype to be an integer",
    )
    mask = astype(mask, datatype.int32)
    require_scalar_type(
        operand,
        datatype.is_integral,
        "Expected shuffle lane mask dtype to be an integer",
    )
    operand = astype(operand, datatype.int32)
    width = require_constant_int(width)
    if width not in (1, 2, 4, 8, 16, 32):
        raise TypeCheckingError(
            f"Expected shuffle width to be a power of two in [1, 32], got {width}"
        )

    WARP_SIZE = 32
    clamp = 0 if mode == 'up' else 0x1F
    mask_and_clamp = strictly_typed_const(
        ((WARP_SIZE - width) << 8) | clamp,
        ScalarTy(int32),
    )

    suffix = "i32" if datatype.is_integral(value_ty.dtype) else "f32"
    intrinsic = f"llvm.nvvm.shfl.sync.{mode}.{suffix}"
    return add_operation(
        RawNVVMIntrinsic,
        value_ty,
        intrinsic=intrinsic,
        operands_=(mask, value, operand, mask_and_clamp),
    )


@impl(core_api.shfl_sync)
def shfl_sync_idx_impl(value: Var, src_lane: Var, width: Var, mask: Var) -> Var:
    return shfl_sync_impl("idx", mask, value, src_lane, width)


@impl(core_api.shfl_up_sync)
def shfl_sync_up_impl(value: Var, delta: Var, width: Var, mask: Var) -> Var:
    return shfl_sync_impl("up", mask, value, delta, width)


@impl(core_api.shfl_down_sync)
def shfl_sync_down_impl(value: Var, delta: Var, width: Var, mask: Var) -> Var:
    return shfl_sync_impl("down", mask, value, delta, width)


@impl(core_api.shfl_xor_sync)
def shfl_sync_xor_impl(value: Var, lane_mask: Var, width: Var, mask: Var) -> Var:
    return shfl_sync_impl("bfly", mask, value, lane_mask, width)


@impl(getattr, overload=(DTypeSpec, "bitwidth"))
def getattr_dtype_bitwidth(object: Var, name: Var):
    dtype = require_dtype_spec(object)
    return loosely_typed_const(dtype.bitwidth)


@impl(getattr, overload=(TensorMapTy, "as_opaque_ptr"))
@impl(getattr, overload=(TensorMapTy, "get_transaction_bytes"))
def getattr_tensor_map_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(tensor_map.TensorMap, name)
    return bind_method(object, unbound_func)


@dataclass(eq=False)
class CreateTensorMap(Operation, opcode="create_tensor_map"):
    base_ptr: Var = operand()
    array_shape: tuple[Var, ...] = operand()
    array_strides: tuple[Var, ...] = operand()


@impl(tensor_map.tensor_map_tiled)
def tensor_map_tiled_impl(array: Var, tile_shape: Var, order: Var, swizzle: Var) -> Var:
    array_ty = require_array_type(array)
    array_val = array.get_aggregate()
    assert isinstance(array_val, ArrayValue)

    tile_shape = require_constant_int_tuple(tile_shape, allow_single_int=True)
    order = require_constant_axis_order(order, array_ty.ndim)
    swizzle = require_constant_enum(swizzle, SwizzleMode)
    data_type = dtype_to_tensor_map_type(array_ty.dtype)
    map_ty = TensorMapTy(data_type=data_type,
                         element_bitwidth=array_ty.dtype.bitwidth,
                         tile_shape=tile_shape,
                         swizzle=swizzle)
    return add_operation(CreateTensorMap, map_ty,
                         base_ptr=array_val.base_ptr,
                         array_shape=tuple(array_val.shape[i] for i in order),
                         array_strides=tuple(array_val.strides[i] for i in order))


@impl(tensor_map.TensorMap.get_transaction_bytes)
def tensor_map_get_transaction_bytes_impl(self: Var, mode: Var):
    map_ty = require_tensor_map_ty(self)
    mode = require_constant_enum(mode, TMALoadMode)

    if map_ty.element_bitwidth % 8 != 0:
        raise TypeCheckingError(
            "Transaction-byte computation does not support sub-byte tensor maps"
        )

    match mode:
        case TMALoadMode.TILE:
            element_count = math.prod(map_ty.tile_shape)
        case TMALoadMode.TILE_GATHER4:
            validate_tensor_map_load_mode(map_ty, mode)
            element_count = 4 * map_ty.tile_shape[0]
        case _:
            raise TypeCheckingError(
                f"Cannot compute {mode.name} transaction bytes from a tiled tensor map"
            )

    return loosely_typed_const(element_count * map_ty.element_bitwidth // 8)


@impl(tensor_map.TensorMap.as_opaque_ptr)
def tensor_map_as_opaque_ptr_impl(self: Var):
    require_tensor_map_ty(self)
    result_ty = PointerTy(opaque_pointer_dtype())
    return add_operation(TensorMapAsOpaquePtr, result_ty, tensor_map=self)


@impl(clusterlaunchcontrol_try_cancel)
def clusterlaunchcontrol_try_cancel_impl(addr: Var, mbar: Var, multicast: Var) -> None:
    addr_info = PointerInfo(require_pointer_type(addr).pointer_dtype)
    mbar_info = PointerInfo(require_pointer_type(mbar).pointer_dtype)
    multicast = require_constant_bool(multicast)

    if (
        addr_info.opaque
        or addr_info.pointee_dtype is not datatype.clusterlaunchcontrol_token
        or addr_info.memory_space is not MemorySpace.SHARED
    ):
        raise TypeCheckingError(
            "Expected a pointer to a cluster launch control "
            f"token in shared memory, got {addr.get_type()}"
        )

    if (
        mbar_info.opaque
        or mbar_info.pointee_dtype is not datatype.mbarrier
        or mbar_info.memory_space is not MemorySpace.SHARED
    ):
        raise TypeCheckingError(
            f"Expected a pointer to an mbarrier in shared memory, got {mbar.get_type()}"
        )

    intrinsic = "llvm.nvvm.clusterlaunchcontrol.try_cancel.async"
    if multicast:
        intrinsic += ".multicast"
    intrinsic += ".shared"

    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(addr, mbar),
    )


@impl(clusterlaunchcontrol_is_canceled)
def clusterlaunchcontrol_is_canceled_impl(token: Var) -> Var:
    require_clusterlaunchcontrol_token_type(token)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic="llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled",
        operands_=(token,),
    )


@impl(clusterlaunchcontrol_get_first_block_index)
def clusterlaunchcontrol_get_first_block_index_impl(token: Var, axis: Var) -> Var:
    require_clusterlaunchcontrol_token_type(token)
    if not axis.is_constant():
        raise TypeCheckingError(
            f"Expected axis to be constant int or None, but got {axis=}"
        )
    axis = axis.get_constant()
    if type(axis) not in (int, None):
        raise TypeCheckingError(
            f"Expected axis to be constant int or None, but got {axis=}"
        )
    cta_ids = tuple(
        add_operation(
            RawNVVMIntrinsic,
            ScalarTy(datatype.int32),
            intrinsic=f"llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.{dim}",
            operands_=(token,),
        )
        for dim in ("x", "y", "z")
    )
    return build_tuple(cta_ids) if axis is None else cta_ids[axis]


def require_constant_result_dtype(dtype: Var) -> Type:
    if not dtype.is_constant():
        raise TypeCheckingError(f"Expected a dtype constructor but got {dtype}")

    const_dtype = dtype.get_constant()
    if datatype.is_pointer_dtype(const_dtype):
        return PointerTy(const_dtype)
    elif isinstance(const_dtype, datatype.DType):
        return ScalarTy(const_dtype)
    else:
        raise TypeCheckingError(f"Expected a type spec but got {dtype}")


@impl(foreign_function._call_foreign_function)
def _call_foreign_function_impl(func: Var, return_type: Var, parameters: Var):
    function_name = require_constant_str(func)
    require_tuple_type(parameters)
    parameters = parameters.get_aggregate().items
    if is_none(return_type):
        add_operation_variadic(
            ForeignFunction,
            (),
            function_name=function_name,
            operands_=parameters,
        )
        return None
    else:
        result_type = require_constant_result_dtype(return_type)
        return add_operation(
            ForeignFunction,
            result_type,
            function_name=function_name,
            operands_=parameters,
        )


__all__ = (
    "AddrSpaceCast",
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
    "TileReshape",
    "TileBroadcast",
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
    "RawNVVMIntrinsic",
    "RawMLIROperation",
    "ForeignFunction",
    "VectorGetItem",
    "VectorReduce",
)
