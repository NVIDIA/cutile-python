# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._enums import SwizzleMode
from cuda.tile._ir.type import ArrayTy
from typing import Any, NamedTuple, cast

from cuda.tile._ir.cast_ops import implicit_cast
from cuda.tile._ir.arithmetic_ops import (
    binary_arithmetic_tensorlike,
    binary_bitwise_tensorlike,
    bitwise_shift_tensorlike,
)
from cuda.tile._ir.type import DataclassValue
from cuda.tile._ir.core_ops import strictly_typed_const
import cuda.lang._datatype as datatype
from cuda.lang._enums import (
    Tcgen05MMABlockScaleKind,
    Tcgen05MMAScaleVectorSize,
    Tcgen05MMACollectorBBuffer,
    Tcgen05MMAKind,
    Tcgen05LoadStoreShape,
    Tcgen05MMACollectorOp,
    CTAGroup,
    Tcgen05CopyMulticast,
    Tcgen05CopyShape,
    Tcgen05CopySourceFormat,
)
from cuda.lang._exception import TypeCheckingError, InvalidValueError
from cuda.lang._ir.type import PointerTy
from cuda.lang._stub import tcgen05 as tcgen05_stub
from cuda.lang._ir.ir import Var, add_operation
from cuda.lang._ir.ops import (
    MemorySpace,
    ScalarTy,
    VectorTy,
    add_operation_variadic,
    astype,
    require_scalar_type,
)
from cuda.lang._ir.op_defs import BitCast, RawNVVMIntrinsic
from .raw_mlir_operation_utils import RawMLIROperationBuilder
from cuda.lang._ir.enum_to_mlir import cl_enum_to_mlir_attribute
from cuda.lang._ir.type_checking_helpers import (
    is_none,
    make_type_checking_error,
    require_integral_scalar_type,
    require_mbarrier_ptr,
    require_pointer_in_memory_space,
    require_vector_type,
)
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_bool,
    require_constant_enum,
    require_constant_int,
)
import cuda.lang._mlir as mlir
from .core_api_impl import bitcast


_registry = ImplRegistry()
impl = _registry.impl


def tcgen05_impl_registry() -> ImplRegistry:
    return _registry


TCGEN05_VALID_COUNTS_BY_SHAPE = {
    Tcgen05LoadStoreShape.SHAPE_16X64B: (1, 2, 4, 8, 16, 32, 64, 128),
    Tcgen05LoadStoreShape.SHAPE_16X128B: (1, 2, 4, 8, 16, 32, 64),
    Tcgen05LoadStoreShape.SHAPE_16X256B: (1, 2, 4, 8, 16, 32),
    Tcgen05LoadStoreShape.SHAPE_32X32B: (1, 2, 4, 8, 16, 32, 64, 128),
    Tcgen05LoadStoreShape.SHAPE_16X32BX2: (1, 2, 4, 8, 16, 32, 64, 128),
}

TCGEN05_REGISTERS_PER_COUNT = {
    Tcgen05LoadStoreShape.SHAPE_16X64B: 1,
    Tcgen05LoadStoreShape.SHAPE_16X128B: 2,
    Tcgen05LoadStoreShape.SHAPE_16X256B: 4,
    Tcgen05LoadStoreShape.SHAPE_32X32B: 1,
    Tcgen05LoadStoreShape.SHAPE_16X32BX2: 1,
}


@impl(tcgen05_stub.tcgen05_allocate)
def tcgen05_allocate_impl(
    address: Var,
    number_of_columns: Var,
    cta_group: Var,
) -> None:
    require_pointer_in_memory_space(
        address, (MemorySpace.SHARED_CLUSTER, MemorySpace.SHARED)
    )
    number_of_columns = implicit_cast(
        number_of_columns, datatype.int32, "cast number of columns to int32"
    )
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.alloc.shared." + cta_group_value.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(address, number_of_columns),
    )


@impl(tcgen05_stub.tcgen05_deallocate)
def tcgen05_deallocate_impl(
    address: Var,
    number_of_columns: Var,
    cta_group: Var,
) -> None:
    require_pointer_in_memory_space(address, (MemorySpace.TENSOR,))
    number_of_columns = implicit_cast(
        number_of_columns, datatype.int32, "cast number of columns to int32"
    )
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.dealloc." + cta_group_value.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(address, number_of_columns),
    )


@impl(tcgen05_stub.tcgen05_tmem_offset)
def tcgen05_tmem_offset_impl(
    pointer: Var,
    lane_offset: Var,
    column_offset: Var,
) -> Var:
    pointer_type = require_pointer_in_memory_space(pointer, (MemorySpace.TENSOR,))
    lane_offset = implicit_cast(lane_offset, datatype.int32, "lane offset")
    column_offset = implicit_cast(column_offset, datatype.int32, "column offset")

    int32_type = ScalarTy(datatype.int32)
    field_mask = strictly_typed_const(0xFFFF, int32_type)
    field_shift = strictly_typed_const(16, int32_type)
    address = add_operation(BitCast, int32_type, x=pointer)

    lane = bitwise_shift_tensorlike("rshift", address, field_shift)
    column = binary_bitwise_tensorlike("and_", address, field_mask)
    lane = binary_arithmetic_tensorlike("add", lane, lane_offset)
    column = binary_arithmetic_tensorlike("add", column, column_offset)

    lane = bitwise_shift_tensorlike("lshift", lane, field_shift)
    address = binary_bitwise_tensorlike("or_", lane, column)
    return add_operation(BitCast, pointer_type, x=address)


@impl(tcgen05_stub.tcgen05_commit)
def tcgen05_commit_impl(
    mbarrier: Var,
    multicast_mask: Var,
    cta_group: Var,
):
    require_mbarrier_ptr(mbarrier)
    operands = [mbarrier]
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.commit"
    if not is_none(multicast_mask):
        intrinsic += ".mc"
        mask = implicit_cast(multicast_mask, datatype.int16, "multicast mask")
        operands.append(mask)
    intrinsic += ".shared." + cta_group_value.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=tuple(operands),
    )


@impl(tcgen05_stub.tcgen05_copy)
def tcgen05_copy_impl(
    address,
    shared_memory_descriptor,
    shape,
    cta_group,
    multicast,
    source_format,
):
    require_pointer_in_memory_space(address, (MemorySpace.TENSOR,))
    require_integral_scalar_type(shared_memory_descriptor, bitwidth=64)
    shared_memory_descriptor = astype(shared_memory_descriptor, datatype.int64)

    group_value = require_constant_enum(cta_group, CTAGroup)
    group_attribute = cl_enum_to_mlir_attribute(group_value)

    shape_value = require_constant_enum(shape, Tcgen05CopyShape)
    shape_attribute = cl_enum_to_mlir_attribute(shape_value)

    builder = (
        RawMLIROperationBuilder(name="nvvm.tcgen05.cp")
        .add_operand(address)
        .add_operand(shared_memory_descriptor)
        .add_attribute("group", group_attribute)
        .add_attribute("shape", shape_attribute)
    )

    if is_none(multicast):
        value = mlir.nvvm.Tcgen05CpMulticast.NONE
        multicast_attribute = mlir.nvvm.Tcgen05CpMulticastAttr(value=value)
    else:
        multicast_value = require_constant_enum(multicast, Tcgen05CopyMulticast)
        multicast_attribute = cl_enum_to_mlir_attribute(multicast_value)

    builder = builder.add_attribute("multicast", multicast_attribute)

    if not is_none(source_format):
        source_format_value = require_constant_enum(
            source_format, Tcgen05CopySourceFormat
        )
        source_format_attribute = cl_enum_to_mlir_attribute(source_format_value)
        builder = builder.add_attribute("srcFormat", source_format_attribute)

    builder.emit()


@impl(tcgen05_stub.tcgen05_store)
def tcgen05_store_impl(
    shape: Var,
    tensor_memory_address: Var,
    value: Var,
    unpack: Var,
    offset: Var,
):
    require_pointer_in_memory_space(tensor_memory_address, (MemorySpace.TENSOR,))
    shape_value = require_constant_enum(shape, Tcgen05LoadStoreShape)
    valid_counts = TCGEN05_VALID_COUNTS_BY_SHAPE[shape_value]
    registers_per_count = TCGEN05_REGISTERS_PER_COUNT[shape_value]
    value_type = value.get_type()
    if is_none(offset):
        offset = None
    else:
        require_constant_int(offset)
        offset = astype(offset, datatype.int64)
    require_constant_bool(unpack)

    def type_error(dtype, count):
        message = (
            "Expected scalar 32-bit integer or vector of 32-bit integers "
            f"but got {count=} and {dtype=}"
        )
        raise TypeCheckingError(message)

    match value_type:
        case ScalarTy() as st:
            count = 1
            if st.dtype != datatype.int32:
                type_error(st.dtype, count)
        case VectorTy() as vt:
            count = vt.length
            if vt.element_dtype != datatype.int32:
                type_error(vt.element_dtype, count)
        case _:
            raise TypeCheckingError("Expected scalar or vector with datatype int32")

    valid_register_counts = tuple(
        valid_count * registers_per_count for valid_count in valid_counts
    )
    if count not in valid_register_counts:
        valid = ", ".join(str(count) for count in valid_register_counts)
        raise InvalidValueError(
            f"Expected register count for {shape_value.name} to be one of "
            f"{valid}, got {count}"
        )

    count_value = count // registers_per_count
    needs_offset = shape_value == Tcgen05LoadStoreShape.SHAPE_16X32BX2
    has_offset = offset is not None
    if needs_offset != has_offset:
        raise TypeCheckingError(
            "offset parameter is only valid with shape "
            "Tcgen05LoadStoreShape.SHAPE_16X32BX2"
        )

    operands = (
        tensor_memory_address,
        *([offset] if offset is not None else []),
        value,
        unpack,
    )
    intrinsic = f"llvm.nvvm.tcgen05.st.{shape_value.value}.x{count_value}"
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=operands,
    )


@impl(tcgen05_stub.tcgen05_load)
def tcgen05_load_impl(
    shape: Var,
    tensor_memory_address: Var,
    count: Var,
    pack: Var,
    offset: Var,
) -> Var:
    require_pointer_in_memory_space(tensor_memory_address, (MemorySpace.TENSOR,))
    shape_value = require_constant_enum(shape, Tcgen05LoadStoreShape)
    count_value = require_constant_int(count)
    valid_counts = TCGEN05_VALID_COUNTS_BY_SHAPE[shape_value]
    if count_value not in valid_counts:
        valid = ", ".join(str(value) for value in valid_counts)
        raise InvalidValueError(
            f"Expected count for {shape_value.name} to be one of {valid}, got {count_value}"
        )

    has_offset = not is_none(offset)
    uses_offset = shape_value is Tcgen05LoadStoreShape.SHAPE_16X32BX2
    if uses_offset and not has_offset:
        raise TypeCheckingError("tcgen05_load with SHAPE_16X32BX2 requires offset")
    if has_offset and not uses_offset:
        raise TypeCheckingError("tcgen05_load offset is only valid with SHAPE_16X32BX2")

    operands = [tensor_memory_address]
    if has_offset:
        require_scalar_type(offset)
        operands.append(astype(offset, datatype.int64))

    if is_none(pack):
        operands.append(strictly_typed_const(False, ScalarTy(datatype.bool_)))
    else:
        require_scalar_type(
            pack,
            lambda dtype: dtype is datatype.bool_,
            f"Expected pack dtype to be {datatype.bool_}",
        )
        operands.append(pack)

    intrinsic = f"llvm.nvvm.tcgen05.ld.{shape_value.value}.x{count_value}"
    total_registers = count_value * TCGEN05_REGISTERS_PER_COUNT[shape_value]
    result_type = (
        ScalarTy(datatype.int32)
        if total_registers == 1
        else VectorTy(datatype.int32, total_registers)
    )

    [result] = add_operation_variadic(
        RawNVVMIntrinsic,
        (result_type,),
        intrinsic=intrinsic,
        operands_=tuple(operands),
    )
    return result


TCGEN05_VALID_BLOCK_SCALE_COMBINATIONS = (
    (
        Tcgen05MMABlockScaleKind.MXF8F6F4,
        Tcgen05MMAScaleVectorSize.DEFAULT,
    ),
    (
        Tcgen05MMABlockScaleKind.MXF8F6F4,
        Tcgen05MMAScaleVectorSize.BLOCK_32,
    ),
    (
        Tcgen05MMABlockScaleKind.MXF4,
        Tcgen05MMAScaleVectorSize.DEFAULT,
    ),
    (
        Tcgen05MMABlockScaleKind.MXF4,
        Tcgen05MMAScaleVectorSize.BLOCK_32,
    ),
    (
        Tcgen05MMABlockScaleKind.MXF4NVF4,
        Tcgen05MMAScaleVectorSize.BLOCK_16,
    ),
    (
        Tcgen05MMABlockScaleKind.MXF4NVF4,
        Tcgen05MMAScaleVectorSize.BLOCK_32,
    ),
)


def _tcgen05_mma_matrix_a(var: Var) -> tuple[Var, bool]:
    ty = var.get_type()

    def error():
        return make_type_checking_error(
            "Expected a tensor memory pointer or a shared memory descriptor "
            f"encoded as a 64 bit integer but got {ty}"
        )

    match ty:
        case PointerTy() as pt:
            info = datatype.PointerInfo(pt.pointer_dtype)
            if info.memory_space is not MemorySpace.TENSOR:
                raise error()
            return var, True
        case ScalarTy() as st:
            if not datatype.is_integral(st.dtype) or st.dtype.bitwidth != 64:
                raise error()
            return astype(var, datatype.int64), False
        case _:
            raise error()


class _Tcgen05MMAOperands(NamedTuple):
    operands: tuple[Var, ...]
    matrix_a_is_tensor: bool


def _tcgen05_mma_operands(
    matrix_d: Var,
    matrix_a: Var,
    matrix_b: Var,
    instruction_descriptor: Var,
    accumulate: Var,
) -> _Tcgen05MMAOperands:
    require_pointer_in_memory_space(matrix_d, (MemorySpace.TENSOR,))
    matrix_a, matrix_a_is_tensor = _tcgen05_mma_matrix_a(matrix_a)

    require_integral_scalar_type(matrix_b, bitwidth=64)
    matrix_b = astype(matrix_b, datatype.int64)

    require_integral_scalar_type(instruction_descriptor, bitwidth=32)
    instruction_descriptor = astype(instruction_descriptor, datatype.int32)
    accumulate = implicit_cast(accumulate, datatype.bool_, "accumulate as bool_")

    return _Tcgen05MMAOperands(
        operands=(
            matrix_d,
            matrix_a,
            matrix_b,
            instruction_descriptor,
            accumulate,
        ),
        matrix_a_is_tensor=matrix_a_is_tensor,
    )


def require_optional_pointer_in_memory_space(
    var: Var, spaces: tuple[MemorySpace, ...]
) -> Var | None:
    if is_none(var):
        return None
    require_pointer_in_memory_space(var, spaces)
    return var


def _i32_const(value: int) -> Var:
    return strictly_typed_const(value, ScalarTy(datatype.int32))


def _mma_kind_flag(kind: Tcgen05MMAKind) -> int:
    match kind:
        case Tcgen05MMAKind.F16:
            return 0
        case Tcgen05MMAKind.TF32:
            return 1
        case Tcgen05MMAKind.F8F6F4:
            return 2
        case Tcgen05MMAKind.I8:
            return 3
    assert False


def _collector_op_flag(collector_op: Tcgen05MMACollectorOp) -> int:
    match collector_op:
        case Tcgen05MMACollectorOp.DISCARD:
            return 0
        case Tcgen05MMACollectorOp.LASTUSE:
            return 1
        case Tcgen05MMACollectorOp.FILL:
            return 2
        case Tcgen05MMACollectorOp.USE:
            return 3
    assert False


def _collector_b_buffer_flag(
    collector_b_buffer: Tcgen05MMACollectorBBuffer,
) -> int:
    match collector_b_buffer:
        case Tcgen05MMACollectorBBuffer.BUFFER_0:
            return 0
        case Tcgen05MMACollectorBBuffer.BUFFER_1:
            return 1
        case Tcgen05MMACollectorBBuffer.BUFFER_2:
            return 2
        case Tcgen05MMACollectorBBuffer.BUFFER_3:
            return 3
    assert False


def _block_scale_name(kind: Tcgen05MMABlockScaleKind) -> str:
    match kind:
        case Tcgen05MMABlockScaleKind.MXF8F6F4:
            return "mxf8f6f4"
        case Tcgen05MMABlockScaleKind.MXF4:
            return "mxf4"
        case Tcgen05MMABlockScaleKind.MXF4NVF4:
            return "mxf4nvf4"
    assert False


def _scale_vector_suffix(
    scale_vector_size: Tcgen05MMAScaleVectorSize,
) -> str:
    match scale_vector_size:
        case Tcgen05MMAScaleVectorSize.DEFAULT:
            return ""
        case Tcgen05MMAScaleVectorSize.BLOCK_16:
            return ".block16"
        case Tcgen05MMAScaleVectorSize.BLOCK_32:
            return ".block32"
    assert False


@impl(tcgen05_stub.tcgen05_mma)
def tcgen05_mma_impl(
    kind: Var[Any],
    matrix_d: Var[Any],
    matrix_a: Var[Any],
    matrix_b: Var[Any],
    instruction_descriptor: Var[Any],
    accumulate: Var[Any],
    cta_group: Var[Any],
    sparse_metadata: Var[Any],
    scale_input_d: Var[Any],
    disable_output_lane: Var[Any],
    collector_op: Var[Any],
    a_shift: Var[Any],
) -> None:
    kind_value = require_constant_enum(kind, Tcgen05MMAKind)
    cta_group_value = require_constant_enum(cta_group, CTAGroup)
    collector_op_value = require_constant_enum(collector_op, Tcgen05MMACollectorOp)
    a_shift_value = require_constant_bool(a_shift)

    mma_operands = _tcgen05_mma_operands(
        matrix_d,
        matrix_a,
        matrix_b,
        instruction_descriptor,
        accumulate,
    )
    operands = list(mma_operands.operands)
    matrix_a_is_tensor = mma_operands.matrix_a_is_tensor

    sparse_metadata = require_optional_pointer_in_memory_space(
        sparse_metadata, (MemorySpace.TENSOR,)
    )
    if sparse_metadata is not None:
        operands.append(sparse_metadata)

    has_scale_input_d = not is_none(scale_input_d)
    if has_scale_input_d:
        if kind_value not in (Tcgen05MMAKind.F16, Tcgen05MMAKind.TF32):
            raise InvalidValueError(
                "scale_input_d is only supported for F16 and TF32 MMA kinds"
            )
        scale_value = require_constant_int(scale_input_d)
        if scale_value < 0 or scale_value > 15:
            raise InvalidValueError("scale_input_d must be an immediate in [0, 15]")
        operands.append(astype(scale_input_d, datatype.int64))

    has_disable_output_lane = not is_none(disable_output_lane)
    if has_disable_output_lane:
        expected_len = 4 if cta_group_value is CTAGroup.CTA_1 else 8
        require_vector_type(disable_output_lane, expected_len)
        operands.append(astype(disable_output_lane, datatype.int32))

    if a_shift_value:
        if not matrix_a_is_tensor:
            raise make_type_checking_error(
                "a_shift can only be applied if A is in tensor memory", a_shift
            )
        if collector_op_value in (
            Tcgen05MMACollectorOp.FILL,
            Tcgen05MMACollectorOp.USE,
        ):
            raise InvalidValueError(
                "a_shift cannot be combined with collector operation FILL or USE"
            )

    intrinsic_parts = ["llvm", "nvvm", "tcgen05", "mma"]
    if sparse_metadata is not None:
        intrinsic_parts.append("sp")
    intrinsic_parts.append("tensor" if matrix_a_is_tensor else "shared")
    if has_scale_input_d:
        intrinsic_parts.append("scale_d")
    if has_disable_output_lane:
        intrinsic_parts.extend(("disable_output_lane", cta_group_value.value))
    if a_shift_value:
        intrinsic_parts.append("ashift")

    operands.append(_i32_const(_mma_kind_flag(kind_value)))
    if not has_disable_output_lane:
        operands.append(_i32_const(1 if cta_group_value is CTAGroup.CTA_1 else 2))
    operands.append(_i32_const(_collector_op_flag(collector_op_value)))
    operands.append(_i32_const(0))

    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=".".join(intrinsic_parts),
        operands_=tuple(operands),
    )


@impl(tcgen05_stub.tcgen05_mma_block_scale)
def tcgen05_mma_block_scale_impl(
    kind: Var[Any],
    matrix_d: Var[Any],
    matrix_a: Var[Any],
    matrix_b: Var[Any],
    instruction_descriptor: Var[Any],
    scale_a: Var[Any],
    scale_b: Var[Any],
    accumulate: Var[Any],
    sparse_metadata: Var[Any],
    cta_group: Var[Any],
    scale_vector_size: Var[Any],
    collector_op: Var[Any],
) -> None:
    kind_value = require_constant_enum(kind, Tcgen05MMABlockScaleKind)
    cta_group_value = require_constant_enum(cta_group, CTAGroup)
    scale_vector_size_value = require_constant_enum(
        scale_vector_size, Tcgen05MMAScaleVectorSize
    )
    collector_op_value = require_constant_enum(collector_op, Tcgen05MMACollectorOp)

    if (
        kind_value,
        scale_vector_size_value,
    ) not in TCGEN05_VALID_BLOCK_SCALE_COMBINATIONS:
        raise InvalidValueError(
            "Invalid tcgen05 block-scale kind and scale-vector-size combination: "
            f"{kind_value.name}, {scale_vector_size_value.name}"
        )

    mma_operands = _tcgen05_mma_operands(
        matrix_d,
        matrix_a,
        matrix_b,
        instruction_descriptor,
        accumulate,
    )
    operands = list(mma_operands.operands)
    matrix_a_is_tensor = mma_operands.matrix_a_is_tensor

    sparse_metadata = require_optional_pointer_in_memory_space(
        sparse_metadata, (MemorySpace.TENSOR,)
    )
    if sparse_metadata is not None:
        operands.append(sparse_metadata)

    require_pointer_in_memory_space(scale_a, (MemorySpace.TENSOR,))
    require_pointer_in_memory_space(scale_b, (MemorySpace.TENSOR,))
    operands.extend((scale_a, scale_b))

    intrinsic_parts = ["llvm", "nvvm", "tcgen05", "mma"]
    if sparse_metadata is not None:
        intrinsic_parts.append("sp")
    intrinsic_parts.extend(
        (
            "tensor" if matrix_a_is_tensor else "shared",
            _block_scale_name(kind_value),
            "block_scale",
        )
    )
    intrinsic = ".".join(intrinsic_parts)
    intrinsic += _scale_vector_suffix(scale_vector_size_value)

    operands.extend(
        (
            _i32_const(1 if cta_group_value is CTAGroup.CTA_1 else 2),
            _i32_const(_collector_op_flag(collector_op_value)),
            _i32_const(0),
        )
    )
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=tuple(operands),
    )


@impl(tcgen05_stub.tcgen05_mma_weight_stationary)
def tcgen05_mma_weight_stationary_impl(
    kind: Var[Any],
    matrix_d: Var[Any],
    matrix_a: Var[Any],
    matrix_b: Var[Any],
    instruction_descriptor: Var[Any],
    accumulate: Var[Any],
    sparse_metadata: Var[Any],
    zero_column_mask: Var[Any],
    collector_op: Var[Any],
    collector_b_buffer: Var[Any],
) -> None:
    kind_value = require_constant_enum(kind, Tcgen05MMAKind)
    collector_op_value = require_constant_enum(collector_op, Tcgen05MMACollectorOp)
    collector_b_buffer_value = require_constant_enum(
        collector_b_buffer, Tcgen05MMACollectorBBuffer
    )

    mma_operands = _tcgen05_mma_operands(
        matrix_d,
        matrix_a,
        matrix_b,
        instruction_descriptor,
        accumulate,
    )
    operands = list(mma_operands.operands)
    matrix_a_is_tensor = mma_operands.matrix_a_is_tensor

    sparse_metadata = require_optional_pointer_in_memory_space(
        sparse_metadata, (MemorySpace.TENSOR,)
    )
    if sparse_metadata is not None:
        operands.append(sparse_metadata)

    has_zero_column_mask = not is_none(zero_column_mask)
    if has_zero_column_mask:
        require_integral_scalar_type(zero_column_mask)
        operands.append(astype(zero_column_mask, datatype.int64))

    intrinsic_parts = ["llvm", "nvvm", "tcgen05", "mma", "ws"]
    if sparse_metadata is not None:
        intrinsic_parts.append("sp")
    intrinsic_parts.append("tensor" if matrix_a_is_tensor else "shared")
    if has_zero_column_mask:
        intrinsic_parts.append("zero_col_mask")

    operands.extend(
        (
            _i32_const(_mma_kind_flag(kind_value)),
            _i32_const(_collector_b_buffer_flag(collector_b_buffer_value)),
            _i32_const(_collector_op_flag(collector_op_value)),
        )
    )
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=".".join(intrinsic_parts),
        operands_=tuple(operands),
    )


@impl(tcgen05_stub.Tcgen05SharedMemoryDescriptor.encode)
def tcgen05_shared_memory_descriptor_encode_impl(self: Var) -> Var:
    descriptor = self.get_aggregate()
    assert isinstance(descriptor, DataclassValue)

    u64_ty = ScalarTy(datatype.uint64)

    def set_bits(value: Var, field: Var, position: int, width: int) -> Var:
        field_mask = (1 << width) - 1
        mask = field_mask << position
        clear_mask = 0xFFFF_FFFF_FFFF_FFFF - mask
        clear_mask = strictly_typed_const(clear_mask, u64_ty)
        field_mask = strictly_typed_const(field_mask, u64_ty)
        position = strictly_typed_const(position, u64_ty)
        field_and_mask = binary_bitwise_tensorlike("and_", field, field_mask)
        insert = bitwise_shift_tensorlike("lshift", field_and_mask, position)
        value_and_clear_mask = binary_bitwise_tensorlike("and_", value, clear_mask)
        return binary_bitwise_tensorlike("or_", value_and_clear_mask, insert)

    leading_dimension_mode = require_constant_int(
        descriptor.get_field("leading_dimension_mode")
    )
    swizzle_mode = descriptor.get_field("swizzle_mode")
    swizzle_mode = require_constant_enum(swizzle_mode, SwizzleMode)
    swizzle_encoding = {
        SwizzleMode.SWIZZLE_NONE: 0,
        SwizzleMode.SWIZZLE_128B_ATOM_32B: 1,
        SwizzleMode.SWIZZLE_128B: 2,
        SwizzleMode.SWIZZLE_64B: 4,
        SwizzleMode.SWIZZLE_32B: 6,
    }
    if swizzle_mode not in swizzle_encoding:
        raise InvalidValueError(
            f"Swizzle mode {swizzle_mode.name} is not supported by "
            "tcgen05 shared-memory descriptors"
        )

    c_0x3ffff = strictly_typed_const(0x3FFFF, u64_ty)
    c_4 = strictly_typed_const(4, u64_ty)
    value = strictly_typed_const(0, u64_ty)
    matrix_start_address = descriptor.get_field("matrix_start_address")

    address_type = matrix_start_address.get_type()
    match address_type:
        case ScalarTy():
            require_integral_scalar_type(matrix_start_address)
        case PointerTy():
            require_pointer_in_memory_space(matrix_start_address, [MemorySpace.SHARED])
            matrix_start_address = bitcast(matrix_start_address, datatype.int32)
        case ArrayTy():
            pointer = matrix_start_address.get_aggregate().base_ptr
            require_pointer_in_memory_space(pointer, [MemorySpace.SHARED])
            matrix_start_address = bitcast(pointer, datatype.int32)
        case _:
            raise TypeCheckingError(
                "Matrix start address must be a pointer or array in shared "
                "memory or already encoded as an integer"
            )

    def encode_field(value, field, position):
        field = astype(field, datatype.uint64)
        field = binary_bitwise_tensorlike("and_", field, c_0x3ffff)
        field = bitwise_shift_tensorlike("rshift", field, c_4)
        return set_bits(value, field, position, 14)

    matrix_start_address = astype(matrix_start_address, datatype.uint64)
    value = encode_field(value, matrix_start_address, 0)
    field = descriptor.get_field("leading_dimension_byte_offset")
    value = encode_field(value, field, 16)
    field = descriptor.get_field("stride_dimension_byte_offset")
    value = encode_field(value, field, 32)

    value = set_bits(value, strictly_typed_const(0b001, u64_ty), 46, 3)

    base_offset = astype(descriptor.get_field("base_offset"), datatype.uint64)
    value = set_bits(value, base_offset, 49, 3)

    leading_dimension_mode = strictly_typed_const(leading_dimension_mode, u64_ty)
    value = set_bits(value, leading_dimension_mode, 52, 1)

    swizzle_value = strictly_typed_const(swizzle_encoding[swizzle_mode], u64_ty)
    value = set_bits(value, swizzle_value, 61, 3)
    return value
