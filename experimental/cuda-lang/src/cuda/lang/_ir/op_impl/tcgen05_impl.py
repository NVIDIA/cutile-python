# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from cuda.tile._ir.cast_ops import implicit_cast
import cuda.lang._datatype as datatype
from cuda.lang._enums import (
    Tcgen05MMAKind,
    Tcgen05LdStShape,
    Tcgen05MMACollectorOp,
    CTAGroup,
)
from cuda.lang._exception import TileInternalError
from cuda.lang._ir.type import PointerTy
from cuda.lang._stub import tcgen05 as tcgen05_stub
from cuda.lang._ir.ir import Var
from cuda.lang._ir.ops import (
    MemorySpace,
    ScalarTy,
    TileTypeError,
    VectorTy,
    add_operation_variadic,
    astype,
    require_scalar_type,
    strictly_typed_const,
)
from cuda.lang._ir.op_defs import RawMLIROperation, RawNVVMIntrinsic
from cuda.lang._ir.type_checking_helpers import (
    is_none,
    make_type_checking_error,
    require_integral_scalar_type,
    require_mbarrier_ptr,
    require_pointer_in_memory_space,
    require_vector_type,
)
from cuda.tile._exception import TileValueError
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_bool,
    require_constant_enum,
    require_constant_int,
    require_optional_constant_enum,
)
import cuda.lang._mlir.nvvm as mlir


_registry = ImplRegistry()
impl = _registry.impl


def tcgen05_impl_registry() -> ImplRegistry:
    return _registry


_TCGEN05_LD_VALID_COUNTS_BY_SHAPE = {
    Tcgen05LdStShape.SHAPE_16X64B: (1, 2, 4, 8, 16, 32, 64, 128),
    Tcgen05LdStShape.SHAPE_16X128B: (1, 2, 4, 8, 16, 32, 64),
    Tcgen05LdStShape.SHAPE_16X256B: (1, 2, 4, 8, 16, 32),
    Tcgen05LdStShape.SHAPE_32X32B: (1, 2, 4, 8, 16, 32, 64, 128),
    Tcgen05LdStShape.SHAPE_16X32BX2: (1, 2, 4, 8, 16, 32, 64, 128),
}

_TCGEN05_LD_REGISTERS_PER_COUNT = {
    Tcgen05LdStShape.SHAPE_16X64B: 1,
    Tcgen05LdStShape.SHAPE_16X128B: 2,
    Tcgen05LdStShape.SHAPE_16X256B: 4,
    Tcgen05LdStShape.SHAPE_32X32B: 1,
    Tcgen05LdStShape.SHAPE_16X32BX2: 1,
}


@impl(tcgen05_stub.tcgen05_alloc)
def tcgen05_alloc_impl(
    addr: Var,
    ncols: Var,
    cta_group: Var,
) -> None:
    require_pointer_in_memory_space(
        addr, (MemorySpace.SHARED_CLUSTER, MemorySpace.SHARED)
    )
    ncols = implicit_cast(ncols, datatype.int32, "cast num columns to int32")
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.alloc.shared." + cta_group_value.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(addr, ncols),
    )


@impl(tcgen05_stub.tcgen05_dealloc)
def tcgen05_dealloc_impl(
    addr: Var,
    ncols: Var,
    cta_group: Var,
) -> None:
    require_pointer_in_memory_space(addr, (MemorySpace.TENSOR,))
    ncols = implicit_cast(ncols, datatype.int32, "cast num columns to int32")
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.dealloc." + cta_group_value.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(addr, ncols),
    )


@impl(tcgen05_stub.tcgen05_commit)
def tcgen05_commit_impl(
    mbar: Var,
    multicast_mask: Var,
    cta_group: Var,
):
    require_mbarrier_ptr(mbar)
    operands = [mbar]
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


@impl(tcgen05_stub.tcgen05_ld)
def tcgen05_ld_impl(
    shape: Var,
    tmem_addr: Var,
    count: Var,
    pack: Var,
    offset: Var,
) -> Var:
    require_pointer_in_memory_space(tmem_addr, (MemorySpace.TENSOR,))
    shape_value = cast(Tcgen05LdStShape, require_constant_enum(shape, Tcgen05LdStShape))
    count_value = cast(int, require_constant_int(count))
    valid_counts = _TCGEN05_LD_VALID_COUNTS_BY_SHAPE[shape_value]
    if count_value not in valid_counts:
        valid = ", ".join(str(value) for value in valid_counts)
        raise TileValueError(
            f"Expected count for {shape_value.name} to be one of {valid}, got {count_value}"
        )

    has_offset = not is_none(offset)
    uses_offset = shape_value is Tcgen05LdStShape.SHAPE_16X32BX2
    if uses_offset and not has_offset:
        raise TileTypeError("tcgen05_ld with SHAPE_16X32BX2 requires offset")
    if has_offset and not uses_offset:
        raise TileTypeError("tcgen05_ld offset is only valid with SHAPE_16X32BX2")

    operands = [tmem_addr]
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
    total_registers = count_value * _TCGEN05_LD_REGISTERS_PER_COUNT[shape_value]
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


def optional_enum_to_mlir_attribute(cl_enum_value, mlir_enum):
    if cl_enum_value is None:
        return None
    return enum_to_mlir_attribute(cl_enum_value, mlir_enum)


def enum_to_mlir_attribute(cl_enum_value, mlir_enum):
    mlir_attribute_class = mlir_enum.__name__ + "Attr"
    mlir_attribute = getattr(mlir, mlir_attribute_class, None)
    if mlir_attribute is None:
        raise TileInternalError(
            f"Expected mlir module to have class {mlir_attribute_class} "
            "but it could not be found"
        )
    mlir_enum_value = getattr(mlir_enum, cl_enum_value.name, None)
    if mlir_enum_value is None:
        raise TileInternalError(
            f"Expected enum {type(cl_enum_value)} to have corresponding "
            "enum in mlir bindings but it could not be found"
        )
    return mlir_attribute(value=mlir_enum_value)


def _require_tcgen05_mma_matrix_a(var: Var):

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
            return var
        case ScalarTy() as st:
            if not datatype.is_integral(st.dtype) or st.dtype.bitwidth != 64:
                raise error()
            return astype(var, datatype.int64)
        case _:
            raise error()


@impl(tcgen05_stub.tcgen05_mma)
def tcgen05_mma_impl(
    kind: Var[Any],
    cta_group: Var[Any],
    matrix_d: Var[Any],
    matrix_a: Var[Any],
    matrix_b: Var[Any],
    idesc: Var[Any],
    enable_input_d: Var[Any],
    scale_input_d: Var[Any],
    disable_output_lane: Var[Any],
    collector_op: Var[Any],
    a_shift: Var[Any],
):
    kind = require_constant_enum(kind, Tcgen05MMAKind)
    cta_group = require_optional_constant_enum(cta_group, CTAGroup) or CTAGroup.CTA_1
    collector_op = require_constant_enum(collector_op, Tcgen05MMACollectorOp)
    require_pointer_in_memory_space(matrix_d, (MemorySpace.TENSOR,))
    matrix_a = _require_tcgen05_mma_matrix_a(matrix_a)
    require_integral_scalar_type(matrix_b, bitwidth=64)
    require_integral_scalar_type(idesc)
    idesc = implicit_cast(idesc, datatype.int32, "idesc as int32")
    enable_input_d = implicit_cast(
        enable_input_d, datatype.bool_, "enable_input_d as bool_"
    )

    operands = [
        matrix_d,
        matrix_a,
        matrix_b,
        idesc,
        enable_input_d,
    ]
    operand_counts = [1 for _ in operands]

    if not is_none(scale_input_d):
        value = require_constant_int(scale_input_d)
        if value < 0 or value > 15:
            raise make_type_checking_error(
                "Expected scale_input_d to be an immediate in [0, 15]"
            )
        scale_input_d = astype(scale_input_d, datatype.int32)
        operands += [scale_input_d]
        operand_counts += [1]
    else:
        operand_counts += [0]

    if not is_none(disable_output_lane):
        expected_len = 4 if cta_group is CTAGroup.CTA_1 else 8
        require_vector_type(disable_output_lane, expected_len)
        disable_output_lane = astype(disable_output_lane, datatype.int32)
        operands += [disable_output_lane]
        operand_counts += [1]
    else:
        operand_counts += [0]

    attributes = dict(
        mmaKind=enum_to_mlir_attribute(kind, mlir.Tcgen05MMAKind),
        ctaGroup=enum_to_mlir_attribute(cta_group, mlir.CTAGroupKind),
        collectorOp=enum_to_mlir_attribute(collector_op, mlir.Tcgen05MMACollectorOp),
        operandSegmentSizes=mlir.DenseI32ArrayAttr(operand_counts),
    )

    if not is_none(a_shift):
        value = require_constant_bool(a_shift)
        if value:
            if not isinstance(matrix_a.get_type(), PointerTy):
                raise make_type_checking_error(
                    "a_shift can only be applied if A is in tensor memory", a_shift
                )
            attributes["aShift"] = mlir.UnitAttr()

    add_operation_variadic(
        RawMLIROperation,
        (),
        op_name="nvvm.tcgen05.mma",
        operands_=tuple(operands),
        mlir_attributes=tuple(attributes.items()),
    )
