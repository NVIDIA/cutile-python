# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._exception import UnsupportedFeatureError, TypeCheckingError
from cuda.lang._target import TargetInfo
from cuda.lang._ir.op_defs import RawLLVMIntrinsic
from cuda.tile._ir.op_impl import require_constant_int
from cuda.tile._ir.op_impl import require_constant_bool
from cuda.tile._ir.ir import Var
from cuda.tile._ir.op_impl import ImplRegistry
from cuda.tile._ir.ops import implicit_cast
from cuda.tile._ir.ir import add_operation_variadic
from cuda.tile._ir.cast_ops import address_space_cast
import cuda.lang._datatype as datatype
from cuda.lang._enums import MemorySpace, AsyncCopyReportMechanism, CTAGroup, CacheHint
from cuda.lang._ir.op_defs import (
    CopyAsyncBulkTensorGlobalToShared,
    CopyAsyncBulkTensorSharedToGlobal,
)
from cuda.lang._stub import copy_async
from ..type_checking_helpers import (
    optional_cast,
    is_none,
    make_type_checking_error,
    require_mbarrier_ptr,
    require_none,
    require_pointer_in_memory_space,
    require_target_greater_equal,
    require_uniform_int_tuple_type,
    tensor_map_descriptor_pointer_like,
    require_integral_scalar_type,
    require_pointer_type,
)
from cuda.tile._ir.op_impl import require_constant_enum, require_optional_constant_enum


_registry = ImplRegistry()
impl = _registry.impl


def copy_async_impl_registry() -> ImplRegistry:
    return _registry


def _optional_operand(value):
    return None if value is None or is_none(value) else value


@impl(copy_async.copy_async_global_to_shared)
def copy_async_global_to_shared_impl(src: Var, dst: Var, cache_hint: Var,
                                     dst_size: Var, src_size: Var):
    require_target_greater_equal(8, 0, " does not support copy_async_global_to_shared")
    require_pointer_in_memory_space(dst, (MemorySpace.SHARED,))

    src = address_space_cast(src, MemorySpace.GLOBAL) if \
        require_pointer_type(src).memory_space == MemorySpace.GENERIC else src
    require_pointer_in_memory_space(src, (MemorySpace.GLOBAL,))

    cache_hint_val = require_constant_enum(cache_hint, CacheHint)
    dst_size_val = require_constant_int(dst_size)

    if cache_hint_val not in (CacheHint.CA, CacheHint.CG):
        raise TypeCheckingError(f"cache_hint must be CA or CG, got {cache_hint_val}")
    if cache_hint_val == CacheHint.CA and dst_size_val not in (16, 8, 4):
        raise TypeCheckingError(f"dst_size must be 4, 8, or 16 for CA, "
                                f"got {dst_size_val}")
    if cache_hint_val == CacheHint.CG and dst_size_val != 16:
        raise TypeCheckingError(f"dst_size must be 16 for CG, got {dst_size_val}")

    intrinsic = f"llvm.nvvm.cp.async.{cache_hint_val.value}.shared.global.{dst_size_val}"
    args = (dst, src)
    if not is_none(src_size):
        require_integral_scalar_type(src_size, bitwidth=32)
        intrinsic += ".s"
        args += (src_size,)
    add_operation_variadic(
        RawLLVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=args,
    )


@impl(copy_async.copy_async_commit_group)
def copy_async_commit_group_impl():
    require_target_greater_equal(8, 0, " does not support copy_async_commit_group_impl")
    intrinsic = "llvm.nvvm.cp.async.commit.group"
    add_operation_variadic(
        RawLLVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(),
    )


@impl(copy_async.copy_async_wait_groups)
def copy_async_wait_groups_impl(count: Var):
    require_target_greater_equal(8, 0, " does not support copy_async_wait_groups")
    require_constant_int(count)
    require_integral_scalar_type(count, bitwidth=32)
    intrinsic = "llvm.nvvm.cp.async.wait.group"
    add_operation_variadic(
        RawLLVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(count,),
    )


@impl(copy_async.copy_async_mbarrier_arrive, fixed_args=["arrive"])
@impl(copy_async.copy_async_mbarrier_arrive_no_increment, fixed_args=["arrive.noinc"])
def copy_async_mbarrier_arrive_impl(kind: str, mbar: Var) -> None:
    require_target_greater_equal(8, 0, "copy_async_mbarrier_arrive")
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    intrinsic = f"llvm.nvvm.cp.async.mbarrier.{kind}.shared"
    add_operation_variadic(
        RawLLVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(mbar,),
    )


def validate_g2s_mode(mode: copy_async.TMALoadMode, im2col_count: int) -> None:
    match mode:
        case copy_async.TMALoadMode.TILE | copy_async.TMALoadMode.TILE_GATHER4:
            if im2col_count != 0:
                raise make_type_checking_error(
                    f"{mode.name} mode does not accept im2col_offsets"
                )

        case (
            copy_async.TMALoadMode.IM2COL
            | copy_async.TMALoadMode.IM2COL_W
            | copy_async.TMALoadMode.IM2COL_W_128
        ):
            if im2col_count == 0:
                raise make_type_checking_error(
                    f"{mode.name} mode requires im2col_offsets"
                )

        case _:
            raise make_type_checking_error(f"Unsupported TMA load mode {mode}")


@impl(copy_async.copy_async_bulk_tensor_global_to_shared)
def copy_async_bulk_tensor_global_to_shared_impl(
    src_tensor_map_descriptor,
    src_coordinates,
    dst_memory,
    mbarrier,
    im2col_offsets,
    multicast_mask,
    l2_cache_hint,
    mode,
    cta_group,
    report_mechanism,
):
    src_coordinate_vars = require_uniform_int_tuple_type(src_coordinates)
    im2col_offset_vars = require_uniform_int_tuple_type(im2col_offsets)
    require_mbarrier_ptr(mbarrier, (MemorySpace.SHARED,))
    mode = require_constant_enum(mode, copy_async.TMALoadMode)
    validate_g2s_mode(mode, len(im2col_offset_vars))
    tensor_map = tensor_map_descriptor_pointer_like(src_tensor_map_descriptor)
    dst_ty = require_pointer_in_memory_space(
        dst_memory,
        (MemorySpace.SHARED, MemorySpace.SHARED_CLUSTER),
    )
    is_cta_only = dst_ty.memory_space == MemorySpace.SHARED
    cta_group_value = None
    src_coordinates = tuple(
        implicit_cast(coord, datatype.int32, "TMA coordinates")
        for coord in src_coordinate_vars
    )
    im2col_offsets = tuple(
        implicit_cast(offset, datatype.int16, "TMA im2col offsets")
        for offset in im2col_offset_vars
    )
    l2_cache_hint = optional_cast(l2_cache_hint, datatype.int64, "TMA L2 cache hint")
    report_mechanism = require_constant_enum(report_mechanism, AsyncCopyReportMechanism)

    if report_mechanism is not AsyncCopyReportMechanism.DISABLED:
        target = TargetInfo.get_current()
        if not (target.major == 10 and target.minor >= 7 and target.suffix in ("a", "f")):
            raise UnsupportedFeatureError(
                "Async-copy reporting requires sm_107f or higher in the same family"
            )

    if is_cta_only:
        message = (
            "When the destination memory is in shared memory, the "
            "multicast mask and cta_group arguments are invalid."
        )
        require_none(multicast_mask, message)
        require_none(cta_group, message)
    else:
        multicast_mask = optional_cast(
            multicast_mask, datatype.int16, "TMA multicast mask"
        )
        cta_group_value = require_optional_constant_enum(
            cta_group, CTAGroup
        )
    add_operation_variadic(
        CopyAsyncBulkTensorGlobalToShared,
        (),
        dst_memory=dst_memory,
        tensor_map=tensor_map,
        coordinates=src_coordinates,
        mbarrier=mbarrier,
        im2col_offsets=im2col_offsets,
        multicast_mask=(
            None if is_cta_only else _optional_operand(multicast_mask)
        ),
        l2_cache_hint=_optional_operand(l2_cache_hint),
        mode=mode,
        is_cta_only=is_cta_only,
        cta_group=cta_group_value,
        report_mechanism=report_mechanism,
    )


@impl(copy_async.copy_async_bulk_tensor_shared_to_global)
def copy_async_bulk_tensor_shared_to_global_impl(
    src_memory,
    dst_tensor_map_descriptor,
    dst_coordinates,
    l2_cache_hint,
    mode,
):
    require_pointer_in_memory_space(src_memory, (MemorySpace.SHARED,))
    tensor_map = tensor_map_descriptor_pointer_like(dst_tensor_map_descriptor)
    dst_coordinate_vars = require_uniform_int_tuple_type(dst_coordinates)
    mode = require_constant_enum(mode, copy_async.TMAStoreMode)
    dst_coordinates = tuple(
        implicit_cast(coord, datatype.int32, "TMA coordinates")
        for coord in dst_coordinate_vars
    )
    l2_cache_hint = optional_cast(l2_cache_hint, datatype.int64, "TMA L2 cache hint")
    add_operation_variadic(
        CopyAsyncBulkTensorSharedToGlobal,
        (),
        tensor_map=tensor_map,
        src_memory=src_memory,
        coordinates=dst_coordinates,
        l2_cache_hint=_optional_operand(l2_cache_hint),
        mode=mode,
    )


@impl(copy_async.copy_async_bulk_wait_group)
def copy_async_bulk_wait_group_impl(number_of_groups: Var[int], read: Var[bool]):
    require_constant_int(number_of_groups)
    read = require_constant_bool(read)
    add_operation_variadic(
        RawLLVMIntrinsic,
        tuple(),
        intrinsic="llvm.nvvm.cp.async.bulk.wait.group" + (".read" if read else ""),
        operands_=(number_of_groups,),
    )
