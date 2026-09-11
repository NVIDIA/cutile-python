# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from cuda.lang._enums import PrefetchLevel, CachePolicy
from cuda.lang._ir.op_defs import RawLLVMIntrinsic
from cuda.lang._ir.type_checking_helpers import (
    require_pointer_type,
    tensor_map_descriptor_pointer_like,
)
from cuda.lang._stub.prefetch import prefetch, prefetch_uniform, prefetch_tensor_map
from cuda.tile._datatype import PointerInfo
from cuda.tile._exception import InvalidValueError, TypeCheckingError
from cuda.tile._ir.ir import Var, add_operation_variadic
from cuda.tile._ir.op_impl import ImplRegistry, require_constant_enum
from cuda.tile._ir.type import NONE
from cuda.tile._memory_model import MemorySpace

_registry = ImplRegistry()
impl = _registry.impl


def prefetch_impl_registry() -> ImplRegistry:
    return _registry


@impl(prefetch)
def prefetch_impl(address: Var, level: Var, eviction_priority: Var):
    address_ty = require_pointer_type(address)
    address_space = PointerInfo(address_ty.pointer_dtype).memory_space
    level = require_constant_enum(level, PrefetchLevel)

    if eviction_priority.get_type() == NONE:
        evict_str = ""
    else:
        eviction_priority = require_constant_enum(eviction_priority, CachePolicy)
        if level != PrefetchLevel.L2:
            raise InvalidValueError("Prefetch eviction priority is supported only for L2")
        priority_map = {CachePolicy.L2_EVICT_NORMAL: "normal", CachePolicy.L2_EVICT_LAST: "last"}
        suffix = priority_map.get(eviction_priority)
        if suffix is None:
            valid = ", ".join(x._name_ for x in priority_map.keys())
            raise InvalidValueError(f"Invalid eviction priority {eviction_priority._name_}."
                                    f" Accepted values: {valid}.")
        if address_space != MemorySpace.GLOBAL:
            raise TypeCheckingError(f"Pointer in GLOBAL address space is required"
                                    f" when eviction priority is specified;"
                                    f" received a {address_space._name_} pointer instead.")
        evict_str = f".evict.{suffix}"

    address_space_map = {
        MemorySpace.GENERIC: "",
        MemorySpace.GLOBAL: ".global",
        MemorySpace.LOCAL: ".local",
    }
    address_space_str = address_space_map.get(address_space)
    if address_space_str is None:
        valid = ", ".join(x._name_ for x in address_space_map.keys())
        raise TypeCheckingError(f"Invalid pointer address space {address_space._name_}."
                                f" Accepted address spaces: {valid}.")

    level_map = {PrefetchLevel.L1: "L1", PrefetchLevel.L2: "L2"}
    level_str = level_map[level]

    add_operation_variadic(
        RawLLVMIntrinsic,
        (),
        intrinsic=f"llvm.nvvm.prefetch{address_space_str}.{level_str}{evict_str}",
        operands_=(address,)
    )


@impl(prefetch_uniform)
def prefetch_uniform_impl(address: Var):
    address_ty = require_pointer_type(address)
    address_space = PointerInfo(address_ty.pointer_dtype).memory_space
    if address_space != MemorySpace.GENERIC:
        raise TypeCheckingError(f"Expected a pointer in GENERIC address space;"
                                f" received {address_space._name_} instead.")

    add_operation_variadic(
        RawLLVMIntrinsic,
        (),
        intrinsic="llvm.nvvm.prefetchu.L1",
        operands_=(address,)
    )


@impl(prefetch_tensor_map)
def prefetch_tensor_map_impl(tensor_map: Var):
    tensor_map = tensor_map_descriptor_pointer_like(
        tensor_map, memory_spaces=(MemorySpace.GENERIC, MemorySpace.CONSTANT)
    )
    add_operation_variadic(
        RawLLVMIntrinsic,
        (),
        intrinsic="llvm.nvvm.prefetch.tensormap",
        operands_=(tensor_map,)
    )
