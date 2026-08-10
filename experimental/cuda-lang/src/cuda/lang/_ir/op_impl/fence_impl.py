# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._enums import FenceProxy, MemoryOrder, MemoryScope
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.op_defs import Fence, RawNVVMIntrinsic
from cuda.lang._ir.type_checking_helpers import is_none, require_pointer_type
from cuda.lang._stub import fence as fence_stub
from cuda.tile._ir.ir import add_operation_variadic
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_enum,
    require_constant_int,
)
from cuda.tile._ir.type import DataclassTy, EnumTy


_registry = ImplRegistry()
impl = _registry.impl


_SCOPE_SUFFIX = {
    MemoryScope.BLOCK: "cta",
    MemoryScope.CLUSTER: "cluster",
    MemoryScope.DEVICE: "gpu",
    MemoryScope.SYS: "sys",
}

_ORDER_SUFFIX = {
    MemoryOrder.ACQUIRE: "acquire",
    MemoryOrder.RELEASE: "release",
    MemoryOrder.ACQ_REL: "acq_rel",
    MemoryOrder.SEQ_CST: "sc",
}

# The restrictions are spelled differently on bidirectional intrinsics, so we
# can't alwyas map the shared cta enum to the same string. For example:
# @llvm.nvvm.fence.proxy.async.shared_cluster
# @llvm.nvvm.fence.proxy.async_generic.release.sync_restrict.space.cta.scope.cluster
_RESTRICTION_SPACE_SUFFIX = {
    fence_stub._FenceRestrictionKind.SHARED_BLOCK: "cta",
    fence_stub._FenceRestrictionKind.SHARED_CLUSTER: "cluster",
    fence_stub._FenceRestrictionKind.GLOBAL: "global",
}

_BIDIRECTIONAL_RESTRICTION_SUFFIX = {
    fence_stub._FenceRestrictionKind.MBARRIER_INIT: "mbarrier_init",
    fence_stub._FenceRestrictionKind.SHARED_BLOCK: "shared_cta",
    fence_stub._FenceRestrictionKind.SHARED_CLUSTER: "shared_cluster",
    fence_stub._FenceRestrictionKind.GLOBAL: "global",
}


def fence_impl_registry() -> ImplRegistry:
    return _registry


__all__ = ("fence_impl_registry",)


def require_fence_order(order):
    order = require_constant_enum(order, MemoryOrder)
    valid = (
        MemoryOrder.ACQUIRE,
        MemoryOrder.RELEASE,
        MemoryOrder.ACQ_REL,
        MemoryOrder.SEQ_CST,
    )
    if order not in valid:
        formatted = ", ".join(str(value) for value in valid)
        raise TypeCheckingError(
            f"Invalid fence memory order {order}, expected one of {formatted}"
        )
    return order


def require_fence_scope(scope):
    scope = require_constant_enum(scope, MemoryScope)
    valid = (
        MemoryScope.BLOCK,
        MemoryScope.CLUSTER,
        MemoryScope.DEVICE,
        MemoryScope.SYS,
    )
    if scope not in valid:
        formatted = ", ".join(str(value) for value in valid)
        raise TypeCheckingError(
            f"Invalid fence memory scope {scope}, expected one of {formatted}"
        )
    return scope


def require_fence_restriction(restriction):
    if is_none(restriction):
        return None

    restriction_ty = restriction.get_type()
    if isinstance(restriction_ty, EnumTy):
        return require_constant_enum(restriction, fence_stub._FenceRestrictionKind)

    if (
        isinstance(restriction_ty, DataclassTy)
        and restriction_ty.cls is fence_stub._FenceAddressRestriction
    ):
        restriction_value = restriction.get_aggregate()
        address = restriction_value.get_field("address")
        size = restriction_value.get_field("size")
        require_pointer_type(address)
        size_value = require_constant_int(size)
        if size_value != 128:
            raise TypeCheckingError(
                f"An address restriction must have size 128, got {size_value}"
            )
        return restriction_value

    raise TypeCheckingError(f"Expected FenceRestriction or None, got {restriction_ty}")


def lower_non_proxy_fence(order, scope, restriction):
    if restriction is None:
        add_operation_variadic(Fence, (), memory_order=order, memory_scope=scope)
        return

    order_suffix = _ORDER_SUFFIX[order]
    scope_suffix = _SCOPE_SUFFIX[scope]
    operands = ()
    if restriction is fence_stub._FenceRestrictionKind.MBARRIER_INIT:
        intrinsic = f"llvm.nvvm.fence.mbarrier_init.{order_suffix}.{scope_suffix}"
    elif isinstance(restriction, fence_stub._FenceRestrictionKind):
        space_suffix = _RESTRICTION_SPACE_SUFFIX[restriction]
        intrinsic = (
            f"llvm.nvvm.fence.{order_suffix}.sync_restrict."
            f"space.{space_suffix}.scope.{scope_suffix}"
        )
    else:
        intrinsic = f"llvm.nvvm.fence.{order_suffix}.address.scope.{scope_suffix}"
        operands = (
            restriction.get_field("address"),
            restriction.get_field("size"),
        )

    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=operands,
    )


def lower_proxy_fence(order, scope, from_proxy, to_proxy, restriction):
    order_suffix = _ORDER_SUFFIX[order]
    scope_suffix = _SCOPE_SUFFIX[scope]
    proxy_suffix = f"{to_proxy.value}_{from_proxy.value}"
    operands = ()
    if restriction is None:
        intrinsic = (
            f"llvm.nvvm.fence.proxy.{proxy_suffix}.{order_suffix}.{scope_suffix}"
        )
    elif not isinstance(restriction, fence_stub._FenceRestrictionKind):
        intrinsic = (
            f"llvm.nvvm.fence.proxy.{proxy_suffix}.{order_suffix}.{scope_suffix}"
        )
        operands = (
            restriction.get_field("address"),
            restriction.get_field("size"),
        )
    elif restriction is fence_stub._FenceRestrictionKind.MBARRIER_INIT:
        intrinsic = (
            f"llvm.nvvm.fence.proxy.{proxy_suffix}.{order_suffix}."
            f"op_restrict.mbarrier_init.scope.{scope_suffix}"
        )
    else:
        space_suffix = _RESTRICTION_SPACE_SUFFIX[restriction]
        intrinsic = (
            f"llvm.nvvm.fence.proxy.{proxy_suffix}.{order_suffix}."
            f"sync_restrict.space.{space_suffix}.scope.{scope_suffix}"
        )

    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=operands,
    )


@impl(fence_stub.fence)
def fence_impl(order, scope, from_proxy, to_proxy, restriction) -> None:
    order = require_fence_order(order)
    scope = require_fence_scope(scope)
    from_proxy = require_constant_enum(from_proxy, FenceProxy)
    to_proxy = require_constant_enum(to_proxy, FenceProxy)
    restriction = require_fence_restriction(restriction)

    if from_proxy is FenceProxy.GENERIC and to_proxy is FenceProxy.GENERIC:
        lower_non_proxy_fence(order, scope, restriction)
    else:
        lower_proxy_fence(order, scope, from_proxy, to_proxy, restriction)


@impl(fence_stub.fence_proxy_bidirectional)
def fence_proxy_bidirectional_impl(proxy, restriction) -> None:
    proxy = require_constant_enum(proxy, FenceProxy)
    restriction = require_fence_restriction(restriction)

    operands = ()
    if restriction is None:
        proxy_suffix = proxy.value
    elif isinstance(restriction, fence_stub._FenceRestrictionKind):
        restriction_suffix = _BIDIRECTIONAL_RESTRICTION_SUFFIX[restriction]
        proxy_suffix = f"{proxy.value}.{restriction_suffix}"
    else:
        proxy_suffix = f"{proxy.value}.address"
        operands = (
            restriction.get_field("address"),
            restriction.get_field("size"),
        )

    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=f"llvm.nvvm.fence.proxy.{proxy_suffix}",
        operands_=operands,
    )
