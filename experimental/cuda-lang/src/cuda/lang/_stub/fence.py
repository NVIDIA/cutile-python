# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import cuda.lang as cl
from cuda.lang._execution import function, stub
from .._enums import FenceProxyKind, MemoryOrder, MemoryScope, MemorySpace
from . import nvvm_mlir_interfaces as _mlir
from .static_requirements import require_constant_enum


@stub
def fence(
    order: Literal[
        MemoryOrder.ACQUIRE,
        MemoryOrder.RELEASE,
        MemoryOrder.ACQ_REL,
        MemoryOrder.SEQ_CST,
    ],
    scope: Literal[
        MemoryScope.BLOCK,
        MemoryScope.CLUSTER,
        MemoryScope.DEVICE,
        MemoryScope.SYS,
    ],
) -> None:
    """Issue a memory fence with the specified order and scope.

    Args:
        order: Memory-ordering semantics.
        scope: Memory-ordering scope.
    """
    ...


@function()
def fence_sc_cluster() -> None:
    """Issue a sequentially consistent memory fence at cluster scope."""
    _mlir.fence_sc_cluster()


@function()
def fence_mbarrier_initialize() -> None:
    """Release prior mbarrier initializations to the thread-block cluster."""
    _mlir.fence_mbarrier_init()


@function()
def fence_sync_restrict(
    order: Literal[MemoryOrder.ACQUIRE, MemoryOrder.RELEASE],
) -> None:
    """
    Uni-directional proxy fence operation with sync_restrict.

    Args:
        order: MemoryOrder.ACQUIRE or MemoryOrder.RELEASE
    """
    require_constant_enum(order, MemoryOrder)
    cl.static_assert(
        order in (MemoryOrder.ACQUIRE, MemoryOrder.RELEASE),
        "fence_sync_restrict order must be MemoryOrder.ACQUIRE or "
        "MemoryOrder.RELEASE",
    )
    _mlir.fence_sync_restrict(order=order)


@function()
def fence_proxy(
    kind: FenceProxyKind,
    *,
    space: MemorySpace | None = None,
) -> None:
    """
    Fence operation with proxy to establish an ordering between memory accesses
    that may happen through different proxies.

    Args:
        kind (FenceProxyKind): Proxy relationship.
        space (MemorySpace): Memory space restriction.
    """
    _mlir.fence_proxy(kind=kind, space=space)


@function()
def fence_proxy_acquire(
    address,
    size: int,
    *,
    scope: MemoryScope,
    from_proxy: FenceProxyKind = FenceProxyKind.GENERIC,
    to_proxy: FenceProxyKind = FenceProxyKind.TENSORMAP,
) -> None:
    """
    Uni-directional proxy fence operation with acquire semantics.

    Args:
        address: Pointer to the beginning of the affected memory range.
        size (int): Number of bytes in the range.
        scope (MemoryScope): Effective scope of the fence.
        from_proxy (FenceProxyKind):
        to_proxy (FenceProxyKind):
    """
    _mlir.fence_proxy_acquire(
        addr=address,
        size=size,
        scope=scope,
        from_proxy=from_proxy,
        to_proxy=to_proxy,
    )


@function()
def fence_proxy_release(
    *,
    scope: MemoryScope,
    from_proxy: FenceProxyKind = FenceProxyKind.GENERIC,
    to_proxy: FenceProxyKind = FenceProxyKind.TENSORMAP,
) -> None:
    """
    Uni-directional proxy fence operation with release semantics.

    Args:
        scope (MemoryScope): Effective scope of the fence.
        from_proxy (FenceProxyKind):
        to_proxy (FenceProxyKind):
    """
    _mlir.fence_proxy_release(
        scope=scope,
        from_proxy=from_proxy,
        to_proxy=to_proxy,
    )


@function()
def fence_proxy_sync_restrict(
    order: Literal[MemoryOrder.ACQUIRE, MemoryOrder.RELEASE],
    *,
    from_proxy: FenceProxyKind = FenceProxyKind.GENERIC,
    to_proxy: FenceProxyKind = FenceProxyKind.ASYNC,
) -> None:
    """
    Uni-directional proxy fence operation with sync_restrict.

    Args:
        order: MemoryOrder.ACQUIRE or MemoryOrder.RELEASE
        from_proxy (FenceProxyKind):
        to_proxy (FenceProxyKind):
    """
    require_constant_enum(order, MemoryOrder)
    cl.static_assert(
        order in (MemoryOrder.ACQUIRE, MemoryOrder.RELEASE),
        "fence_proxy_sync_restrict order must be MemoryOrder.ACQUIRE or "
        "MemoryOrder.RELEASE",
    )
    _mlir.fence_proxy_sync_restrict(
        order=order,
        from_proxy=from_proxy,
        to_proxy=to_proxy,
    )


__all__ = (
    "FenceProxyKind",
    "fence",
    "fence_sync_restrict",
    "fence_sc_cluster",
    "fence_mbarrier_initialize",
    "fence_proxy_sync_restrict",
    "fence_proxy",
    "fence_proxy_acquire",
    "fence_proxy_release",
)
