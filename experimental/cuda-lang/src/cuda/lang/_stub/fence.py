# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum, auto
from cuda.lang._execution import stub
from .._enums import FenceProxy, MemoryOrder, MemoryScope


class FenceRestriction:
    """A restriction on the memory operations affected by a fence."""

    @staticmethod
    def mbarrier_initialize() -> "FenceRestriction":
        return _FenceRestrictionKind.MBARRIER_INIT

    @staticmethod
    def shared_block() -> "FenceRestriction":
        return _FenceRestrictionKind.SHARED_BLOCK

    @staticmethod
    def shared_cluster() -> "FenceRestriction":
        return _FenceRestrictionKind.SHARED_CLUSTER

    @staticmethod
    def global_memory() -> "FenceRestriction":
        return _FenceRestrictionKind.GLOBAL

    @staticmethod
    def address_range(address, size: int = 128) -> "FenceRestriction":
        return _FenceAddressRestriction(address, size)


class _FenceRestrictionKind(FenceRestriction, Enum):
    MBARRIER_INIT = auto()
    SHARED_BLOCK = auto()
    SHARED_CLUSTER = auto()
    GLOBAL = auto()


@dataclass(frozen=True)
class _FenceAddressRestriction(FenceRestriction):
    """Restrict a fence to a range of memory."""

    address: object
    size: int = 128


@stub
def fence(
    order: MemoryOrder = MemoryOrder.SEQ_CST,
    scope: MemoryScope = MemoryScope.SYS,
    *,
    from_proxy: FenceProxy = FenceProxy.GENERIC,
    to_proxy: FenceProxy = FenceProxy.GENERIC,
    restriction: FenceRestriction | None = None,
) -> None:
    """Issue a non-proxy fence or a split unidirectional proxy fence.

    A split proxy fence orders accesses from ``from_proxy`` to ``to_proxy``.

    Args:
        order: Memory-ordering semantics.
        scope: Threads that can observe the ordering effect.
        from_proxy: Proxy used by the earlier memory accesses.
        to_proxy: Proxy used by the later memory accesses.
        restriction: Restriction on the affected memory operations.
    """
    ...


@stub
def fence_proxy_bidirectional(
    proxy: FenceProxy,
    *,
    restriction: FenceRestriction | None = None,
) -> None:
    """Issue a bidirectional fence between the generic proxy and ``proxy``.

    Args:
        proxy: The non-generic memory access proxy.
        restriction: Restriction on the affected memory operations.
    """
    ...


__all__ = (
    "FenceProxy",
    "FenceRestriction",
    "fence",
    "fence_proxy_bidirectional",
)
