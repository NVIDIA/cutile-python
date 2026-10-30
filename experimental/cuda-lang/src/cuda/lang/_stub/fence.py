# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum, auto
from cuda.lang._execution import simt_stub
from .._enums import FenceProxy, MemoryOrder, MemoryScope


class FenceRestriction:
    """A restriction on the memory operations affected by a fence."""

    @staticmethod
    def mbarrier_initialize() -> "FenceRestriction":
        """Restrict ordering to preceding mbarrier initialization operations."""
        return _FenceRestrictionKind.MBARRIER_INIT

    @staticmethod
    def shared_block() -> "FenceRestriction":
        """Restrict ordering to the current thread block's shared memory."""
        return _FenceRestrictionKind.SHARED_BLOCK

    @staticmethod
    def shared_cluster() -> "FenceRestriction":
        """Restrict ordering to the cluster shared-memory address space."""
        return _FenceRestrictionKind.SHARED_CLUSTER

    @staticmethod
    def global_memory() -> "FenceRestriction":
        """Restrict ordering to the global-memory address space."""
        return _FenceRestrictionKind.GLOBAL

    @staticmethod
    def address_range(address, size: int = 128) -> "FenceRestriction":
        """Restrict ordering to a range of global memory.

        Args:
            address: Pointer to the start of the range in global memory.
            size (const int): Range size in bytes. Must be 128.
        """
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


@simt_stub
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
        order (MemoryOrder): Memory-ordering semantics.
        scope (MemoryScope): Threads that can observe the ordering effect.
        from_proxy (FenceProxy): Proxy used by the earlier memory accesses.
        to_proxy (FenceProxy): Proxy used by the later memory accesses.
        restriction (FenceRestriction | None): Restriction on the affected memory
            operations. An address-range restriction contains a pointer to
            global memory and a size of 128 bytes. See
            :meth:`FenceRestriction.address_range`.

    :Target: Plain acquire-release or sequentially consistent fences require
        SM70 or newer. Split proxy fences, cluster scope, and restricted forms
        require SM90 or newer; specific proxy forms also depend on the
        toolchain's PTX support. Use the combinations defined by the `PTX fence
        reference
        <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar-fence>`_.
    :Participation: :ref:`Per-thread <api-participation>`.
    """
    ...


@simt_stub
def fence_proxy_bidirectional(
    proxy: FenceProxy,
    *,
    restriction: FenceRestriction | None = None,
) -> None:
    """Issue a bidirectional fence between the generic proxy and ``proxy``.

    Args:
        proxy (FenceProxy): The non-generic memory access proxy.
        restriction (FenceRestriction | None): Restriction on the affected memory
            operations. An address range contains a pointer to global memory
            and a size of 128 bytes. See :meth:`FenceRestriction.address_range`.

    :Target: The async-proxy form requires SM90 or newer. Other forms follow
        the `PTX proxy-fence requirements
        <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar-fence>`_.
    :Participation: :ref:`Per-thread <api-participation>`.
    :Ordering: For shared-to-global TMA, writers fence their stores before
        synchronizing and submitting the copy. See :ref:`lang-tma-proxy-fences`.
    """
    ...


__all__ = (
    "FenceProxy",
    "FenceRestriction",
    "fence",
    "fence_proxy_bidirectional",
)
