# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from cuda.lang._enums import MbarrierScope
from cuda.lang._execution import stub
from cuda.lang._datatype import uint64, bool_
from cuda.tile._memory_model import MemoryOrder


ArriveOrdering = Literal[MemoryOrder.RELAXED, MemoryOrder.RELEASE]
WaitOrdering = Literal[MemoryOrder.RELAXED, MemoryOrder.ACQUIRE]


@stub
def mbarrier_initialize(mbar, participants: int) -> None:
    """Initialize an mbarrier with the expected participant count.

    Args:
        mbar: Pointer to mbarrier in shared memory.
        participants: Initial expected arrival count for each phase.
    """
    ...


@stub
def mbarrier_invalidate(mbar) -> None:
    """Invalidate an mbarrier object before its storage is reused.

    Args:
        mbar: Pointer to mbarrier in shared memory.
    """
    ...


@stub
def mbarrier_arrive(
    mbar,
    count: int = 1,
    *,
    drop: bool = False,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: ArriveOrdering = MemoryOrder.RELEASE,
) -> "uint64 | None":
    """Arrive at ``mbar``. When the mbarrier resides in ``MemorySpace.SHARED``,
    an opaque 64-bit value capturing the phase of the mbarrier object _prior_
    to this arrive operation is returned. ``drop=True`` drops a participant
    from the barrier.

    Args:
        mbar: Pointer to mbarrier in shared memory.
        count: The amount by which the pending arrival count is decremented.
        drop: Whether to decrement the expected arrival count of the mbarrier.
        scope: Visibility scope.
        memory_order:

    Returns:
        On a block-local barrier, returns an opaque token.
        On a cluster barrier, returns ``None``.
    """


@stub
def mbarrier_arrive_expect_transaction(
    mbar,
    bytes: int,
    *,
    drop: bool = False,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: ArriveOrdering = MemoryOrder.RELEASE,
) -> "uint64 | None":
    """Arrive at ``mbar`` and add expected transaction bytes.

    Args:
        mbar: Pointer to mbarrier in shared memory.
        bytes: Transaction bytes added to the current phase.
        drop: Whether to decrement the expected arrival count of the mbarrier.
        scope: Visibility scope.
        memory_order:

    Returns:
        On a block-local barrier, returns an opaque token.
        On a cluster barrier, returns ``None``.
    """
    ...


@stub
def mbarrier_expect_transaction(
    mbar,
    bytes: int,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
) -> None:
    """Add expected transaction bytes to ``mbar`` without arriving.

    Args:
        mbar: Pointer to mbarrier in shared memory.
        bytes: Transaction bytes added to the current phase.
        scope: Visibility scope.
    """
    ...


@stub
def mbarrier_complete_transaction(
    mbar,
    bytes: int,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
) -> None:
    """Mark transaction bytes as complete for ``mbar``.

    Args:
        mbar: Pointer to mbarrier in shared memory.
        bytes: Number of completed transaction bytes to subtract.
        scope: Visibility scope.
    """
    ...


@stub
def mbarrier_test_wait(
    mbar,
    state,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitOrdering = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Non-blocking test whether ``mbar`` has completed.

    Args:
        mbar: Pointer to mbarrier in shared memory.
        state: Opaque token returned by an arrival on the same barrier.
        scope: Visibility scope.
        memory_order:

    Returns:
        Indicates whether the phase is complete.
    """


@stub
def mbarrier_test_wait_parity(
    mbar,
    parity: int,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitOrdering = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Phase-parity variant of ``mbarrier_test_wait``.
    ``parity`` is the 0/1 integer parity of the phase to test for.

    Args:
        mbar: Pointer to mbarrier in shared memory.
        parity: Phase parity, either 0 or 1.
        scope: Visibility scope.
        memory_order:

    Returns:
        Indicates whether the selected phase is complete.
    """


@stub
def mbarrier_try_wait(
    mbar,
    state,
    *,
    time_hint: int | None = None,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitOrdering = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Bounded-wait test whether ``mbar`` has completed.

    Args:
        mbar: Pointer to mbarrier in shared memory.
        state: Opaque token returned by an arrival on the same mbarrier.
        time_hint: Optional nonnegative ``int32`` time limit in nanoseconds.
        scope: Visibility scope.
        memory_order:

    Returns:
        Indicates whether the phase completed before the instruction resumed.
    """


def mbarrier_wait(*args, time_hint=10_000, **kwargs):
    """Synchronously wait for an mbarrier to complete.
    Accepts the same arguments as :func:``mbarrier_try_wait``.
    """
    while not mbarrier_try_wait(*args, time_hint=time_hint, **kwargs):
        pass


@stub
def mbarrier_try_wait_parity(
    mbar,
    parity: int,
    *,
    time_hint: int | None = None,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitOrdering = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Phase-parity variant of ``mbarrier_try_wait``.
    ``parity`` is the 0/1 integer parity of the phase to test for.

    Args:
        mbar: Pointer to mbarrier in shared memory.
        parity: Phase parity, either 0 or 1.
        time_hint: Optional nonnegative ``int32`` time limit in nanoseconds.
        scope: Visibility scope.
        memory_order:

    Returns:
        Indicates whether the phase completed before the instruction resumed.
    """


def mbarrier_wait_parity(*args, time_hint=10_000, **kwargs):
    """Synchronously wait for an mbarrier to complete for a given phase parity.
    Accepts the same arguments as :func:``mbarrier_try_wait_parity``.
    """
    while not mbarrier_try_wait_parity(*args, time_hint=time_hint, **kwargs):
        pass
