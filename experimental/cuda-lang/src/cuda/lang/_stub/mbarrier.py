# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from cuda.lang._enums import MbarrierLayout, MbarrierScope
from cuda.lang._execution import stub
from cuda.lang._datatype import uint32, uint64, bool_
from cuda.lang._enums import MemoryOrder


ArriveMemoryOrder = Literal[MemoryOrder.RELAXED, MemoryOrder.RELEASE]
WaitMemoryOrder = Literal[MemoryOrder.RELAXED, MemoryOrder.ACQUIRE]


@stub
def mbarrier_initialize(
    mbar, participants: int, *, layout: MbarrierLayout | None = None
) -> None:
    """Initialize an mbarrier with the expected participant count and layout.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        participants: Initial expected arrival count for each phase. Must be
            between 1 and 1,048,575 for version 0, or 1 and 511 for version 1.
        layout: Storage layout. Explicit layouts require SM90 or newer.
    """
    ...


@stub
def mbarrier_has_layout(mbar, layout: MbarrierLayout) -> "bool_":
    """Test whether an initialized mbarrier has the specified storage layout.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        layout: Storage layout to check for.

    Returns:
        True if the mbarrier has the specified layout.
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
    memory_order: ArriveMemoryOrder = MemoryOrder.RELEASE,
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
def mbarrier_arrive_nocomplete(mbar, count: int) -> "uint64":
    """Arrive without completing the current phase.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        count: Amount to subtract from the pending arrival count.

    Returns:
        Opaque 64-bit token for the phase before the arrival.
    """
    ...


@stub
def mbarrier_arrive_drop_nocomplete(mbar, count: int) -> "uint64":
    """Arrive without completing the phase and drop expected arrivals.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        count: Amount to subtract from the pending and expected arrival counts.

    Returns:
        Opaque 64-bit token for the phase before the arrival.
    """
    ...


@stub
def mbarrier_arrive_expect_transaction(
    mbar,
    bytes: int,
    *,
    drop: bool = False,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: ArriveMemoryOrder = MemoryOrder.RELEASE,
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
def mbarrier_pending_count(state) -> "uint32":
    """Query the pending arrival count from the opaque mbarrier state.

    Args:
        state: 64-bit token returned by ``mbarrier_arrive_drop_nocomplete``
            or ``mbarrier_arrive_nocomplete``.

    Returns:
        Pending arrival count before the arrival that produced ``state``.
    """
    ...


@stub
def mbarrier_test_wait(
    mbar,
    state,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
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
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
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
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
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
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
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
