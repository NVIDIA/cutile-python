# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from cuda.lang._enums import MbarrierLayout, MbarrierScope
from cuda.lang._execution import simt_stub
from cuda.lang._datatype import uint32, uint64, bool_
from cuda.lang._enums import MemoryOrder


ArriveMemoryOrder = Literal[MemoryOrder.RELAXED, MemoryOrder.RELEASE]
WaitMemoryOrder = Literal[MemoryOrder.RELAXED, MemoryOrder.ACQUIRE]


@simt_stub
def mbarrier_initialize(
    mbar, participants: int, *, layout: MbarrierLayout | None = None
) -> None:
    """Initialize an mbarrier with the expected participant count and layout.

    Args:
        mbar: Pointer to eight bytes of block-local shared storage for one
            ``cl.mbarrier``, with eight-byte alignment.
        participants: Initial expected arrival count for each phase. Must be
            between 1 and 1,048,575 for version 0, or 1 and 511 for version 1.
        layout (MbarrierLayout | None): Storage layout. Explicit layouts require SM90 or newer.

    :Memory: Invalidate an existing object with :func:`mbarrier_invalidate`
        before initializing that storage again.
    :Participation: :ref:`Single-thread issue <api-participation>` per mbarrier.
    :Completion: Synchronous; initializes phase zero.
    :Ordering: Publish initialization with a block barrier before other
        threads use the object.
    """
    ...


@simt_stub
def mbarrier_has_layout(mbar, layout: MbarrierLayout) -> "bool_":
    """Test whether an initialized mbarrier has the specified storage layout.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.
        layout (MbarrierLayout): Storage layout to check for.

    Returns:
        True if the mbarrier has the specified layout.

    :Target: SM90 or newer, with a toolchain supporting ``mbarrier.check_layout``.
    :Participation: :ref:`Per-thread <api-participation>`.
    """
    ...


@simt_stub
def mbarrier_invalidate(mbar) -> None:
    """Invalidate an mbarrier object before its storage is reused.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.

    :Participation: :ref:`Single-thread issue <api-participation>` per mbarrier, after all threads
        and asynchronous operations finish using it.
    :Completion: Synchronous; storage can be repurposed after invalidation.
        Reuse across ordinary phases keeps the object initialized.
    """
    ...


@simt_stub
def mbarrier_arrive(
    mbar,
    count: int = 1,
    *,
    drop: bool = False,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: ArriveMemoryOrder = MemoryOrder.RELEASE,
) -> "uint64 | None":
    """Arrive at ``mbar``. When the mbarrier resides in ``MemorySpace.SHARED``,
    an opaque 64-bit value capturing the phase of the mbarrier object *prior*
    to this arrive operation is returned. ``drop=True`` drops a participant
    from the barrier.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in ``MemorySpace.SHARED`` or ``MemorySpace.SHARED_CLUSTER``.
        count: Positive amount subtracted from pending arrivals, at most the
            pending arrival count. A runtime value or a value greater than one
            requires SM90 or newer.
        drop (const bool): Whether to also subtract ``count`` from subsequent phases'
            expected arrival count, which must remain positive.
        scope (MbarrierScope): Visibility scope.
        memory_order (MemoryOrder): ``MemoryOrder.RELEASE`` or
            ``MemoryOrder.RELAXED``. Release orders prior memory accesses;
            relaxed provides arrival accounting only.

    Returns:
        On a block-local barrier, returns an opaque token.
        On a cluster barrier, returns ``None``.

    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Arrival only; observe phase completion with a matching
        mbarrier test or wait.
    :Ordering: Before any next-phase arrival, at least one test or wait must
        observe completion of the preceding phase.
    """


@simt_stub
def mbarrier_arrive_nocomplete(mbar, count: int) -> "uint64":
    """Arrive without completing the current phase.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.
        count: Amount subtracted from pending arrivals. Must be positive and
            strictly less than the pending count, keeping the phase incomplete.

    Returns:
        Opaque 64-bit token for the phase before the arrival.

    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Arrival only; returns a token for a later wait.
    :Ordering: Before any next-phase arrival, at least one test or wait must
        observe completion of the preceding phase.
    """
    ...


@simt_stub
def mbarrier_arrive_drop_nocomplete(mbar, count: int) -> "uint64":
    """Arrive without completing the phase and drop expected arrivals.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.
        count: Positive amount subtracted from pending arrivals and subsequent
            phases' expected arrivals. Must be less than the pending arrival
            count and leave the expected arrival count positive.

    Returns:
        Opaque 64-bit token for the phase before the arrival.

    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Arrival only; returns a token for a later wait.
    :Ordering: Before any next-phase arrival, at least one test or wait must
        observe completion of the preceding phase.
    """
    ...


@simt_stub
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
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in ``MemorySpace.SHARED`` or ``MemorySpace.SHARED_CLUSTER``.
        bytes: Transaction bytes added to the current phase. Each call also
            subtracts one pending arrival. Account for each expected byte once
            through asynchronous completion or :func:`mbarrier_complete_transaction`.
        drop (const bool): Whether to also subtract one from subsequent phases' expected
            arrival count, which must remain positive.
        scope (MbarrierScope): Visibility scope.
        memory_order (MemoryOrder): ``MemoryOrder.RELEASE`` or
            ``MemoryOrder.RELAXED``. Release orders prior memory accesses;
            relaxed provides arrival accounting only.

    Returns:
        On a block-local barrier, returns an opaque token.
        On a cluster barrier, returns ``None``.

    :Target: SM90 or newer.
    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Arrival only; observe the phase with an mbarrier wait
        after all expected arrivals and transaction bytes are satisfied.
    :Ordering: Before any next-phase arrival, at least one test or wait must
        observe completion of the preceding phase.
    """
    ...


@simt_stub
def mbarrier_expect_transaction(
    mbar,
    bytes: int,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
) -> None:
    """Add expected transaction bytes to ``mbar`` without arriving.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in ``MemorySpace.SHARED`` or ``MemorySpace.SHARED_CLUSTER``.
        bytes: Transaction bytes added to the current phase. Register these
            while an arrival is still pending to prevent premature completion.
            Account for each byte once through asynchronous completion or
            :func:`mbarrier_complete_transaction`.
        scope (MbarrierScope): Visibility scope.

    :Target: SM90 or newer.
    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Accounting update only; observe the phase with an
        mbarrier wait.
    """
    ...


@simt_stub
def mbarrier_complete_transaction(
    mbar,
    bytes: int,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
) -> None:
    """Mark transaction bytes as complete for ``mbar``.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in ``MemorySpace.SHARED`` or ``MemorySpace.SHARED_CLUSTER``.
        bytes: Completed transaction bytes to subtract. The corresponding work
            must already be complete. TMA updates this count automatically;
            count each completion once.
        scope (MbarrierScope): Visibility scope.

    :Target: SM90 or newer.
    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Accounting update only; observe the phase with an
        mbarrier wait.
    """
    ...


@simt_stub
def mbarrier_pending_count(state) -> "uint32":
    """Query the pending arrival count from the opaque mbarrier state.

    Args:
        state: 64-bit token returned by ``mbarrier_arrive_drop_nocomplete``
            or ``mbarrier_arrive_nocomplete``.

    Returns:
        Pending arrival count before the arrival that produced ``state``.

    :Participation: :ref:`Per-thread <api-participation>`.
    """
    ...


@simt_stub
def mbarrier_test_wait(
    mbar,
    state,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Non-blocking test whether ``mbar`` has completed.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.
        state: Opaque arrival token from this mbarrier's current incomplete
            phase or immediately preceding completed phase.
        scope (MbarrierScope): Visibility scope.
        memory_order (MemoryOrder): ``MemoryOrder.ACQUIRE`` or
            ``MemoryOrder.RELAXED``. Acquire ordering applies when the
            completion test succeeds.

    Returns:
        True if the selected phase completed: its pending arrival and
        transaction counts both reached zero. Otherwise False.

    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Nonblocking test; retry until ``True`` to observe completion.
    :Ordering: A successful acquire test makes the completed phase's
        release-ordered writes visible within ``scope``.
    """


@simt_stub
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
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.
        parity: Parity (0 or 1) of the current incomplete phase or immediately
            preceding completed phase.
        scope (MbarrierScope): Visibility scope.
        memory_order (MemoryOrder): ``MemoryOrder.ACQUIRE`` or
            ``MemoryOrder.RELAXED``. Acquire ordering applies when the
            completion test succeeds.

    Returns:
        True if the selected phase completed: its pending arrival and
        transaction counts both reached zero. Otherwise False.

    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Nonblocking test; retry until ``True`` to observe completion.
    :Ordering: A successful acquire test makes the completed phase's
        release-ordered writes visible within ``scope``.
    """


@simt_stub
def mbarrier_try_wait(
    mbar,
    state,
    *,
    time_limit_hint_ns: int | None = None,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Bounded-wait test whether ``mbar`` has completed.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.
        state: Opaque arrival token from this mbarrier's current incomplete
            phase or immediately preceding completed phase.
        time_limit_hint_ns: Optional time limit hint in nanoseconds that may be
            used instead of the system-dependent limit.
        scope (MbarrierScope): Visibility scope.
        memory_order (MemoryOrder): ``MemoryOrder.ACQUIRE`` or
            ``MemoryOrder.RELAXED``. Acquire ordering applies when the
            completion test succeeds.

    Returns:
        True if the selected phase completed: its pending arrival and
        transaction counts both reached zero. Otherwise False.

    :Target: SM90 or newer.
    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: May resume with ``False``; retry until ``True`` to observe completion.
    :Ordering: A successful acquire test makes the completed phase's
        release-ordered writes visible within ``scope``.
    """


def mbarrier_wait(
    mbar,
    state,
    *,
    time_limit_hint_ns: int = 10_000,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
):
    """Wait for a phase's pending arrivals and transactions to reach zero.

    Repeats :func:`mbarrier_try_wait` until it succeeds. The time hint
    applies to each attempt.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.
        state: Opaque arrival token from this mbarrier's current incomplete
            phase or immediately preceding completed phase.
        time_hint: Nonnegative nanosecond time hint for each wait attempt.
        scope (MbarrierScope): Visibility scope.
        memory_order (MemoryOrder): ``MemoryOrder.ACQUIRE`` or ``MemoryOrder.RELAXED``.

    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Synchronous.
    :Ordering: Acquire makes the completed phase's release-ordered writes
        visible within ``scope``.
    """
    while not mbarrier_try_wait(
        mbar, state, time_limit_hint_ns=time_limit_hint_ns, scope=scope, memory_order=memory_order
    ):
        pass


@simt_stub
def mbarrier_try_wait_parity(
    mbar,
    parity: int,
    *,
    time_limit_hint_ns: int | None = None,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Phase-parity variant of ``mbarrier_try_wait``.
    ``parity`` is the 0/1 integer parity of the phase to test for.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.
        parity: Parity (0 or 1) of the current incomplete phase or immediately
            preceding completed phase.
        time_limit_hint_ns: Optional time limit hint in nanoseconds that may be
            used instead of the system-dependent limit.
        scope (MbarrierScope): Visibility scope.
        memory_order (MemoryOrder): ``MemoryOrder.ACQUIRE`` or
            ``MemoryOrder.RELAXED``. Acquire ordering applies when the
            completion test succeeds.

    Returns:
        True if the selected phase completed: its pending arrival and
        transaction counts both reached zero. Otherwise False.

    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: May resume with ``False``; retry until ``True`` to observe completion.
    :Ordering: A successful acquire test makes the completed phase's
        release-ordered writes visible within ``scope``.
    """


def mbarrier_wait_parity(
    mbar,
    parity: int,
    *,
    time_limit_hint_ns: int = 10_000,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
):
    """Wait for the selected phase's pending arrivals and transactions to reach zero.

    Repeats :func:`mbarrier_try_wait_parity` until it succeeds. The time
    hint applies to each attempt.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.
        parity: Parity (0 or 1) of the current incomplete phase or immediately
            preceding completed phase.
        time_hint: Nonnegative nanosecond time hint for each wait attempt.
        scope (MbarrierScope): Visibility scope.
        memory_order (MemoryOrder): ``MemoryOrder.ACQUIRE`` or ``MemoryOrder.RELAXED``.

    :Participation: :ref:`Per-thread <api-participation>`.
    :Completion: Synchronous.
    :Ordering: Acquire makes the completed phase's release-ordered writes
        visible within ``scope``.
    """
    while not mbarrier_try_wait_parity(
        mbar, parity, time_limit_hint_ns=time_limit_hint_ns, scope=scope, memory_order=memory_order
    ):
        pass


@simt_stub
def mbarrier_test_wait_report(
    mbar,
    state,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
) -> "tuple[bool_, bool_, uint32]":
    """Test primary-phase completion.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        state: Opaque token from an arrival on the same barrier.
        scope: Visibility scope.
        memory_order: Memory ordering.

    Returns:
        Completion, report predicate, and value. Reports are valid on completion.
    """
    ...


@simt_stub
def mbarrier_test_wait_report_parity(
    mbar,
    parity: int,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
) -> "tuple[bool_, bool_, uint32]":
    """Test primary-phase completion by parity.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        parity: Phase parity, either 0 or 1.
        scope: Visibility scope.
        memory_order: Memory ordering.

    Returns:
        Completion, report predicate, and value. Reports are valid on completion.
    """
    ...


@simt_stub
def mbarrier_try_wait_report(
    mbar,
    state,
    *,
    time_limit_hint_ns: int | None = None,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
) -> "tuple[bool_, bool_, uint32]":
    """Try waiting for primary-phase completion.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        state: Opaque token from an arrival on the same barrier.
        time_limit_hint_ns: Optional time limit hint in nanoseconds that may be
            used instead of the system-dependent limit.
        scope: Visibility scope.
        memory_order: Memory ordering.

    Returns:
        Completion, report predicate, and value. Reports are valid on completion.
    """
    ...


@simt_stub
def mbarrier_try_wait_report_parity(
    mbar,
    parity: int,
    *,
    time_limit_hint_ns: int | None = None,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
) -> "tuple[bool_, bool_, uint32]":
    """Try waiting for primary-phase completion by parity.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        parity: Phase parity, either 0 or 1.
        time_limit_hint_ns: Optional time limit hint in nanoseconds that may be
            used instead of the system-dependent limit.
        scope: Visibility scope.
        memory_order: Memory ordering.

    Returns:
        Completion, report predicate, and value. Reports are valid on completion.
    """
    ...


def mbarrier_wait_report(
    mbar,
    state,
    *,
    time_limit_hint_ns: int = 10_000,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
) -> "tuple[bool_, uint32]":
    """Wait for primary-phase completion.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        state: Opaque token from an arrival on the same barrier.
        time_limit_hint_ns: Optional time limit hint in nanoseconds that may be
            used instead of the system-dependent limit.
        scope: Visibility scope.
        memory_order: Memory ordering.

    Returns:
        Report predicate and value.
    """
    while True:
        complete, report_predicate, report_value = mbarrier_try_wait_report(
            mbar, state, time_limit_hint_ns=time_limit_hint_ns,
            scope=scope, memory_order=memory_order,
        )
        if complete:
            return report_predicate, report_value


def mbarrier_wait_report_parity(
    mbar,
    parity: int,
    *,
    time_limit_hint_ns: int = 10_000,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    memory_order: WaitMemoryOrder = MemoryOrder.ACQUIRE,
) -> "tuple[bool_, uint32]":
    """Wait for primary-phase completion by parity.

    Args:
        mbar: Pointer to mbarrier in block-local shared memory.
        parity: Phase parity, either 0 or 1.
        time_limit_hint_ns: Optional time limit hint in nanoseconds that may be
            used instead of the system-dependent limit.
        scope: Visibility scope.
        memory_order: Memory ordering.

    Returns:
        Report predicate and value.
    """
    while True:
        complete, report_predicate, report_value = mbarrier_try_wait_report_parity(
            mbar, parity, time_limit_hint_ns=time_limit_hint_ns,
            scope=scope, memory_order=memory_order,
        )
        if complete:
            return report_predicate, report_value
