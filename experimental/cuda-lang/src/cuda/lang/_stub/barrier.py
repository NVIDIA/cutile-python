# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from cuda.lang._execution import function, simt_stub
from .._enums import BarrierReductionKind, MemoryOrder
from .core_api import FULL_MASK
from . import nvvm as _nvvm


@simt_stub
def barrier_sync_block(
    number_of_threads: int | None = None,
    barrier_id: int = 0,
) -> None:
    """Arrive at a named block barrier and wait for its completion.

    Participating warps may reach the barrier through different call sites.

    Args:
        number_of_threads (int | None): Expected thread arrivals. An explicit
            count must be positive and a multiple of 32. ``None`` includes all
            non-exited threads in the block. Arrivals are counted by warp.
        barrier_id (int): Named barrier ID, from 0 through 15. Reserve it for
            one active protocol. The ``syncthreads`` family uses barrier zero.

    :Participation: All non-exited lanes of each participating warp, with the same
        barrier ID and thread count.
    :Scope: Block.
    :Completion: Synchronous; resets after the expected arrivals.
    :Ordering: When this call returns, participants' loads made before
        their arrivals have finished, and their earlier stores are
        visible to the caller.
    """


@simt_stub
def barrier_sync_block_aligned(
    number_of_threads: int | None = None,
    barrier_id: int = 0,
) -> None:
    """Arrive and wait at a named block barrier with block-uniform control flow.

    Args:
        number_of_threads (int | None): Expected thread arrivals. An explicit
            count must be positive and a multiple of 32. ``None`` includes all
            non-exited threads in the block. Arrivals are counted by warp.
        barrier_id (int): Named barrier ID, from 0 through 15. Reserve it for
            one active protocol. The ``syncthreads`` family uses barrier zero.

    :Participation: :ref:`Block aligned <api-participation>`.
    :Scope: Block.
    :Completion: Synchronous; resets after the expected arrivals.
    :Ordering: When this call returns, participants' loads made before
        their arrivals have finished, and their earlier stores are
        visible to the caller.
    """


@function()
def barrier_arrive_block(
    number_of_threads: int,
    barrier_id: int = 0,
) -> None:
    """Arrive at a named block barrier and continue executing this warp.

    Args:
        number_of_threads (int): Expected thread arrivals. Must be positive
            and a multiple of 32. Arrivals are counted by warp.
        barrier_id (int): Named barrier ID, from 0 through 15. Reserve it for
            one active protocol. The ``syncthreads`` family uses barrier zero.
            Each warp contributes once per phase; complete that phase before
            the warp's next arrival.

    :Participation: All non-exited lanes of each participating warp, with the same
        barrier ID and thread count.
    :Scope: Block.
    :Completion: Arrival only; waits for its own warp. Consumer warps call
        :func:`barrier_sync_block` with the same ID and thread count.
    :Ordering: After the matching :func:`barrier_sync_block` returns,
        this warp's stores made before this call are visible to
        consumers, and its earlier loads have completed.
    """
    _nvvm.barrier_cta_arrive_count(barrier_id, number_of_threads)


@function()
def barrier_arrive_block_aligned(
    number_of_threads: int,
    barrier_id: int = 0,
) -> None:
    """Arrive and continue at a named block barrier with block-uniform control flow.

    All block threads execute this call site. Different warps may select
    different barrier IDs to signal their own barrier before waiting on a
    peer's. For a producer/consumer split where only producer warps call
    arrive, use :func:`barrier_arrive_block`.

    Args:
        number_of_threads (int): Expected thread arrivals. Must be positive
            and a multiple of 32. Arrivals are counted by warp.
        barrier_id (int): Named barrier ID, from 0 through 15. Reserve it for
            one active protocol. The ``syncthreads`` family uses barrier zero.
            Each warp contributes once per phase; complete that phase before
            the warp's next arrival.

    :Participation: :ref:`Block aligned <api-participation>`.
    :Scope: Block.
    :Completion: Arrival only; waits for its own warp.
    :Ordering: After the matching :func:`barrier_sync_block` returns,
        this warp's stores made before this call are visible to
        consumers, and its earlier loads have completed.
    """
    _nvvm.barrier_cta_arrive_aligned_count(barrier_id, number_of_threads)


@simt_stub
def barrier_reduce_block(
    op: BarrierReductionKind,
    predicate: bool,
    number_of_threads: int | None = None,
    barrier_id: int = 0,
) -> int | bool:
    """Synchronize at a named block barrier and reduce thread predicates.

    Args:
        op (BarrierReductionKind): Reduction applied to the predicates.
        predicate (bool): This thread's predicate.
        number_of_threads (int | None): Expected thread arrivals. An explicit
            count must be positive and a multiple of 32. ``None`` includes all
            non-exited threads in the block. Arrivals are counted by warp.
        barrier_id (int): Named barrier ID, from 0 through 15. Reserve it for
            one active protocol. The ``syncthreads`` family uses barrier zero.
            Use a dedicated ID for the reduction until it completes.

    Returns:
        An ``int`` count for ``POP_COUNT``, or a ``bool`` for ``AND`` and ``OR``.

    :Participation: All non-exited lanes of each participating warp, with the same
        ``op``, barrier ID, and thread count.
    :Scope: Block.
    :Completion: Synchronous.
    :Ordering: When this call returns, participants' loads made before
        their arrivals have finished, and their earlier stores are
        visible to the caller.
    """


@simt_stub
def barrier_reduce_block_aligned(
    op: BarrierReductionKind,
    predicate: bool,
    number_of_threads: int | None = None,
    barrier_id: int = 0,
) -> int | bool:
    """Synchronize at a named block barrier and reduce thread predicates.

    Args:
        op (BarrierReductionKind): Reduction applied to the predicates.
        predicate (bool): This thread's predicate.
        number_of_threads (int | None): Expected thread arrivals. An explicit
            count must be positive and a multiple of 32. ``None`` includes all
            non-exited threads in the block. Arrivals are counted by warp.
        barrier_id (int): Named barrier ID, from 0 through 15. Reserve it for
            one active protocol. The ``syncthreads`` family uses barrier zero.
            Use a dedicated ID for the reduction until it completes.

    Returns:
        An ``int`` count for ``POP_COUNT``, or a ``bool`` for ``AND`` and ``OR``.

    :Participation: :ref:`Block aligned <api-participation>`. Use the same
        ``op``, barrier ID, and count.
    :Scope: Block.
    :Completion: Synchronous.
    :Ordering: When this call returns, participants' loads made before
        their arrivals have finished, and their earlier stores are
        visible to the caller.
    """


@simt_stub
def barrier_arrive_cluster(
    *,
    memory_order: Literal[
        MemoryOrder.RELEASE, MemoryOrder.RELAXED
    ] = MemoryOrder.RELEASE,
) -> None:
    """Record an arrival at the cluster barrier.

    Args:
        memory_order (MemoryOrder): ``RELEASE`` orders prior ordinary
            accesses before matching acquire waits. ``RELAXED`` performs
            arrival accounting.

    :Target: SM90 or newer.
    :Participation: All non-exited cluster threads, once per phase.
    :Scope: Cluster.
    :Completion: Arrival only; waits for its own warp. Observe cluster completion
        with :func:`barrier_wait_cluster`.
    :Ordering: The selected memory order applies to accesses before arrival.
        The current phase must complete before the next arrival.
    """


@simt_stub
def barrier_arrive_cluster_aligned(
    *,
    memory_order: Literal[
        MemoryOrder.RELEASE, MemoryOrder.RELAXED
    ] = MemoryOrder.RELEASE,
) -> None:
    """Record an arrival at the cluster barrier.

    Args:
        memory_order (MemoryOrder): ``RELEASE`` orders prior ordinary
            accesses before matching acquire waits. ``RELAXED`` performs
            arrival accounting.

    :Target: SM90 or newer.
    :Participation: :ref:`Warp aligned <api-participation>`. All non-exited cluster threads
        arrive once per phase.
    :Scope: Cluster.
    :Completion: Arrival only; waits for its own warp. Observe cluster completion
        with :func:`barrier_wait_cluster`.
    :Ordering: The selected memory order applies to accesses before arrival.
        The current phase must complete before the next arrival.
    """


@function()
def barrier_wait_cluster() -> None:
    """Wait for the current cluster-barrier phase to complete.

    :Target: SM90 or newer.
    :Participation: All non-exited lanes of each calling warp, after their matching
        :func:`barrier_arrive_cluster`.
    :Scope: Cluster.
    :Completion: Synchronous; waits for all non-exited cluster threads to arrive.
    :Ordering: Acquire; observes ordinary accesses published by release arrivals.
    """
    _nvvm.barrier_cluster_wait()


@function()
def barrier_wait_cluster_aligned() -> None:
    """Wait for the current cluster-barrier phase to complete.

    :Target: SM90 or newer.
    :Participation: :ref:`Warp aligned <api-participation>`. Each caller must first call
        :func:`barrier_arrive_cluster` for this phase.
    :Scope: Cluster.
    :Completion: Synchronous; waits for all non-exited cluster threads to arrive.
    :Ordering: Acquire; observes ordinary accesses published by release arrivals.
    """
    _nvvm.barrier_cluster_wait_aligned()


@function()
def barrier_sync_cluster() -> None:
    """Arrive at and wait for the cluster barrier.

    :Target: SM90 or newer.
    :Participation: All non-exited cluster threads.
    :Scope: Cluster.
    :Completion: Synchronous; waits for all non-exited cluster threads to arrive.
    :Ordering: Release/acquire ordering for ordinary accesses across the barrier.
    """
    barrier_arrive_cluster()
    barrier_wait_cluster()


@function()
def barrier_sync_cluster_aligned() -> None:
    """Arrive at and wait for the cluster barrier.

    :Target: SM90 or newer.
    :Participation: :ref:`Warp aligned <api-participation>`. All non-exited
        cluster threads participate.
    :Scope: Cluster.
    :Completion: Synchronous; waits for all non-exited cluster threads to arrive.
    :Ordering: Release/acquire ordering for ordinary accesses across the barrier.
    """
    barrier_arrive_cluster_aligned()
    barrier_wait_cluster_aligned()


@function()
def barrier_sync_warp(mask: int = FULL_MASK) -> None:
    """Synchronize the warp lanes selected by ``mask``.

    Args:
        mask (int): Participation mask; bit ``i`` selects lane ``i``.

    :Participation: :ref:`Masked warp <api-participation>` (``mask``).
    :Scope: Warp.
    :Completion: Synchronous.
    :Ordering: Orders memory accesses among the selected lanes.
    """
    _nvvm.bar_warp_sync(mask)


@function()
def syncthreads() -> None:
    """Synchronize the block.

    :Participation: :ref:`Block aligned <api-participation>`.
    :Scope: Block; named barrier zero.
    :Completion: Synchronous.
    :Ordering: When this call returns, participants' loads made before
        their arrivals have finished, and their earlier stores are
        visible to the caller.
    """
    barrier_sync_block_aligned()


@function()
def syncwarp(mask: int = FULL_MASK) -> None:
    """Synchronize the warp lanes selected by ``mask``.

    Args:
        mask (int): Participation mask; bit ``i`` selects lane ``i``.

    :Participation: :ref:`Masked warp <api-participation>` (``mask``).
    :Scope: Warp.
    :Completion: Synchronous.
    :Ordering: Orders memory accesses among the selected lanes.
    """
    barrier_sync_warp(mask)


@function()
def syncthreads_count(predicate: bool) -> int:
    """Synchronize the block and count true predicates.

    Args:
        predicate (bool): This thread's predicate.

    Returns:
        Number of participating threads whose predicate is true.

    :Participation: :ref:`Block aligned <api-participation>`.
    :Scope: Block; named barrier zero.
    :Completion: Synchronous.
    :Ordering: When this call returns, participants' loads made before
        their arrivals have finished, and their earlier stores are
        visible to the caller.
    """
    return barrier_reduce_block_aligned(BarrierReductionKind.POP_COUNT, predicate)


@function()
def syncthreads_and(predicate: bool) -> bool:
    """Synchronize the block and test whether all predicates are true.

    Args:
        predicate (bool): This thread's predicate.

    Returns:
        True if every participating thread's predicate is true.

    :Participation: :ref:`Block aligned <api-participation>`.
    :Scope: Block; named barrier zero.
    :Completion: Synchronous.
    :Ordering: When this call returns, participants' loads made before
        their arrivals have finished, and their earlier stores are
        visible to the caller.
    """
    return barrier_reduce_block_aligned(BarrierReductionKind.AND, predicate)


@function()
def syncthreads_or(predicate: bool) -> bool:
    """Synchronize the block and test whether any predicate is true.

    Args:
        predicate (bool): This thread's predicate.

    Returns:
        True if any participating thread's predicate is true.

    :Participation: :ref:`Block aligned <api-participation>`.
    :Scope: Block; named barrier zero.
    :Completion: Synchronous.
    :Ordering: When this call returns, participants' loads made before
        their arrivals have finished, and their earlier stores are
        visible to the caller.
    """
    return barrier_reduce_block_aligned(BarrierReductionKind.OR, predicate)


__all__ = (
    "BarrierReductionKind",
    "barrier_sync_warp",
    "barrier_sync_block",
    "barrier_sync_block_aligned",
    "barrier_arrive_block",
    "barrier_arrive_block_aligned",
    "barrier_reduce_block",
    "barrier_reduce_block_aligned",
    "barrier_arrive_cluster",
    "barrier_arrive_cluster_aligned",
    "barrier_wait_cluster",
    "barrier_wait_cluster_aligned",
    "barrier_sync_cluster",
    "barrier_sync_cluster_aligned",
    "syncthreads",
    "syncwarp",
    "syncthreads_count",
    "syncthreads_and",
    "syncthreads_or",
)
