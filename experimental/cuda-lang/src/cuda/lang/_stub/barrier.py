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
    """Synchronize threads participating in a named block barrier.

    Args:
        number_of_threads: Specifies the number of threads participating in the
            barrier. When specified, the value must be a multiple of the warp size.
            If not specified, all threads in the CTA participate in the barrier.
        barrier_id: Specifies a logical barrier resource with value 0 through
            15. Each CTA instance has sixteen barriers numbered 0..15.
    """


@simt_stub
def barrier_sync_block_aligned(
    number_of_threads: int | None = None,
    barrier_id: int = 0,
) -> None:
    """Same as ``barrier_sync_block``, but must be textually aligned."""


@function()
def barrier_arrive_block(
    number_of_threads: int,
    barrier_id: int = 0,
) -> None:
    """Arrive at a named block barrier without waiting for other warps.

    Args:
        number_of_threads: Specifies the number of threads participating in the
            barrier. The value must be a multiple of the warp size.
        barrier_id: Specifies a logical barrier resource with value 0 through
            15. Each CTA instance has sixteen barriers numbered 0..15.
    """
    _nvvm.barrier_cta_arrive_count(barrier_id, number_of_threads)


@function()
def barrier_arrive_block_aligned(
    number_of_threads: int,
    barrier_id: int = 0,
) -> None:
    """Same as ``barrier_arrive_block``, but must be textually aligned."""
    _nvvm.barrier_cta_arrive_aligned_count(barrier_id, number_of_threads)


@simt_stub
def barrier_reduce_block(
    op: BarrierReductionKind,
    predicate: bool,
    number_of_threads: int | None = None,
    barrier_id: int = 0,
) -> int | bool:
    """Synchronize at a named block barrier and reduce a per-thread predicate.

    Args:
        op: The operation used to perform the reduction
        predicate: The per-thread predicate fed into the reduction.
        number_of_threads: Specifies the number of threads participating in the
           barrier. When specified, the value must be a multiple of the warp size.
           If not specified, all threads in the CTA participate in the barrier.
        barrier_id: Specifies a logical barrier resource with value 0 through
            15. Each CTA instance has sixteen barriers numbered 0..15.
    """


@simt_stub
def barrier_reduce_block_aligned(
    op: BarrierReductionKind,
    predicate: bool,
    number_of_threads: int | None = None,
    barrier_id: int = 0,
) -> int | bool:
    """Same as ``barrier_reduce_block``, but must be textually aligned."""


@simt_stub
def barrier_arrive_cluster(
    *,
    memory_order: Literal[
        MemoryOrder.RELEASE, MemoryOrder.RELAXED
    ] = MemoryOrder.RELEASE,
) -> None:
    """Arrive at the current thread-block-cluster barrier without waiting.

    Args:
        memory_order: The memory ordering applied to the barrier operation.
    """


@simt_stub
def barrier_arrive_cluster_aligned(
    *,
    memory_order: Literal[
        MemoryOrder.RELEASE, MemoryOrder.RELAXED
    ] = MemoryOrder.RELEASE,
) -> None:
    """Same as ``barrier_arrive_cluster``, but must be textually aligned."""


@function()
def barrier_wait_cluster() -> None:
    """Wait for completion of the current thread-block-cluster barrier."""
    _nvvm.barrier_cluster_wait()


@function()
def barrier_wait_cluster_aligned() -> None:
    """Same as ``barrier_wait_cluster``, but must be textually aligned."""
    _nvvm.barrier_cluster_wait_aligned()


@function()
def barrier_sync_cluster() -> None:
    """Arrive at and wait for the current thread-block-cluster barrier."""
    barrier_arrive_cluster()
    barrier_wait_cluster()


@function()
def barrier_sync_cluster_aligned() -> None:
    """Same as ``barrier_sync_cluster``, but must be textually aligned."""
    barrier_arrive_cluster_aligned()
    barrier_wait_cluster_aligned()


@function()
def barrier_sync_warp(mask: int = FULL_MASK) -> None:
    """Synchronize the warp lanes selected by ``mask``.

    Args:
        mask: Mask indicating membership where the ith bit selects lane i.
    """
    _nvvm.bar_warp_sync(mask)


@function()
def syncthreads() -> None:
    """Synchronize all threads in the current block.

    CUDA C++ style convenience wrapper around
    :func:`barrier_sync_block_aligned` with its default arguments.
    """
    barrier_sync_block_aligned()


@function()
def syncwarp(mask: int = FULL_MASK) -> None:
    """Synchronize the warp lanes selected by ``mask``.

    CUDA C++ style convenience wrapper around
    :func:`barrier_sync_warp`.

    Args:
        mask: Mask indicating membership where the ith bit selects lane i.
    """
    barrier_sync_warp(mask)


@function()
def syncthreads_count(predicate: bool) -> int:
    """Synchronize the block and count threads for which ``predicate`` is true.

    CUDA C++ style convenience wrapper around
    :func:`barrier_reduce_block` with ``op=BarrierReductionKind.POP_COUNT``

    Args:
        predicate: The per-thread predicate fed into the reduction.
    """
    return barrier_reduce_block_aligned(BarrierReductionKind.POP_COUNT, predicate)


@function()
def syncthreads_and(predicate: bool) -> bool:
    """Synchronize the block and return whether ``predicate`` is true for all threads.

    CUDA C++ style convenience wrapper around
    :func:`barrier_reduce_block` with ``op=BarrierReductionKind.AND``

    Args:
        predicate: The per-thread predicate fed into the reduction.
    """
    return barrier_reduce_block_aligned(BarrierReductionKind.AND, predicate)


@function()
def syncthreads_or(predicate: bool) -> bool:
    """Synchronize the block and return whether ``predicate`` is true for any thread.

    CUDA C++ style convenience wrapper around
    :func:`barrier_reduce_block` with ``op=BarrierReductionKind.OR``

    Args:
        predicate: The per-thread predicate fed into the reduction.
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
