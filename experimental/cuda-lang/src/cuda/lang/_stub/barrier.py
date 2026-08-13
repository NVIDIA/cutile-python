# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import cuda.lang as cl
from cuda.lang._execution import function, stub
from .._enums import BarrierReductionKind, MemoryOrder
from .core_api import FULL_MASK
from . import nvvm as _nvvm
from .static_requirements import require_constant_bool, require_constant_enum


@function()
def barrier_sync_block(
    number_of_threads: int | None = None,
    barrier_id: int = 0,
    *,
    aligned: bool = True,
) -> None:
    """Synchronize threads participating in a named block barrier.

    Args:
        number_of_threads: Specifies the number of threads participating in the
           barrier. When specified, the value must be a multiple of the warp size.
           If not specified, all threads in the CTA participate in the barrier.
        barrier_id: Specifies a logical barrier resource with value 0 through
            15. Each CTA instance has sixteen barriers numbered 0..15.
        aligned: Requires every thread in the block to reach this same barrier
             instruction, otherwise the behavior is undefined.
    """
    require_constant_bool(aligned)
    if number_of_threads is None:
        if aligned:
            _nvvm.barrier_cta_sync_aligned_all(barrier_id)
        else:
            _nvvm.barrier_cta_sync_all(barrier_id)
    else:
        if aligned:
            _nvvm.barrier_cta_sync_aligned_count(barrier_id, number_of_threads)
        else:
            _nvvm.barrier_cta_sync_count(barrier_id, number_of_threads)


@function()
def barrier_arrive_block(
    number_of_threads: int,
    barrier_id: int = 0,
    *,
    aligned: bool = True,
) -> None:
    """Arrive at a named block barrier without waiting for other warps.

    Args:
        number_of_threads: Specifies the number of threads participating in the
           barrier. When specified, the value must be a multiple of the warp size.
           If not specified, all threads in the CTA participate in the barrier.
        barrier_id: Specifies a logical barrier resource with value 0 through
            15. Each CTA instance has sixteen barriers numbered 0..15.
        aligned: Requires every thread in the block to reach this same barrier
             instruction, otherwise the behavior is undefined.
    """
    require_constant_bool(aligned)
    if aligned:
        _nvvm.barrier_cta_arrive_aligned_count(barrier_id, number_of_threads)
    else:
        _nvvm.barrier_cta_arrive_count(barrier_id, number_of_threads)


@stub
def barrier_reduce_block(
    op: BarrierReductionKind,
    predicate: bool,
    number_of_threads: int | None = None,
    barrier_id: int = 0,
    *,
    aligned: bool = True,
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
        aligned: Requires every thread in the block to reach this same barrier
             instruction, otherwise the behavior is undefined.
    """


def barrier_arrive_cluster(
    *,
    aligned: bool = True,
    memory_order: Literal[
        MemoryOrder.RELEASE, MemoryOrder.RELAXED
    ] = MemoryOrder.RELEASE,
) -> None:
    """Arrive at the current thread-block-cluster barrier without waiting.

    Args:
        aligned: Requires every thread in the block to reach this same barrier
            instruction, otherwise the behavior is undefined.
        memory_order:
    """
    require_constant_bool(aligned)
    require_constant_enum(memory_order, MemoryOrder)
    cl.static_assert(
        memory_order in (MemoryOrder.RELEASE, MemoryOrder.RELAXED),
        "barrier_arrive_cluster memory_order must be "
        "MemoryOrder.RELEASE or MemoryOrder.RELAXED",
    )
    if memory_order == MemoryOrder.RELAXED:
        if aligned:
            _nvvm.barrier_cluster_arrive_relaxed_aligned()
        else:
            _nvvm.barrier_cluster_arrive_relaxed()
    else:
        if aligned:
            _nvvm.barrier_cluster_arrive_aligned()
        else:
            _nvvm.barrier_cluster_arrive()


@function()
def barrier_wait_cluster(*, aligned: bool = True) -> None:
    """Wait for completion of the current thread-block-cluster barrier.

    Args:
        aligned: Requires every thread in the block to reach this same barrier
            instruction, otherwise the behavior is undefined.
    """
    require_constant_bool(aligned)
    if aligned:
        _nvvm.barrier_cluster_wait_aligned()
    else:
        _nvvm.barrier_cluster_wait()


@function()
def barrier_sync_cluster(*, aligned: bool = True) -> None:
    """Arrive at and wait for the current thread-block-cluster barrier.

    Args:
        aligned: Requires every thread in the block to reach this same barrier
            instruction, otherwise the behavior is undefined.
    """
    barrier_arrive_cluster(aligned=aligned)
    barrier_wait_cluster(aligned=aligned)


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
    :func:`barrier_sync_block` with its default arguments.
    """
    barrier_sync_block()


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
    return barrier_reduce_block(BarrierReductionKind.POP_COUNT, predicate)


@function()
def syncthreads_and(predicate: bool) -> bool:
    """Synchronize the block and return whether ``predicate`` is true for all threads.

    CUDA C++ style convenience wrapper around
    :func:`barrier_reduce_block` with ``op=BarrierReductionKind.AND``

    Args:
        predicate: The per-thread predicate fed into the reduction.
    """
    return barrier_reduce_block(BarrierReductionKind.AND, predicate)


@function()
def syncthreads_or(predicate: bool) -> bool:
    """Synchronize the block and return whether ``predicate`` is true for any thread.

    CUDA C++ style convenience wrapper around
    :func:`barrier_reduce_block` with ``op=BarrierReductionKind.OR``

    Args:
        predicate: The per-thread predicate fed into the reduction.
    """
    return barrier_reduce_block(BarrierReductionKind.OR, predicate)


__all__ = (
    "BarrierReductionKind",
    "barrier_sync_warp",
    "barrier_sync_block",
    "barrier_arrive_block",
    "barrier_reduce_block",
    "barrier_arrive_cluster",
    "barrier_wait_cluster",
    "barrier_sync_cluster",
    "syncthreads",
    "syncwarp",
    "syncthreads_count",
    "syncthreads_and",
    "syncthreads_or",
)
