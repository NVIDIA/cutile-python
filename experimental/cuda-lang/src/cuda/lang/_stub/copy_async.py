# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._execution import simt_stub, function
from .._enums import TMALoadMode, TMAStoreMode, AsyncCopyReportMechanism
from . import nvvm as _nvvm


@simt_stub
def copy_async_mbarrier_arrive(mbar) -> None:
    """Signal an arrival after earlier ``cp.async`` copies complete.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.

    :Target: SM80 or newer.
    :Participation: :ref:`Per-thread <api-participation>`; tracks copies
        issued by the calling thread.
    :Completion: Asynchronous; observe the tracked copies with a successful
        mbarrier test or wait for this phase.
    :Accounting: Increments the pending arrival count by one before scheduling
        the completion arrival. The increment must fit the mbarrier's count
        limit; the completion arrival cancels it during this phase.
    """
    ...


@simt_stub
def copy_async_mbarrier_arrive_no_increment(mbar) -> None:
    """Signal an asynchronous-copy arrival without incrementing the count.

    Args:
        mbar: Pointer to an initialized, eight-byte-aligned ``cl.mbarrier``
            in block-local shared memory.

    :Target: SM80 or newer.
    :Participation: :ref:`Per-thread <api-participation>`; tracks copies
        issued by the calling thread.
    :Completion: Asynchronous; observe the tracked copies with a successful
        mbarrier test or wait for this phase.
    :Accounting: Include one expected arrival for each call when initializing
        the mbarrier. Copy completion decrements the pending arrival count
        by one.
    """
    ...


@simt_stub
def copy_async_bulk_tensor_global_to_shared(
    src_tensor_map_descriptor,
    src_coordinates,
    dst_memory,
    mbarrier,
    /,
    *,
    im2col_offsets=(),
    multicast_mask=None,
    l2_cache_hint=None,
    mode=TMALoadMode.TILE,
    cta_group=None,
    report_mechanism=AsyncCopyReportMechanism.DISABLED,
):
    """Initiate a multi-dimensional TMA copy from global to shared memory.

    See the `CUDA Programming Guide's multi-dimensional TMA alignment requirements
    <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#table-alignment-multi-dim-tma>`_
    for source and destination alignment requirements.

    See the `CUDA Driver API tensor-map documentation
    <https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html>`_
    for swizzle-mode and copy-direction compatibility.

    Args:
        src_tensor_map_descriptor: Pointer to a tensor-map descriptor.
        src_coordinates (tuple[int, ...]): 32-bit integer source coordinates
            in descriptor-axis order, measured in elements. For tiled copies,
            coordinate ``i`` addresses array axis ``order[i]`` of the
            :func:`tensor_map_tiled` descriptor.
        dst_memory: Pointer to the destination in shared or shared-cluster
            memory.
        mbarrier: Pointer to an initialized mbarrier in block-local shared
            memory used for the transaction.
        im2col_offsets (tuple[int, ...]): Integral offsets for image-to-column
            variants of this operation.
        multicast_mask (int | None): Mask selecting the destination block ranks
            within the cluster for multi-block operations.
        l2_cache_hint (int | None): Integer-encoded L2 cache policy.
        mode (TMALoadMode): :class:`TMALoadMode` selecting the load behavior.
        cta_group (CTAGroup | None): Selects the behavior of a multi-block
            async copy.
        report_mechanism (AsyncCopyReportMechanism): Specifies the mbarrier
            reporting mechanism.

    :Target: SM90 or newer for the basic tiled copy. Additional modes,
        CTA-group forms, and reporting mechanisms have instruction-specific
        target and toolchain requirements; consult the PTX ``cp.async.bulk.tensor``
        Target ISA Notes.
    :Memory: The shared destination must be 128-byte aligned. The mbarrier
        must be eight-byte aligned in block-local shared memory. The tensor-map
        descriptor must be 128-byte aligned, with a layout valid for this copy.
        Keep source data and the descriptor stable until transfer completion.
        For a ``MemorySpace.SHARED`` destination, ``predicate``,
        ``multicast_mask``, and ``cta_group`` must all be ``None``. Cluster
        forms require destination storage and initialized barriers in each
        recipient block.
    :Participation: :ref:`Single-thread issue <api-participation>` per transfer,
        including multicast transfers.
    :Completion: Asynchronous; observe with an acquire mbarrier wait after
        accounting for the expected bytes and arrivals. Finish consumers
        before reusing the destination buffer. See :doc:`/lang/tma`.
    """


@simt_stub
def copy_async_bulk_tensor_shared_to_global(
    src_memory,
    dst_tensor_map_descriptor,
    dst_coordinates,
    /,
    *,
    l2_cache_hint=None,
    mode=TMAStoreMode.TILE,
):
    """Initiate a multi-dimensional TMA copy from shared to global memory.

    See the `CUDA Programming Guide's multi-dimensional TMA alignment requirements
    <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#table-alignment-multi-dim-tma>`_
    for source and destination alignment requirements.

    See the `CUDA Driver API tensor-map documentation
    <https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html>`_
    for swizzle-mode and copy-direction compatibility.

    Args:
        src_memory: Pointer to source data in shared memory.
        dst_tensor_map_descriptor: Pointer to a tensor-map descriptor.
        dst_coordinates (tuple[int, ...]): Destination coordinates in
            descriptor-axis order, measured in elements. For tiled copies,
            coordinate ``i`` addresses array axis ``order[i]`` of the
            :func:`tensor_map_tiled` descriptor.
        l2_cache_hint (int | None): Integer-encoded L2 cache policy.
        mode (TMAStoreMode): :class:`TMAStoreMode` selecting the store
            behavior.
    :Target: SM90 or newer for the basic tiled copy; additional modes have
        instruction-specific target and toolchain requirements.
    :Memory: Source storage must be block-local shared memory with 128-byte
        alignment. The tensor-map descriptor must be 128-byte aligned and
        describe a valid global destination.
        Keep the source and descriptor stable until the copy finishes
        reading them. Each writer fences its shared-memory stores against
        the async proxy; coordinate writers before submission. See
        :ref:`lang-tma-proxy-fences`.
    :Participation: :ref:`Single-thread issue <api-participation>` per transfer.
    :Completion: Asynchronous; the issuer calls :func:`copy_async_bulk_commit_group`,
        then :func:`copy_async_bulk_wait_group`. ``read=True`` permits source
        reuse; a full wait also completes destination writes.
    """


@function()
def copy_async_bulk_commit_group():
    """
    Commit the calling thread's uncommitted bulk-group copies.

    :Target: SM90 or newer.
    :Participation: :ref:`Per-thread <api-participation>`; call from the thread
        that issued the copies.
    :Completion: Commits a group for :func:`copy_async_bulk_wait_group`.
    :Ordering: Copies within a group are unordered; their destinations
        must not overlap.
    """
    _nvvm.cp_async_bulk_commit_group()


@simt_stub
def copy_async_bulk_wait_group(number_of_groups, *, read=False):
    """Wait until at most ``number_of_groups`` committed bulk groups remain pending.

    Groups belong to the executing thread. A count of zero waits for all of
    that thread's previously committed bulk async-groups.

    Args:
        number_of_groups (const int): Maximum number of the most recent
            committed groups allowed to remain pending. Must be nonnegative.
        read (const bool): If True, wait only for reads of source locations and
            tensor-map descriptors, allowing source storage to be reused.
            If False, also wait for destination writes to complete and become
            visible to the executing thread. Default is False.

    :Target: SM90 or newer.
    :Participation: :ref:`Per-thread <api-participation>`; call from the thread
        that issued and committed the copies.
    :Completion: Waits for committed bulk groups as selected by
        ``number_of_groups`` and ``read``. Use mbarrier waits for
        global-to-shared copies tracked by an mbarrier.
    :Ordering: Synchronize other threads before they consume the writes
        or reuse source storage.
    """
