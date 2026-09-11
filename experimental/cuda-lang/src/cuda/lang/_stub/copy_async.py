# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._execution import stub, function
from .._enums import TMALoadMode, TMAStoreMode, CTAGroup  # noqa: F401
from . import nvvm as _nvvm


@stub
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
    predicate=None,
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
        src_coordinates (tuple[int, ...]): 32-bit integer source coordinates.
        dst_memory: Pointer to the destination in shared or shared-cluster
            memory.
        mbarrier: Pointer to mbarrier used for the transaction.
        im2col_offsets (tuple[int, ...]): Integral offsets for image-to-column
            variants of this operation.
        multicast_mask (int | None): Mask selecting the destination block ranks
            within the cluster for multi-block operations.
        l2_cache_hint (int | None): Integer-encoded L2 cache policy.
        mode (TMALoadMode): :class:`TMALoadMode` selecting the load behavior.
        cta_group (CTAGroup | None): Selects the behavior of a multi-block
            async copy.
        predicate (bool | None):
    """


@stub
def copy_async_bulk_tensor_shared_to_global(
    src_memory,
    dst_tensor_map_descriptor,
    dst_coordinates,
    /,
    *,
    l2_cache_hint=None,
    mode=TMAStoreMode.TILE,
    predicate=None,
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
        dst_coordinates (tuple[int, ...]): Destination coordinates.
        l2_cache_hint (int | None): Integer-encoded L2 cache policy.
        mode (TMAStoreMode): :class:`TMAStoreMode` selecting the store
            behavior.
        predicate (bool | None):
    """


@function()
def copy_async_bulk_commit_group():
    """
    Commit all prior initiated but uncommitted cp.async.bulk instructions into
    a group of cp.async.bulk instructions.
    """
    _nvvm.cp_async_bulk_commit_group()


@stub()
def copy_async_bulk_wait_group(number_of_groups, *, read=False):
    """Wait for completion of the most recent bulk async-groups.

    Args:
        number_of_groups (32-bit integer): How many of the prior async-bulk
            operation groups should be waited on.
        read (bool): Indicates that executing thread should wait until the bulk
            async operations in the specified bulk async-group must complete
            reading from the tensor map and reading from their source
            locations.
    """
