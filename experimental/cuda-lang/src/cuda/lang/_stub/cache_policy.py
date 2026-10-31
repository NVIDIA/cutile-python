# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from .._enums import CachePolicy
from .._execution import simt_stub


@simt_stub
def create_range_cache_policy(
    base_address,
    primary_size,
    total_size,
    primary_policy,
    /,
    *,
    secondary_policy=CachePolicy.L2_EVICT_UNCHANGED,
):
    """Creates a cache eviction policy for the specified cache level for the
    specified address ranges given by the base address and the primary and
    total address range sizes.

    Args:
        base_address (pointer): Pointer to beginning of the primary range.
        primary_size (int): Extent of the primary range this hint applies to.
        total_size (int): Extent of the secondary range this hint applies to.
        primary_policy: Cache policy this hint requests be applied to the
            primary address range.
        secondary_policy: Cache policy this hint requests be applied to the
            secondary address range.

    Returns:
        Opaque encoded cache policy.

    See the PTX documentation for more information.
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy
    """


@simt_stub
def create_fractional_cache_policy(
    primary_policy,
    /,
    *,
    fraction=1.0,
    secondary_policy=CachePolicy.L2_EVICT_UNCHANGED,
):
    """Creates a cache eviction policy for the specified cache level for
    the specified fraction of cache accesses.

    Args:
        primary_policy: Cache policy applied ``fraction`` fraction of the cache
            accesses.
        fraction: Fraction of cache accesses this hint requests have
            ``primary_policy`` policy.
        secondary_policy: Cache policy this hint requests to be used on
            ``1 - fraction`` cache accesses.

    Returns:
        Opaque encoded cache policy.

    See the PTX documentation for more information.
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy
    """
