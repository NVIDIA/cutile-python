# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._execution import simt_stub

from .._enums import CachePolicy, PrefetchLevel


@simt_stub
def prefetch(
    address,
    /,
    *,
    level: PrefetchLevel,
    eviction_priority: CachePolicy | None = None,
) -> None:
    """Prefetch the cache line containing ``address``.

    Args:
        address: Global, local, or generic pointer to the cache line.
        level: Cache level into which the line is prefetched.
        eviction_priority: Optional L2 eviction priority. Only
            :attr:`CachePolicy.L2_EVICT_NORMAL` and
            :attr:`CachePolicy.L2_EVICT_LAST` are supported.
    """


@simt_stub
def prefetch_uniform(address, /) -> None:
    """Prefetch the cache line containing a generic address into uniform L1.

    Args:
        address: Address contained by the cache line to be prefetched.
    """


@simt_stub
def prefetch_tensor_map(tensor_map, /) -> None:
    """Prefetch a tensor-map descriptor.

    Args:
        tensor_map: Tensor-map descriptor to prefetch.
    """


__all__ = (
    "PrefetchLevel",
    "prefetch",
    "prefetch_uniform",
    "prefetch_tensor_map",
)
