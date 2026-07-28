# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from cuda.lang._execution import stub
from .._enums import MemoryScope


@stub
def store_async_global(
    destination_address,
    value,
    /,
    *,
    scope: Literal[MemoryScope.DEVICE, MemoryScope.SYS] = MemoryScope.DEVICE,
    is_multimem: bool = False,
) -> None:
    """Start an asynchronous release store to global memory.

    Args:
        destination_address: Generic pointer to the destination in global
            memory.
        value: Scalar value to store. The value must have a size of 1, 2, 4,
            or 8 bytes.
        scope: Memory-ordering scope. Valid values are
            ``MemoryScope.DEVICE`` and ``MemoryScope.SYS``.
        is_multimem: Whether ``destination_address`` is a multimem address.
            A multimem address refers to a virtual multicast mapping.
    """
    ...


@stub
def store_async_cluster(
    destination_address,
    value,
    mbarrier,
    /,
) -> None:
    """Start a weak asynchronous store to cluster-shared memory.

    Args:
        destination_address: Pointer to the destination in
            shared-cluster memory.
        value: Scalar or vector value to store. A scalar must have a size of
            4 or 8 bytes. A vector must have two 4- or 8-byte elements, or
            four 4-byte elements.
        mbarrier: Pointer to an mbarrier in shared-cluster memory.
    """
    ...
