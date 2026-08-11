# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from .core_api import Pointer
from cuda.lang._datatype import (
    bool_,
    cluster_launch_control_token,
    mbarrier,
    int32,
)
from cuda.tile._memory_model import MemorySpace
from cuda.lang._execution import stub


@stub
def cluster_launch_control_try_cancel(
    addr: "Pointer[cluster_launch_control_token, MemorySpace.SHARED]",
    mbar: "Pointer[mbarrier, MemorySpace.SHARED]",
    multicast: bool = False,
) -> None:
    """Try to cancel a pending block and write the response token to ``addr``.

    Args:
        addr: Pointer to cancellation token in block-local shared memory.
        mbar: Pointer to initialized mbarrier. The current phase should expect
            a 16-byte transaction.
        multicast: Whether the response should be asynchronously written to the
            local shared memory of each block in the requesting cluster.
    """
    ...


@stub
def cluster_launch_control_is_canceled(token: cluster_launch_control_token) -> "bool_":
    """Return whether ``token`` represents a canceled block.

    Args:
        token: Response token.

    Returns:
        Indicates if an unlaunched cluster was canceled.
    """
    ...


@stub
def cluster_launch_control_get_first_block_index(
    token: cluster_launch_control_token, axis: int | None = None
) -> "int32 | tuple[int32, int32, int32]":
    """Return the first block index encoded in ``token``.

    Args:
        token: Response token for which :func:`cluster_launch_control_is_canceled`
            returned True.
        axis: Values 0, 1, or 2 select axis x, y, or z. None returns the tuple
            of all three.

    Returns:
        The selected coordinate or coordinates given by ``axis``.
    """
    ...
