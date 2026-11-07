# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from .._enums import (
    MatrixStoreShape,
    MatrixLoadShape,
    MatrixLoadSourceFormat,
)
from cuda.lang._execution import simt_stub
from typing import Literal


@simt_stub
def load_matrix(
    src,
    /,
    *,
    shape: MatrixLoadShape,
    count: Literal[1, 2, 4] = 1,
    transpose: bool = False,
    source_format: MatrixLoadSourceFormat | None = None,
):
    """
    Collectively load one or more matrices from shared memory for mma instruction

    Args:
        src: Pointer to a shared-memory matrix row. Per-lane row addresses
            follow the PTX shape/count layout.
        shape (MatrixLoadShape): Shape and element size.
        count (const int): Number of matrices to load.
        transpose (const bool): False loads the matrix in row-major order, True loads in
            column-major order.
        source_format (MatrixLoadSourceFormat | None): Packed source format.

    Returns:
        Scalar or vector of 32-bit integers holding this lane's fragment.
        ``M16N16`` returns ``2 * count`` words; other shapes return ``count``.

    :Target: The basic ``M8N8`` form requires SM75 or newer. Additional shapes
        and source formats follow the `PTX ldmatrix Target ISA Notes
        <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix>`_.
    :Memory: Source rows must be in block-local shared memory with the natural
        alignment required by the selected shape. For ``M8N8``, this is 16 bytes.
    :Participation: :ref:`Warp aligned <api-participation>` (32 lanes).
    :Completion: Synchronous.
    :Ordering: Weak memory operation.
    """
    ...


@simt_stub
def store_matrix(
    dst,
    values,
    /,
    *,
    shape: MatrixStoreShape,
    transpose: bool = False,
):
    """
    Collectively store one or more matrices to shared memory.

    Args:
        dst: Pointer to a shared-memory matrix row. Per-lane row addresses
            follow the PTX shape/count layout.
        values: This lane's fragment: one 32-bit scalar or a ``Vector`` of 1, 2,
            or 4 32-bit values. The number of values selects the number of matrices.
        shape (MatrixStoreShape): Matrix shape.
        transpose (const bool): False stores the matrix in row-major order, True stores in
            column-major order.

    :Target: Requires SM90 or newer. Additional shapes follow the `PTX stmatrix
        Target ISA Notes
        <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix>`_.
    :Memory: Destination rows must be in block-local shared memory with the
        natural alignment required by the selected shape. For ``M8N8``, this is
        16 bytes. ``M16N8`` requires ``transpose=True``.
    :Participation: :ref:`Warp aligned <api-participation>` (32 lanes).
    :Completion: Synchronous.
    :Ordering: Weak memory operation.
    """
    ...
