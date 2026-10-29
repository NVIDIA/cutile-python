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
        src: Pointer to a matrix row in shared memory.
        shape: Shape and element size.
        count: Number of matrices to load.
        transpose: False loads the matrix in row-major order, True loads in
            column-major order.
        source_format: Packed source format.

    Returns:
        Scalar or vector of 32 bit integers depending on ``count``.
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
        dst: Pointer to a matrix row in shared memory.
        values: One 32-bit scalar or a ``Vector`` of 1, 2, or 4 32-bit values.
            The number of values selects the number of matrices.
        shape: Matrix shape.
        transpose: False stores the matrix in row-major order, True stores in
            column-major order.
    """
    ...
