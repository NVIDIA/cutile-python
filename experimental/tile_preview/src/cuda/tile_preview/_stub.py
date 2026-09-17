# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple

from cuda.tile._execution import stub

from ._types import Tilelib


@stub
def foreign_call(
    *,
    tilelibs: Tuple[Tilelib, ...],
    symbol_name: str,
    constant_params: Tuple[Any, ...],
    inputs: Tuple[Any, ...],
    output_types: Tuple[Any, ...],
) -> Tuple[Any, ...]:
    """Calls a foreign function supplied by a TileIR library (``.tilelib`` file).

    Requires TileIR bytecode version 13.5 or later.

    Args:
        tilelibs (tuple[Tilelib, ...]): A non-empty tuple containing the TileIR
            libraries (``.tilelib`` files) needed by the call. A library's
            version must be incremented whenever it or one of its dependencies
            changes.
        symbol_name (str): Name of the function to call. Must be a non-empty
            compile-time constant string.
        constant_params (tuple): Compile-time constant arguments forwarded to
            the library to dispatch the implementation. Each element must be
            an ``int``, ``str``, or ``cuda.tile`` dtype.
        inputs (tuple): Runtime inputs passed to the callee. Each element must
            be a tile or scalar.
        output_types (tuple): Descriptors for returned values. Use a dtype for
            a scalar, or ``(dtype, shape)`` for a tile with an explicit shape.

    Returns:
        tuple: One result value per entry in `output_types`, in order.

    Examples:

        @ct.kernel
        def kernel(src, dst):
            tile = ct.load(src, (0, 0), shape=(8, 8))
            result, info = tile_preview.foreign_call(
                tilelibs=(
                    tile_preview.Tilelib(
                        path="cholesky.tilelib",
                        version="1.0.0",
                    ),
                ),
                symbol_name="cholesky",
                constant_params=("lower", ct.float32),
                inputs=(tile,),
                output_types=(
                    (ct.float32, (8, 8)),
                    ct.int32,
                ),
            )
            ct.store(dst, (0, 0), result)
    """
