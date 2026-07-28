# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from cuda.lang._enums import SwizzleMode, TensorMapL2Promotion, TMALoadMode
from cuda.lang._execution import stub


class TensorMap:
    """Descriptor for TMA access to a global array."""

    @stub
    def get_transaction_bytes(
        self,
        *,
        mode: TMALoadMode = TMALoadMode.TILE,
    ) -> int:
        """Return the mbarrier transaction-byte count for a TMA load.

        The returned value is the number of bytes that the TMA load contributes
        to the mbarrier transaction count when the copy completes. The count
        includes bytes corresponding to out-of-bounds elements that are
        zero-filled in shared memory and is unaffected by shared-memory
        swizzling. It can also be used as the destination payload size when
        calculating required shared memory; alignment, layout padding, pipeline
        stages, and other shared-memory objects must be accounted for separately.

        Args:
            mode: TMA global-to-shared load mode whose transaction count is
                requested. Supported values are :attr:`TMALoadMode.TILE` and
                :attr:`TMALoadMode.TILE_GATHER4`.
        """
        ...

    @stub
    def as_opaque_ptr(self):
        """Return this descriptor as an opaque pointer for low-level TMA intrinsics."""
        ...


@stub
def tensor_map_tiled(array,
                     tile_shape: int | tuple[int, ...],
                     *,
                     order: tuple[int, ...] | Literal["C", "F"] = "C",
                     swizzle: SwizzleMode = SwizzleMode.SWIZZLE_NONE,
                     l2_promotion: TensorMapL2Promotion =
                     TensorMapL2Promotion.NONE) -> TensorMap:
    """
    Creates a tiled tensor-map descriptor for TMA access to a global `array`.

    `array` must be a global :class:`Array` passed as a kernel parameter. Its
    element type, base address, shape, and strides supply the corresponding
    tensor-map fields. `tile_shape`, `order`, `swizzle`, and `l2_promotion` must
    be compile-time constants.

    `l2_promotion` controls the L2-promotion size encoded in the descriptor.

    `order` maps tensor-map descriptor axes to `array` axes. If ``order[i] == j``,
    descriptor axis ``i`` describes array axis ``j``. ``order="C"`` is equivalent
    to ``(0, 1, ..., array.ndim - 1)`` and ``order="F"`` reverses that order. An
    explicit permutation such as ``(1, 0, 2)`` may also be used.

    Descriptor axis zero is the fastest-moving TMA dimension, so the corresponding
    array stride must be one element. The remaining strides must be positive and
    satisfy the alignment restrictions of the CUDA tensor-map API. For example,
    an array with shape ``(M, K)`` and strides ``(K, 1)`` requires K to be the
    first descriptor axis::

        # Descriptor axes are (K, M), and TMA coordinates are passed as (k, m).
        a_map = cl.tensor_map_tiled(
            a,
            (BLOCK_K, BLOCK_M),
            order=(1, 0),              # Equivalent to "F" for a rank-two array.
        )

    In general, if ``P = order`` and the array has shape ``S`` and element
    strides ``T``, the descriptor encodes::

        descriptor_shape[i]  = S[P[i]]
        descriptor_stride[i] = T[P[i]]

    A descriptor coordinate ``d`` selects the global element offset::

        global_offset(d) = sum(
            d[i] * descriptor_stride[i] for i in range(array.ndim)
        )

    Coordinates passed as ``src_coordinates`` to
    :func:`copy_async_bulk_tensor_global_to_shared` and as ``dst_coordinates`` to
    :func:`copy_async_bulk_tensor_shared_to_global` use this descriptor-axis order.

    The global strides determine how TMA gathers or scatters global-memory
    elements. They are not preserved as shared-memory strides. Before applying
    an optional swizzle, TMA packs the box densely with descriptor axis zero
    fastest. For tile shape ``(T0, T1, ..., Tn)`` and local coordinate
    ``(d0, d1, ..., dn)``, the logical shared-memory element offset is::

        d0 + T0 * (d1 + T1 * (d2 + ... + T[n - 1] * dn))

    Thus a rank-three descriptor with axes ``(D0, D1, D2)`` has conventional
    outermost-to-innermost shared-memory nesting ``[D2][D1][D0]``.

    `swizzle` optionally permutes the physical shared-memory addresses of the
    densely packed box. It does not change the logical tile coordinates or the
    global-memory address calculation. For 32-, 64-, and 128-byte swizzles,
    the first tile extent multiplied by the array element size must not exceed
    the selected swizzle width. This quantity is the byte width of one tile row
    because descriptor axis zero is the fastest-moving dimension.

    The shared-memory destination must also satisfy the alignment requirement of
    the selected mode. A valid tensor map does not necessarily produce the layout
    expected by its consumer. In particular, see
    :class:`Tcgen05SharedMemoryDescriptor` for arranging swizzled matrix operands
    consumed by :func:`tcgen05_mma`.

    The array rank must be between one and five. `tile_shape` must contain one
    positive integer per array dimension; a scalar integer is accepted for a
    rank-one array. `order` must be a permutation of all array axes.

    Returns a :class:`TensorMap`. Use :meth:`TensorMap.as_opaque_ptr` when a
    low-level TMA intrinsic expects an opaque tensor-map pointer.
    """
    ...
