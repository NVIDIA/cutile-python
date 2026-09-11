# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from cuda.tile import _cext
from cuda.lang._enums import (
    SwizzleMode,
    TensorMapInterleave,
    TensorMapFloatOOBFill,
    TensorMapL2Promotion,
)
from cuda.lang._execution import stub

from cuda.lang._datatype import tensor_map_descriptor

TensorMapDataType = _cext.TensorMapDataType


class TensorMap:
    """Immutable opaque tensor-map descriptor value.

    Created by :func:`tensor_map_tiled`. A kernel argument receives a pointer
    to its own read-only copy of this value. Descriptor fields are not exposed
    as language metadata. The referenced tensor data remains mutable.

    The caller must keep the memory referenced by the descriptor alive until
    all GPU operations using it have completed. This value does not own that memory.
    """

    __slots__ = ("_data",)

    def __new__(cls):
        raise TypeError("Use tensor_map_tiled() to create a TensorMap")

    @classmethod
    def _from_bytes(cls, data: bytes):
        if type(data) is not bytes or len(data) != _cext._TENSOR_MAP_DESCRIPTOR_BYTES:
            raise ValueError("Expected 128 descriptor bytes")
        value = object.__new__(cls)
        object.__setattr__(value, "_data", data)
        return value

    def __setattr__(self, name, value):
        raise AttributeError("TensorMap values are immutable")

    def __delattr__(self, name):
        raise AttributeError("TensorMap values are immutable")

    def __bytes__(self) -> bytes:
        return self._data

    @property
    def dtype(self):
        return tensor_map_descriptor

    @property
    def _cuda_lang_tensor_map_bytes(self):
        return self._data


def _normalize_order(
    order: tuple[int, ...] | Literal["C", "F"], rank: int
) -> tuple[int, ...]:
    if order == "C":
        return tuple(range(rank))
    if order == "F":
        return tuple(range(rank - 1, -1, -1))
    if isinstance(order, str):
        raise ValueError("order must be 'C', 'F', or an axis permutation")
    if not isinstance(order, tuple) or len(order) != rank:
        raise ValueError("order must be a permutation of all array axes")

    result = []
    seen = set()
    for axis in order:
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise TypeError("order must contain only integers")
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank or axis in seen:
            raise ValueError("order must be a permutation of all array axes")
        result.append(axis)
        seen.add(axis)
    return tuple(result)


# Keep tile execution enabled during the compatibility phase. Once existing
# call sites have migrated, this becomes host-only and CreateTensorMap hoisting
# can be removed from device compilation.
@stub(host=True, compiled_host=True)
def tensor_map_tiled(array,
                     tile_shape: int | tuple[int, ...],
                     *,
                     order: tuple[int, ...] | Literal["C", "F"] = "C",
                     interleave: TensorMapInterleave = TensorMapInterleave.NONE,
                     swizzle: SwizzleMode = SwizzleMode.SWIZZLE_NONE,
                     l2_promotion: TensorMapL2Promotion =
                     TensorMapL2Promotion.NONE,
                     oob_fill: TensorMapFloatOOBFill =
                     TensorMapFloatOOBFill.NONE) -> TensorMap:
    """Create a tiled tensor-map descriptor for TMA access to a global array.

    Call this function from Python or a :func:`host_entry`, then pass the
    resulting :class:`TensorMap` to a kernel through :func:`launch`. The
    ``order``, ``interleave``, ``swizzle``, ``l2_promotion``, and ``oob_fill``
    arguments must be compile-time constants.

    Args:
        array: Global array supplying the element type, base address, shape,
            and strides. Its rank must be between one and five.
        tile_shape (int | tuple[int, ...]): One positive integer no greater
            than 256 per array dimension. For a one-dimensional array, an
            integer is equivalent to a one-element tuple. May contain runtime
            integers inside a :func:`host_entry`.
        order (tuple[int, ...] | str): Permutation mapping descriptor axes to
            array axes. ``"C"`` selects ``(0, 1, ..., array.ndim - 1)`` and
            ``"F"`` reverses that order. An explicit permutation such as
            ``(1, 0, 2)`` may also be used.
        interleave (TensorMapInterleave): Global-memory interleave mode.
        swizzle (SwizzleMode): Shared-memory swizzle mode.
        l2_promotion (TensorMapL2Promotion): L2-promotion size encoded in the
            descriptor.
        oob_fill (TensorMapFloatOOBFill): Floating-point out-of-bounds fill
            behavior encoded in the descriptor.

    Returns:
        TensorMap: Immutable value containing the descriptor bytes. A kernel
        launch copies the value into parameter storage and passes a pointer to
        ``tensor_map_descriptor`` for that copy.

    Notes:
        If ``order[i] == j``, descriptor axis ``i`` describes array axis ``j``.
        Descriptor axis zero is the innermost TMA dimension, so its array stride
        must be one element. Other strides must be positive and satisfy the
        CUDA tensor-map alignment requirements. With ``P = order``, array shape
        ``S``, and element strides ``T``, the descriptor encodes::

            descriptor_shape[i]  = S[P[i]]
            descriptor_stride[i] = T[P[i]]

        The ``src_coordinates`` of
        :func:`copy_async_bulk_tensor_global_to_shared` and ``dst_coordinates``
        of :func:`copy_async_bulk_tensor_shared_to_global` follow this axis
        order. Global strides describe global memory, not shared memory.

        For ``interleave=TensorMapInterleave.NONE``, a descriptor coordinate
        ``d`` selects the global element offset::

            global_offset(d) = sum(
                d[i] * descriptor_stride[i] for i in range(array.ndim)
            )

        Before an optional swizzle, TMA packs the box densely in shared memory
        with descriptor axis zero fastest. For tile shape ``(T0, T1, ..., Tn)``
        and local coordinate ``(d0, d1, ..., dn)``, the logical shared-memory
        element offset is::

            d0 + T0 * (d1 + T1 * (d2 + ... + T[n - 1] * dn))

        Thus a rank-three descriptor with axes ``(D0, D1, D2)`` has
        outermost-to-innermost shared-memory nesting ``[D2][D1][D0]``.
        Swizzling changes physical shared-memory addresses, not logical tile
        coordinates or global-memory addresses. With noninterleaved data and a
        32-, 64-, or 128-byte swizzle, the first tile extent times the element
        size must not exceed the swizzle width. The shared-memory destination
        must also meet the alignment requirement of the selected mode.

        Interleaved maps require rank at least three. The 32-byte interleave
        mode requires a 32-byte aligned base address and byte strides, and
        ``SWIZZLE_32B``. See the `PTX interleave layout
        <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#interleave-layout>`_
        for channel-slice organization and the `CUDA Driver API tensor-map
        documentation
        <https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html>`_
        for encoding restrictions.

        A valid tensor map may still produce a layout unsuitable for its
        consumer. See :class:`Tcgen05SharedMemoryDescriptor` for arranging
        swizzled matrix operands consumed by :func:`tcgen05_mma`.

    Examples:
        For an array with shape ``(M, K)`` and strides ``(K, 1)``, make K the
        first descriptor axis::

            # Descriptor axes are (K, M); TMA coordinates are (k, m).
            a_map = cl.tensor_map_tiled(
                a,
                (BLOCK_K, BLOCK_M),
                order=(1, 0),  # Equivalent to "F" for a rank-two array.
            )
    """
    if not isinstance(interleave, TensorMapInterleave):
        raise TypeError(
            f"interleave must be a TensorMapInterleave, got {type(interleave).__name__}"
        )
    if not isinstance(swizzle, SwizzleMode):
        raise TypeError(
            f"swizzle must be a SwizzleMode, got {type(swizzle).__name__}"
        )
    if not isinstance(l2_promotion, TensorMapL2Promotion):
        raise TypeError(
            "l2_promotion must be a TensorMapL2Promotion, "
            f"got {type(l2_promotion).__name__}"
        )
    if not isinstance(oob_fill, TensorMapFloatOOBFill):
        raise TypeError(
            "oob_fill must be a TensorMapFloatOOBFill, "
            f"got {type(oob_fill).__name__}"
        )
    if type(tile_shape) is int:
        tile_shape = (tile_shape,)
    order = _normalize_order(order, len(tile_shape))
    data = _cext._tensor_map_tiled(
            array,
            tile_shape,
            order,
            interleave.value,
            swizzle.value,
            l2_promotion.value,
            oob_fill.value,
            )
    return TensorMap._from_bytes(data)
