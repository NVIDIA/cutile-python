# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING

from cuda.lang._execution import stub
from cuda.lang._enums import MemoryOrder
from cuda.tile._memory_model import MemoryScope

if TYPE_CHECKING:
    from .types import Pointer, Scalar, Vector


@stub
def load(
    pointer: Pointer,
    /,
    *,
    count: int | None = None,
    alignment: int | None = None,
) -> Scalar | Pointer | Vector:
    """Load one or more consecutive values from this address.

    This operation is valid only for a typed pointer.

    Args:
        pointer: Typed pointer to read from.
        count: Compile-time number of values to load. ``None`` and ``1``
            return a scalar. A value greater than ``1`` returns a vector.
            For best performance, align a vector load to the total size of
            the vector in bytes.
        alignment: Minimum byte alignment that the compiler can assume.
            The value must be a positive power of two. The address must
            have this alignment. If the value is ``None``, the compiler
            does not get an alignment hint.
    """
    ...


@stub
def store(
    pointer: Pointer,
    value: Scalar | Pointer | Vector | int | float | bool,
    /,
    *,
    alignment: int | None = None,
) -> None:
    """Store one or more consecutive values at this address.

    This operation is valid only for a typed pointer. A scalar value stores
    one value. A vector stores all its elements in consecutive locations.

    Args:
        pointer: Typed pointer to write to.
        value: Scalar or vector to store. The value must be compatible with
            the pointee data type.
        alignment: Minimum byte alignment that the compiler can assume.
            The value must be a positive power of two. The address must
            have this alignment. If the value is ``None``, the compiler
            does not get an alignment hint.
    """
    ...


@stub
def atomic_load(
    pointer: Pointer,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQUIRE,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
    mmio: bool = False,
    alignment: int | None = None,
) -> Scalar | Pointer:
    """Atomically load one value from this address.

    This operation is valid only for a typed pointer. The pointee size must
    be a power-of-two number of bytes.

    Args:
        pointer: Typed pointer to read from.
        memory_order: Memory order for the load. Supported values are
            ``MemoryOrder.RELAXED`` and ``MemoryOrder.ACQUIRE``.
        memory_scope: Scope of threads that participate in memory ordering.
        mmio: Whether to use a memory-mapped I/O access. MMIO requires
            ``MemoryOrder.RELAXED`` or ``MemoryOrder.ACQUIRE`` and
            ``MemoryScope.SYS``. The pointer must refer to global memory.
        alignment: Minimum byte alignment that the compiler can assume.
            The value must be a positive power of two. The address must
            have at least this alignment. If the value is ``None``, the
            natural alignment of the pointee data type is used.
    """
    ...


@stub
def atomic_store(
    pointer: Pointer,
    value: Scalar | Pointer | int | float | bool,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.RELEASE,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
    mmio: bool = False,
    alignment: int | None = None,
) -> None:
    """Atomically store one value to this address.

    This operation is valid only for a typed pointer. The pointee size must
    be a power-of-two number of bytes.

    Args:
        pointer: Typed pointer to write to.
        value: Scalar value to store. The value must be compatible with the
            pointee data type.
        memory_order: Memory order for the store. Supported values are
            ``MemoryOrder.RELAXED`` and ``MemoryOrder.RELEASE``.
        memory_scope: Scope of threads that participate in memory ordering.
        mmio: Whether to use a memory-mapped I/O access. MMIO requires
            ``MemoryOrder.RELAXED`` or ``MemoryOrder.RELEASE`` and
            ``MemoryScope.SYS``. The pointer must refer to global memory.
        alignment: Minimum byte alignment that the compiler can assume.
            The value must be a positive power of two. The address must
            have at least this alignment. If the value is ``None``, the
            natural alignment of the pointee data type is used.
    """
    ...


__all__ = (
    "load",
    "store",
    "atomic_load",
    "atomic_store"
)
