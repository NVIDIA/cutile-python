# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING

from cuda.lang._execution import simt_stub
from cuda.lang._enums import MemoryOrder
from cuda.tile._memory_model import MemoryScope

if TYPE_CHECKING:
    from .types import Pointer, Scalar, Vector


@simt_stub
def load(
    pointer: Pointer,
    /,
    *,
    count: int = 1,
    alignment: int | None = None,
) -> Scalar | Pointer | Vector:
    """Load one or more consecutive values from this address.

    This operation is valid only for a typed pointer.

    Args:
        pointer: Typed pointer to read from.
        count: Compile-time number of values to load. ``1``
            returns a scalar. A value greater than ``1`` returns a vector.
            For best performance, align a vector load to the total size of
            the vector in bytes.
        alignment: Minimum byte alignment that the compiler can assume.
            The value must be a positive power of two. The address must
            have this alignment. If the value is ``None``, the compiler
            does not get an alignment hint.
    """
    ...


@simt_stub
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


@simt_stub
def atomic_load(
    pointer: Pointer,
    /,
    *,
    count: int = 1,
    memory_order: MemoryOrder = MemoryOrder.ACQUIRE,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
    mmio: bool = False,
    alignment: int | None = None,
) -> Scalar | Pointer | Vector:
    """Atomically load one or more consecutive values from this address.

    This operation is valid only for a typed pointer. The total access size
    must be a power-of-two number of bytes. A vector is loaded as one atomic
    value instead of atomic loads for each vector element.

    Args:
        pointer: Typed pointer to read from.
        count: Compile-time number of values to load. ``1``
            returns a scalar. A value greater than ``1`` returns a vector.
        memory_order: Memory order for the load. Supported values are
            ``MemoryOrder.RELAXED`` and ``MemoryOrder.ACQUIRE``.
        memory_scope: Scope of threads that participate in memory ordering.
        mmio: Whether to use a memory-mapped I/O access. MMIO requires
            ``MemoryOrder.RELAXED`` or ``MemoryOrder.ACQUIRE`` and
            ``MemoryScope.SYS``. The pointer must refer to global memory.
        alignment: Minimum byte alignment that the compiler can assume.
            The value must be a positive power of two. The address must
            have at least this alignment. If the value is ``None``, the
            total access size is used as the alignment.
    """
    ...


@simt_stub
def atomic_store(
    pointer: Pointer,
    value: Scalar | Pointer | Vector | int | float | bool,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.RELEASE,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
    mmio: bool = False,
    alignment: int | None = None,
) -> None:
    """Atomically store one or more consecutive values to this address.

    This operation is valid only for a typed pointer. The total access size
    must be a power-of-two number of bytes. A vector is stored as one atomic
    value instead of atomic stores for each vector element.

    Args:
        pointer: Typed pointer to write to.
        value: Scalar or vector to store. The element data type must be
            compatible with the pointee data type.
        memory_order: Memory order for the store. Supported values are
            ``MemoryOrder.RELAXED`` and ``MemoryOrder.RELEASE``.
        memory_scope: Scope of threads that participate in memory ordering.
        mmio: Whether to use a memory-mapped I/O access. MMIO requires
            ``MemoryOrder.RELAXED`` or ``MemoryOrder.RELEASE`` and
            ``MemoryScope.SYS``. The pointer must refer to global memory.
        alignment: Minimum byte alignment that the compiler can assume.
            The value must be a positive power of two. The address must
            have at least this alignment. If the value is ``None``, the
            total access size is used as the alignment.
    """
    ...


__all__ = (
    "load",
    "store",
    "atomic_load",
    "atomic_store"
)
