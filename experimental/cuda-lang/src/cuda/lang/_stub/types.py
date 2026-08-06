# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, TypeVar

from cuda.tile import DType
from cuda.tile._execution import stub
from cuda.lang._enums import MemoryOrder
from cuda.tile._memory_model import MemoryScope, MemorySpace
from .._enums import VectorReduction
from .._stub import math as cl_math


T = TypeVar("T")


class Scalar(Generic[T]):
    def __add__(self, other):
        return cl_math.add(self, other)

    def __sub__(self, other):
        return cl_math.sub(self, other)

    def __mul__(self, other):
        return cl_math.mul(self, other)

    def __truediv__(self, other):
        return cl_math.truediv(self, other)

    def __floordiv__(self, other):
        return cl_math.floordiv(self, other)

    def __mod__(self, other):
        return cl_math.mod(self, other)

    def __pow__(self, other):
        return cl_math.pow(self, other)

    def __and__(self, other):
        return cl_math.bitwise_and(self, other)

    def __or__(self, other):
        return cl_math.bitwise_or(self, other)

    def __xor__(self, other):
        return cl_math.bitwise_xor(self, other)

    def __radd__(self, other):
        return cl_math.add(other, self)

    def __rsub__(self, other):
        return cl_math.sub(other, self)

    def __rmul__(self, other):
        return cl_math.mul(other, self)

    def __rtruediv__(self, other):
        return cl_math.truediv(other, self)

    def __rfloordiv__(self, other):
        return cl_math.floordiv(other, self)

    def __rmod__(self, other):
        return cl_math.mod(other, self)

    def __rpow__(self, other):
        return pow(other, self)

    def __rand__(self, other):
        return cl_math.bitwise_and(other, self)

    def __ror__(self, other):
        return cl_math.bitwise_or(other, self)

    def __rxor__(self, other):
        return cl_math.bitwise_xor(other, self)

    def __ge__(self, other):
        return cl_math.greater_equal(self, other)

    def __gt__(self, other):
        return cl_math.greater(self, other)

    def __le__(self, other):
        return cl_math.less_equal(self, other)

    def __lt__(self, other):
        return cl_math.less(self, other)

    def __eq__(self, other):
        return cl_math.equal(self, other)

    def __ne__(self, other):
        return cl_math.not_equal(self, other)

    def __neg__(self):
        return cl_math.negative(self)

    def __invert__(self):
        return cl_math.bitwise_not(self)

    def __divmod__(self, other):
        return cl_math.divmod(self, other)

    def __rdivmod__(self, other):
        return cl_math.divmod(other, self)


class Vector(Generic[T]):
    """Fixed-size collection returned by vectorized pointer operations."""

    @stub
    def __init__(self, *elements: T, dtype: DType | None = None) -> None:
        """Constructs a vector from scalar elements, optionally with an explicit dtype.

        Args:
            elements: Variable number of elements used to construct the vector.
            dtype: Data type of the vector.
        """

    @property
    @stub
    def dtype(self) -> "DType": ...

    @property
    @stub
    def element_count(self) -> int: ...

    @stub
    def __getitem__(self, item): ...

    @stub
    def __setitem__(self, key, value): ...

    @stub
    def with_item(self, index: int, value: T) -> "Vector[T]":
        """Return a new vector with one element replaced.

        Vectors have value semantics, so this operation does not modify the
        original vector. ``index`` must select an element of the vector.

        Args:
            index: Index in vector to replace.
            value: New value.
        """

    @stub
    def astype(self, dtype: "DType") -> "Vector":
        """Convert each element to ``dtype``.

        Returns a new vector of the same length with the given dtype.

        Args:
            dtype: Target data type of the result vector.
        """

    @stub
    def bitcast(self, dtype: "DType") -> "Vector":
        """Reinterpret each element's bits as ``dtype``.

        The element and target dtype must have the same bitwidth, producing a
        ``Vector`` of the same length.

        Args:
            dtype: Target element data type (same bitwidth as the current one).
        """

    @stub
    def reinterpret_as_scalar(self, dtype: "DType"):
        """Reinterpret the whole vector's bits as a single scalar of ``dtype``.

        ``dtype``'s bitwidth must equal the vector's total bitwidth (element
        bitwidth times length).

        The bytes are packed **little-endian**, as if the vector were stored to
        memory and loaded back as ``dtype``.

        Args:
            dtype: Target scalar data type.
        """

    @stub
    def reinterpret_as_vector(self, dtype: "DType", length: int) -> "Vector":
        """Reinterpret the whole vector's bits as ``Vector[dtype, length]``.

        ``length * dtype.bitwidth`` must equal the source vector's total
        bitwidth.

        The raw bytes are preserved and re-split **little-endian**, as if the
        vector were stored to memory and loaded back as the new vector type.

        Args:
            dtype: Target element data type of the result vector.
            length: Number of elements in the result vector.
        """

    @stub
    def reduce(
        self,
        op: VectorReduction,
        /,
        *,
        propagate_nan: bool = False,
        reassociate: bool = False,
    ) -> T:
        """Reduce vector ``self`` to a scalar.

        Args:
            op: Operation to apply to the vector elements.
            propagate_nan: For floating-point min and max, return NaN if any
                element is NaN.
            reassociate: Permit the compiler to change the operation order.
        """

    @stub
    def __len__(self): ...


class Pointer(Generic[T]):
    """Address in a CUDA memory space.

    A typed pointer identifies the data type at its address. An opaque pointer
    does not identify a data type. Pointer arithmetic and memory access require
    a typed pointer.
    Pointer offsets are given in element counts, not in bytes.
    """

    @stub
    def __add__(self, other):
        """Return a pointer that is ``other`` elements after this pointer.

        Args:
            other: Integral scalar that gives the element offset.
        """

    @stub
    def __sub__(self, other):
        """Return a pointer that is ``other`` elements before this pointer.

        Args:
            other: Integral scalar that gives the element offset.
        """

    @stub
    def __getitem__(self, index):
        """Load one value at an element offset from this pointer.

        ``self[index]`` is equivalent to ``(self + index).load()``.

        Args:
            index: Integral scalar that gives the element offset.
        """

    @stub
    def __setitem__(self, index, value):
        """Store one value at an element offset from this pointer.

        ``self[index] = value`` is equivalent to
        ``(self + index).store(value)``.

        Args:
            index: Integral scalar that gives the element offset.
            value: Value to store.
        """

    @stub
    def load(
        self,
        *,
        count: int | None = None,
        alignment: int | None = None,
    ) -> T | Vector[T]:
        """Load one or more consecutive values from this address.

        This operation is valid only for a typed pointer.

        Args:
            count: Compile-time number of values to load. ``None`` and ``1``
                return a scalar. A value greater than ``1`` returns a vector.
                For best performance, align a vector load to the total size of
                the vector in bytes.
            alignment: Minimum byte alignment that the compiler can assume.
                The value must be a positive power of two. The address must
                have this alignment. If the value is ``None``, the compiler
                does not get an alignment hint.
        """

    @stub
    def store(
        self,
        value: T | Vector[T],
        *,
        alignment: int | None = None,
    ) -> None:
        """Store one or more consecutive values at this address.

        This operation is valid only for a typed pointer. A scalar value stores
        one value. A vector stores all its elements in consecutive locations.

        Args:
            value: Scalar or vector to store. The value must be compatible with
                the pointee data type.
            alignment: Minimum byte alignment that the compiler can assume.
                The value must be a positive power of two. The address must
                have this alignment. If the value is ``None``, the compiler
                does not get an alignment hint.
        """

    @stub
    def atomic_load(
        self,
        *,
        memory_order: MemoryOrder = MemoryOrder.ACQUIRE,
        memory_scope: MemoryScope = MemoryScope.DEVICE,
        mmio: bool = False,
        alignment: int | None = None,
    ) -> T:
        """Atomically load one value from this address.

        This operation is valid only for a typed pointer. The pointee size must
        be a power-of-two number of bytes.

        Args:
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

    @stub
    def atomic_store(
        self,
        value: T,
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

    @property
    @stub
    def opaque(self) -> bool:
        """Whether the pointer has no pointee data type.

        This value is a compile-time constant.
        """

    @property
    @stub
    def pointee_dtype(self) -> DType:
        """Data type of the value at this address.

        Access to this property causes a compilation error if the pointer is
        opaque.
        """

    @property
    @stub
    def memory_space(self) -> MemorySpace:
        """CUDA memory space of this pointer."""
