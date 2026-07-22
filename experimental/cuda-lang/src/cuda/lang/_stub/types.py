# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, TypeVar, Literal

from cuda.tile import MemoryOrder, DType
from cuda.tile._execution import stub
from cuda.tile._memory_model import MemorySpace
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


class Pointer(Generic[T]):
    """Typed address into a CUDA memory space with low-level load and store operations."""

    @stub
    def load(
            self,
            *,
            count: int | None = None,
            alignment: int | None = None,
            volatile: bool = False,
            memory_order: MemoryOrder | None = None,
    ) -> T | Vector[T]:
        """
        Low-level API to read from memory.

        Args:
            count: If count is provided, a vector will be returned.
                For best performance, vector loads should be aligned to the
                number of bytes in the vector.
            alignment: Inform the compiler that the address being loaded from
                is aligned to at least this many bytes.
                The user is responsible for ensuring aligned loads occur only
                on appropriately aligned pointers.
                If alignment is None, do not give the compiler any alignment
                hints.
            volatile: If True, the compiler will not modify the number of times
                this load is performed nor the order of execution with respect
                to other volatile operations.
            memory_order: When memory_order is specified, the load is atomic.
                If alignment is None, the natural alignment of the loaded type
                (its size in bytes) is used.
                Atomic loads require a pointee type with a bit width that
                is a power of two greater than or equal to one byte.
        """

    @stub
    def store(
            self,
            value: T | Vector[T],
            *,
            alignment: int | None = None,
            volatile: bool = False,
            memory_order: Literal[MemoryOrder.RELAXED,
                                  MemoryOrder.RELEASE, MemoryOrder.WEAK] | None = None,
    ) -> None:
        """
        Low-level API to store to memory.

        Args:
            value: Scalar or vector to be stored to the given address.
            alignment: Inform the compiler that the address being stored to
                is aligned to at least this many bytes.
                The user is responsible for ensuring aligned loads occur only
                on appropriately aligned pointers.
                If alignment is None, do not give the compiler any alignment
                hints.
            volatile: If True, the compiler will not modify the number of times
                this store is performed nor the order of execution with respect
                to other volatile operations.
            memory_order: When memory_order is specified, the store is atomic.
                If alignment is None, the natural alignment of the stored type
                (its size in bytes) is used.
                Atomic stores require a pointee type with a bit width that
                is a power of two greater than or equal to one byte.
                Only relaxed, release, and weak are valid memory orders on
                stores.
        """

    @property
    @stub
    def opaque(self) -> bool:
        """
        Whether the pointer is opaque, i.e. doesn't point to a value of a specific data type.
        This is a compile-time constant boolean.
        """

    @property
    @stub
    def pointee_dtype(self) -> DType:
        """
        Data type of the value that this pointer points to.
        Raises a compilation error if the pointer is opaque.
        """

    @property
    @stub
    def memory_space(self) -> MemorySpace:
        """
        Memory space of this pointer.
        """
