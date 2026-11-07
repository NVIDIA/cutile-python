# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class MemoryScope(Enum):
    """
    The scope of threads that participate in memory ordering.
    """
    NONE = "none"
    """No memory scope. Used for load/store operations with WEAK memory ordering."""

    BLOCK = "block"
    """Ordering guarantees apply to threads within the same block."""

    CLUSTER = "cluster"
    """Ordering guarantees apply to all threads within the same
       thread-block cluster. cuda.lang only."""

    DEVICE = "device"
    """Ordering guarantees apply to all threads on the same GPU."""

    SYS = "sys"
    """Ordering guarantees apply to all threads across the entire system,
       including multiple GPUs and the host."""


class MemoryOrder(Enum):
    """
    Memory ordering semantics of a memory operation.
    """

    WEAK = "weak"
    """Weak (non-atomic) ordering. The default for load/store operations."""

    RELAXED = "relaxed"
    """No ordering guarantees. Cannot be used to synchronize between threads."""

    ACQUIRE = "acquire"
    """Acquire semantics. When this reads a value written by a release,
       the releasing thread's prior writes become visible.
       Subsequent reads/writes within the same block cannot be reordered before this operation."""

    RELEASE = "release"
    """Release semantics. When an acquire reads the value written by this,
       this thread's prior writes become visible to the acquiring thread.
       Prior reads/writes within the same block cannot be reordered after this operation."""

    ACQ_REL = "acq_rel"
    """Combined acquire and release semantics."""


class MemorySpace(Enum):
    """
    CUDA memory address spaces used by pointer dtypes and address-space casts.
    """

    GENERIC = 0
    """Generic address space."""

    GLOBAL = 1
    """Global memory address space."""

    SHARED = 3
    """Shared memory address space local to a thread block."""

    CONSTANT = 4
    """Constant memory address space."""

    LOCAL = 5
    """Thread-local memory address space."""

    TENSOR = 6
    """Tensor memory address space."""

    SHARED_CLUSTER = 7
    """Cluster-visible shared memory address space."""
