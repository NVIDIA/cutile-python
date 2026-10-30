.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.tile

Memory Model
============

cuTile's memory model permits the compiler and hardware to reorder operations
for performance. Without explicit synchronization, the ordering of memory
accesses across |blocks| is not guaranteed.

To coordinate memory accesses between |blocks|, cuTile provides two attributes
for atomic operations:

*   **Memory Order** --- defines the ordering semantics of an atomic operation.
*   **Memory Scope** --- defines the set of |blocks| that participate in ordering.

Synchronization operates at per-element granularity: each element in the array participates
independently in the memory model.

For further details, see the Memory Model section of the `Tile IR documentation <https://docs.nvidia.com/cuda/tile-ir/>`_.

.. _memory-model-memory-order:

Memory Order
-------------------------

.. autoclass:: cuda.tile.MemoryOrder()
   :members:
   :undoc-members:
   :member-order: bysource

.. _memory-model-memory-scope:

Memory Scope
------------

The shared :class:`MemoryScope <cuda.lang.MemoryScope>` enum defines the available
scopes. Both ``cuda.tile.MemoryScope`` and ``cuda.lang.MemoryScope`` export this enum.
