.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

CUDA Languages for Python
=========================

CUDA Languages brings two GPU programming models to Python:
``cuda.lang`` for SIMT programming and ``cuda.tile`` (cuTile) for
Tile programming. They share language design and compiler infrastructure.

Both namespaces are included in the ``cuda-tile`` package, which will be renamed
to ``cuda-languages`` in the near future.
Package setup is described in :doc:`getting-started/installation`.

.. _choose-a-programming-model:

Programming Models
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``cuda.tile [Stable]``
     - ``cuda.lang [Beta]``
   * - Execution unit
     - Abstract tile blocks.
     - Threads, warps, thread blocks, and thread block clusters.
   * - Data model
     - Array and tile.
     - Array, pointer, scalar, and vector.
   * - Execution control
     - Compiler-managed mapping to threads, synchronization within a block,
       and hardware operations.
     - Explicit thread-block size, shared memory, tensor memory, barriers, and
       other hardware primitives.
   * - Architecture portability
     - Arch-agnostic tile operations let the compiler target different
       supported GPU architectures, providing portability without rewriting
       the kernel for each architecture.
     - Portable SIMT operations, with optional architecture-specific primitives
       that may require different implementations across GPU generations.
   * - Start here
     - :doc:`getting-started/tile`
     - :doc:`getting-started/lang`

``cuda.tile`` expresses parallel array computations at tile granularity, with
compiler-managed execution across supported GPU architectures. ``cuda.lang``
provides explicit control over threads, memory, and synchronization, including
architecture-specific optimizations.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting-started/index
   language
   tile/index
   lang/index
   integration/index
   reference/index
