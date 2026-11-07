.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

Execution Model
===============

This chapter describes tile blocks, kernel launches, and array lifetimes.
The :doc:`/language` covers shared Python semantics, helper functions,
constants, static evaluation, control flow, and JIT specialization. See
:ref:`tile-operations-annotations` for parameter annotation APIs and
:ref:`language-execution-spaces` for host, SIMT, and tile execution spaces.

Abstract Machine
----------------

.. _grid:

A *tile kernel* is executed by logical thread |blocks| organized in a 1D,
2D, or 3D *grid*.

.. _block:

Each *block* executes the kernel body as one logical thread operating on
scalars and tiles. The compiler maps that logical execution onto hardware
threads and instructions, choosing the participating threads for each
operation.

Tile programs express |block|-level parallelism only with no exposure to
individual threads within the block.

Explicit synchronization or communication within a |block| is not permitted,
but is allowed between different |blocks|.

The compiler maps tile operations to threads and manages synchronization within
each block. For ordering and synchronization between blocks, see
:doc:`memory_model`. A tile kernel can also call a SIMT function through
tile/SIMT interoperability; see :doc:`interoperability`.

A |block| is a unit of execution, while a |tile| is a unit of data.
A single block may operate on multiple |tiles|
with different shapes originating from different |global arrays|.


.. _execution-tile-functions:

Tile Functions
--------------

.. autoclass:: cuda.tile.function

.. _tile-kernels:
.. _execution-tile-kernels:

Tile Kernels
------------

Mark a kernel with :py:obj:`ct.kernel <cuda.tile.kernel>` and queue it with
:py:obj:`ct.launch(stream, grid, kernel, args) <cuda.tile.launch>`. The grid specifies the number of
logical blocks along each axis; :py:func:`cuda.tile.bid` identifies a block's
index along an axis. The compiler chooses the thread mapping within each block,
so the launch specifies only the grid dimensions.

.. autoclass:: cuda.tile.kernel
   :members: replace_hints

.. autofunction:: cuda.tile.launch

Array Lifetimes and Aliasing
----------------------------

Global array views are immutable, while their underlying device memory can be
read and written. The caller must keep every array alive until the kernel
completes and obey the aliasing requirements in :ref:`data-global-arrays`.

Loop Index Types
----------------

For the supported control-flow syntax, see
:ref:`language-supported-python-features`. In tile code, the bounds, step, and
induction variable of a ``for`` loop share one integer type. The common type is
at least ``int32`` and widens to ``int64`` if any bound or step is ``int64``.
This also applies to the implicit lower bound and step in ``range(n)``. A literal
bound too large for ``int32``, such as ``range(3_000_000_000)``, widens the loop
to ``int64`` as well.
