.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _execution-execution-model:

Execution Model
===============

Abstract Machine
----------------

.. _thread:
.. _block:
.. _grid:

A *SIMT kernel* is executed using the following thread hierarchy:

- A *grid* is a 1D, 2D, or 3D collection of thread blocks.
- A *thread-block cluster* is an optional 1D, 2D, or 3D group of blocks within
  the grid.
- A *thread block* is a 1D, 2D, or 3D collection of threads.
- A *warp* groups threads within a block for SIMT execution.
- A *thread* executes the body of the |kernel|.

A thread can query its position in the thread, block, warp, and cluster
hierarchy using the functions listed under
:ref:`SIMT Model <operations-simt-model>`.

Threads in a block can communicate through shared memory and synchronize
explicitly. Blocks in the same cluster can map each other's shared memory and
synchronize at cluster scope. Global memory is accessible to all blocks,
subject to CUDA synchronization and memory-ordering rules. See the
:ref:`Data Model <data-data-model>` for the corresponding memory spaces.


.. _execution-execution-spaces:

Execution Spaces
----------------

|cuda lang| code runs on one or more *targets*. A target is an execution
environment defined by its hardware resources and programming model.

.. _host code:
.. _SIMT code:

The set of targets where a construct can be used is called its *execution
space*. |cuda lang| defines two execution spaces:

- *Host code* --- Python code that prepares data and launches kernels on a CPU.
- *SIMT code* --- code compiled for CUDA SIMT targets and executed by GPU
  threads.

Some constructs span multiple execution spaces. For example, :func:`cdiv` is
usable in both host code and SIMT code.

A function whose decorator explicitly specifies its execution space is called
an *annotated function*.


.. _execution-simt-functions:

SIMT Functions
--------------

A *SIMT function* is a helper function usable from SIMT code. The
``@cl.function`` decorator explicitly marks such a function. An undecorated
Python function called from a kernel or SIMT function is compiled recursively
as SIMT code, so an explicit decorator is not required.


.. _execution-simt-kernels:

SIMT Kernels
------------

A *SIMT kernel* is an entry point executed by each thread in each block in a
grid. Kernels cannot be called directly. Use :func:`launch` to queue a kernel
for execution with explicit grid and block dimensions.


Python Subset
-------------

|SIMT code| supports a subset of the Python language. There is no Python
runtime within SIMT code. Features such as exceptions and coroutines are not
supported today.

Object Model & Lifetimes
~~~~~~~~~~~~~~~~~~~~~~~~

Arrays and pointers are views of memory and may alias or update the same
storage. Their visibility and lifetime depend on their CUDA memory space.
Because a launch is asynchronous with respect to the host, kernel arguments
must remain valid until the kernel completes.

Control Flow
~~~~~~~~~~~~

Supported Python control flow statements include ``if``, ``while``, and
range-based ``for`` loops.

Current limitations
^^^^^^^^^^^^^^^^^^^

The ``step`` of a ``range`` must be strictly positive. Negative-step ranges
such as ``range(10, 0, -1)`` are not supported today.
