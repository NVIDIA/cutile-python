.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _lang-execution:

Execution Model
===============

This chapter describes the SIMT thread block hierarchies, kernel launches, and
array lifetimes.
The :doc:`/language` covers shared Python semantics, helper functions,
constants, static evaluation, control flow, and JIT specialization. See
:ref:`lang-operations-annotations` for parameter annotation APIs and
:ref:`language-execution-spaces` for host, SIMT, and tile execution spaces.

A SIMT kernel describes the work of one CUDA thread. A launch creates a grid
of thread blocks, and each thread in each block executes the kernel body.
Threads use their indices to select data and decide which work to perform.
The host chooses both the grid dimensions and the thread-block dimensions.

Threads and Blocks
------------------

The grid and each block can have one, two, or three dimensions. Axes ``0``,
``1``, and ``2`` correspond to CUDA's x, y, and z axes. Omitted launch dimensions
default to one. Index-query APIs require a compile-time axis.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Query
     - Meaning
   * - :func:`thread_index`
     - This thread's index within its block along an axis.
   * - :func:`thread_count`
     - The block's size in threads along an axis.
   * - :func:`block_index`
     - This block's index within the grid along an axis.
   * - :func:`block_count`
     - The grid's size in blocks along an axis.

For a one-dimensional launch, a thread's global element index is:

.. code-block:: python

   i = cl.block_index(0) * cl.thread_count(0) + cl.thread_index(0)

When the grid covers more elements than the input contains, some threads have
out-of-bounds indices. The :doc:`vector-add quickstart </getting-started/lang>`
guards those threads' array accesses.

Mapping a Two-Dimensional Array
-------------------------------

Grid dimensions specify the number of blocks along each axis. In this
mapping between threads and array elements, x selects columns and y
selects rows. The array has seventeen rows and thirty-five columns, so both
axes have partial edge blocks:

.. testcode:: simt-copy-2d
   :template: setup_only.py

   @cl.kernel
   def copy_2d(source, out):
       col = cl.block_index(0) * cl.thread_count(0) + cl.thread_index(0)
       row = cl.block_index(1) * cl.thread_count(1) + cl.thread_index(1)
       if row < out.shape[0] and col < out.shape[1]:
           out[row, col] = source[row, col]

   rows, cols = 17, 35
   source = torch.arange(rows * cols, dtype=torch.float32,
                         device="cuda:0").reshape(rows, cols)
   out = torch.empty_like(source)
   block = (16, 8)
   grid = (cl.cdiv(cols, block[0]), cl.cdiv(rows, block[1]))
   cl.launch(stream, grid, block, copy_2d, (source, out))
   torch.cuda.synchronize()
   torch.testing.assert_close(out, source)
   print("2D copy passed.")

.. testoutput:: simt-copy-2d

   2D copy passed.

Each block has 128 threads, arranged as sixteen columns by eight rows. The
grid has three blocks along each axis. Threads outside the output shape do
no memory accesses.

Warps and Lanes
---------------

Threads within a block are grouped into warps of 32 lanes.
:func:`lane_index` and :func:`lane_count` return a thread's lane index and the
warp size. :func:`warp_index` identifies the warp within its block, and
:func:`warp_count` gives the number of warps, including a partial final warp.
For multidimensional blocks, threads are numbered with x varying fastest,
then y, then z. Consecutive groups of 32 thread numbers form warps.
This determines which threads communicate through warp operations. For
example, in a ``(16, 8)`` block, each warp spans two rows of sixteen threads.

Different threads may take different branches and compute different scalar
values. Warp shuffles and votes provide direct
communication between participating lanes. Every calling lane must be named
in the mask, and all non-exited lanes named in that mask must execute the
corresponding operation with the same mask. A shuffle's source lane must also
be active and participating for its value to be defined.

:func:`barrier_sync_warp` orders shared-memory accesses between lanes. See :func:`shuffle_sync`,
:func:`vote_ballot_sync`, and the `PTX warp barrier rules
<https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-warp-sync>`_.

For a GEMM that assigns data movement and computation to different warps, see
:doc:`warp-specialization`.

Launching and Ordering Work
---------------------------

Mark the entry point with :py:obj:`@cl.kernel <cuda.lang.kernel>` and enqueue it with
:py:obj:`cl.launch(stream, grid, block, kernel, args) <cuda.lang.launch>`. The tuple of arguments follows
the kernel's parameter order. The first launch compiles a specialization;
see :ref:`language-jit-compilation` for the shared specialization rules.

Launches are asynchronous with respect to Python. Work enqueued on the same
stream is ordered, while work on different streams needs explicit dependencies
when it shares data. Input and output allocations must remain alive until GPU
work using them completes. See :doc:`/integration/index` for host integration.

To launch SIMT kernels from compiled CPU code, see :ref:`lang-compiled-host-functions`.

Blocks in a grid may execute in any order. A block barrier coordinates threads
within one block. Blocks can communicate and
synchronize through global memory using atomic operations with appropriate
memory ordering and scope. For example, a producer thread in one block can
publish data with a device-scope release store to a flag, and a consumer thread
in another block can observe that flag with a device-scope acquire load before
reading the data. See :doc:`sync`.

Such protocols must account for block scheduling: blocks that spin waiting for
unscheduled blocks can prevent those blocks from running and deadlock. For
grid-wide phase boundaries, separate kernel launches on the same stream are
another option.

Thread-Block Clusters
---------------------

On SM90 and newer GPUs, a launch can group blocks into clusters using the
``block_in_cluster_count`` keyword of :func:`launch`. Grid dimensions still
count blocks, and each must be divisible by the corresponding cluster dimension.

:func:`cluster_index` and :func:`cluster_count` describe the cluster grid.
:func:`block_in_cluster_index` and :func:`block_in_cluster_count` describe
blocks within a cluster. Cluster operations expose additional communication
and synchronization mechanisms, including mapped shared memory and cluster
barriers. See :doc:`sync` for memory visibility and
synchronization scope.
