.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

Tile Programming Guide
======================

cuTile is a parallel programming model for NVIDIA GPUs and a Python-based
:abbr:`DSL (Domain-Specific Language)`, available through ``cuda.tile``.
It lets you express computations as operations on multidimensional tiles while
the compiler manages thread mapping, synchronization within a block, and use of
hardware features such as Tensor Cores and the Tensor Memory Accelerator (TMA).
This architecture-agnostic model provides portability across supported NVIDIA
GPU architectures, letting the compiler adapt execution to each target.

cuTile |kernels| are GPU programs executed in parallel on a logical |grid| of
|blocks|. The :py:class:`@ct.kernel <cuda.tile.kernel>` decorator marks a Python
function as a kernel's entry point. Host code queues a kernel for execution
with :py:func:`ct.launch() <cuda.tile.launch>`.

|Kernels| move data between |arrays| and |tiles| using functions such as
:py:func:`ct.load() <cuda.tile.load>` and
:py:func:`ct.store() <cuda.tile.store>`. Both arrays and tiles have a shape
and a |dtype|, but they serve different roles:

* |Arrays| refer to mutable data in global memory with a physical, strided
  layout. Inside a tile kernel, array operations primarily load data into tiles
  and store tiles back to memory. Host objects such as PyTorch tensors and CuPy
  arrays can be passed as kernel arguments.
* |Tiles| are immutable values without defined storage that exist only in
  kernel code. Their dimensions must be compile-time constants that are powers
  of two. Tiles support elementwise arithmetic, matrix multiplication,
  reductions, shape manipulation, and other :doc:`operations </tile/operations>`.

The :doc:`tile quickstart </getting-started/tile>` presents a complete vector-add
kernel that handles a partial final tile. The paired
:doc:`SIMT quickstart </getting-started/lang>` expresses the same computation
with work assigned to individual threads.

.. toctree::
   :maxdepth: 1

   execution
   data
   memory_model
   interoperability
   performance
   compilation
   debugging
   known_issues
