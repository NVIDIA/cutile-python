.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

SIMT Programming Guide
======================

``cuda.lang`` is in beta; its API is subject to change.

``cuda.lang`` exposes CUDA's low level SIMT execution model in Python. Each
thread executes the kernel body, computes its own indices, and accesses memory
explicitly. You choose how threads divide the work and communicate, including
when to use shared memory, atomics, and synchronization.

Its low-level APIs expose hardware features such as TMA transfers and Tensor
Core instructions on GPU architectures that support them.

The :py:class:`@cl.kernel <cuda.lang.kernel>` decorator marks a Python function
as a kernel's entry point. Host code queues it for execution with
:py:func:`cl.launch() <cuda.lang.launch>`, specifying both the grid size in
blocks and the block size in threads.

The :doc:`SIMT </getting-started/lang>` and
:doc:`tile </getting-started/tile>` quickstarts present the same vector-add
computation in both programming models.
The shared :doc:`/language` covers Python semantics and
compile-time programming; :doc:`/operations` provides the API catalog.

.. toctree::
   :maxdepth: 1

   execution
   data
   sync
   tma
   tcgen05
   warp-specialization
   debugging
