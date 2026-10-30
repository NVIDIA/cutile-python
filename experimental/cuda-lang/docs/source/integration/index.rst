.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

Integration and Tooling
=======================

Third-Party Arrays and Streams
------------------------------

Both ``cuda.lang`` and ``cuda.tile`` accept GPU arrays from other Python
libraries directly as kernel arguments. The kernel receives an array view of
the existing allocation, including its dtype, shape, and strides.

Supported Arrays
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Array object
     - Support
   * - PyTorch CUDA tensors
     - GPU ``torch.Tensor`` objects are accepted directly.
   * - CuPy GPU arrays
     - ``cupy.ndarray`` objects are accepted through their array protocols.
   * - DLPack-compatible GPU objects
     - Objects implementing ``__dlpack__`` can expose their GPU allocation
       and array metadata through the DLPack protocol.
   * - CUDA Array Interface objects
     - Objects exposing ``__cuda_array_interface__``, such as Numba CUDA
       device arrays, can supply their device pointer and array metadata.

The array's dtype and layout must be supported by the kernel's operations.
Views retain the producer's shape and strides. See :doc:`/lang/data` for the SIMT array interface and
:doc:`/tile/data` for the tile array interface.

Supported Streams
~~~~~~~~~~~~~~~~~

The first argument to :py:obj:`cl.launch <cuda.lang.launch>` or
:py:obj:`ct.launch <cuda.tile.launch>` selects the stream on which work is
enqueued. Supported stream arguments are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Source
     - Stream argument
   * - PyTorch
     - A ``torch.cuda.Stream`` object, including
       ``torch.cuda.current_stream()``.
   * - CuPy
     - A ``cupy.cuda.Stream`` object, including
       ``cupy.cuda.get_current_stream()``.
   * - Numba CUDA
     - A stream object returned by ``numba.cuda.stream()``.
   * - Raw CUDA handle
     - A Python integer containing a ``CUstream`` handle, such as
       ``stream.cuda_stream`` for PyTorch or ``stream.ptr`` for CuPy.

Stream handles obtained through CUDA Python bindings or other CUDA APIs can
be converted to Python integers, for example with ``int(handle)``.
Arrays and the launch stream must belong to a compatible CUDA device/context.

The array allocations and stream must remain alive until GPU work using them
completes. Communication between a producer and a kernel on different streams
requires explicit CUDA stream dependencies.
The :doc:`SIMT </getting-started/lang>` and :doc:`tile </getting-started/tile>`
quickstarts show complete CuPy examples.

Debugging and Profiling
-----------------------

The quickstarts compare kernel results with framework implementations.
Synchronization before the comparison makes asynchronous GPU errors visible
before validation.

NVIDIA Nsight Compute can profile GPU kernels from either model. To profile a
saved quickstart script, run:

.. code-block:: bash

   ncu -o vector_add_profile python vector_add_tile.py
