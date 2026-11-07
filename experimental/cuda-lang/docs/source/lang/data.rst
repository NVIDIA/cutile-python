.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _lang-data-data-model:


Data Model
==========

|cuda lang| is a low-level CUDA programming model. Its fundamental data
structures are multidimensional arrays, typed pointers, and tensor maps.

An array-based interface is convenient for structured data because arrays know
their element type, shape, and layout. Pointers are also exposed because
|cuda lang| is designed for SIMT code that may need explicit address-space
casts, vectorized memory operations, or low-level APIs.

Tensor maps are exposed for Tensor Memory Accelerator (TMA) operations that use
hardware descriptors to move multidimensional regions of global memory.

For how threads share these allocations and coordinate accesses, see
:doc:`sync`.

Memory Spaces and Lifetimes
---------------------------

Within |SIMT code|, arrays and pointers are associated with a CUDA memory space,
represented by the :class:`MemorySpace` enum:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Space
     - Accessibility
     - Allocation and lifetime
   * - Global
     - Accessible to threads across blocks.
     - Pass a GPU array into the kernel and keep its underlying allocation
       alive until all GPU operations using it finish.
   * - Shared
     - Accessible within the owning thread block; on supporting GPUs, pointers
       can also be mapped to other blocks in the same cluster.
     - Allocate with :py:obj:`cl.shared_array <cuda.lang.shared_array>`. Each
       shared array has one allocation per thread block, with the lifetime of
       that block.
   * - Local
     - Private to a single thread.
     - Allocate with ``with cl.local_array(...) as scratch:``; storage is valid
       within that context.


.. _data-arrays:

Arrays
------

A |cuda lang| :class:`Array` is an N-dimensional view of elements stored in a
CUDA memory space.

Global arrays are typically allocated by host code and passed to a kernel as
arguments. The kernel can index the array, take pointers to its elements, and
create views that refer to the same underlying data.

Scalar indexing in SIMT code requires in-bounds indices. The examples in
:doc:`execution` use explicit bounds checks to guard array accesses.

:func:`local_array` provides storage private to a thread. Its values can often
remain in registers, especially when indices are known at compile time, so
using it does not necessarily cause local-memory traffic. Local arrays require
a compile-time constant shape.

Shared-array contents are uninitialized until explicitly written. Shared-arrays
cannot be allocated inside runtime control flow. They can follow other
statements, and static shared arrays can be created in helpers called outside
runtime control flow. They are also allowed in branches selected by
compile-time conditions.

By default, :py:func:`cl.shared_array <cuda.lang.shared_array>` requires a
compile-time constant shape, and the compiler fixes the allocation size per
block. With ``dynamic=True``, :py:func:`cl.launch <cuda.lang.launch>` computes
the required shared memory per block at launch time. Shape dimensions can be
constants or direct references to ``int32`` kernel parameters.

The following example uses a global array for the kernel result, a shared array
for communication between threads in the block, and a local array for per-thread
temporary storage:

.. testcode::
   :template: setup_only.py

   @cl.kernel
   def kernel(out):
       smem = cl.shared_array(2, cl.int32)
       tx = cl.thread_index(0)

       if tx == 0:
           with cl.local_array(shape=(1,), dtype=cl.int32) as tmp:
               tmp[0] = 7
               smem[0] = tmp[0]
       if tx == 1:
           smem[1] = 35

       cl.barrier_sync_block()

       if tx == 0:
           out[0] = smem[0] + smem[1]

   out = torch.empty(1, dtype=torch.int32, device="cuda:0")
   cl.launch(stream, (1,), (2,), kernel, (out,))
   torch.cuda.synchronize()
   print(out.cpu().item())

.. testoutput::

   42

.. seealso::
  :ref:`cuda.lang.Array class documentation <data-array-cuda-lang-array>`

.. _data-pointers:

Pointers
--------

A :class:`Pointer` is a typed address into a CUDA memory space. Pointers provide
low-level load and store operations, including vector loads and stores.
:meth:`Array.pointer` creates a pointer to an element of an array.
With no argument, ``array.pointer()`` returns a pointer to the first element.
A scalar index or tuple of scalar indices returns a pointer to an element,
as in ``array.pointer(i)`` or ``array.pointer((i, j))``.

Pointer dtypes encode both the pointee type and the memory space.
:func:`pointer_dtype` constructs typed pointer dtypes, and
:func:`opaque_pointer_dtype` constructs opaque pointer dtypes.
:func:`address_space_cast` casts a pointer to another memory space.
:func:`map_shared_to_cluster` returns a pointer to the corresponding shared
allocation in another block of the same cluster. That block must remain alive
until remote accesses finish; see :doc:`sync` for the cluster-barrier protocol.

An address-space cast interprets an existing address in the requested space.
The address must refer to storage in that space.

The following example takes a shared array's base pointer and performs a
vectorized load:

.. testcode::
   :template: setup_only.py

   @cl.kernel
   def kernel(out):
       values = cl.shared_array(4, cl.int32, alignment=16)
       tx = cl.thread_index(0)

       values[tx] = tx + 1
       cl.barrier_sync_block()

       if tx == 0:
           ptr = values.pointer()
           vec = ptr.load(count=4, alignment=16)
           out[0] = vec[0] + vec[1] + vec[2] + vec[3]

   out = torch.empty(1, dtype=torch.int32, device="cuda:0")
   cl.launch(stream, (1,), (4,), kernel, (out,))
   torch.cuda.synchronize()
   print(out.cpu().item())

.. testoutput::

   10

``alignment=16`` promises that the address is aligned to sixteen bytes. Here,
the shared allocation explicitly requests that alignment. All four elements
must also be accessible. The compiler chooses the hardware
instructions used for the vector load.

.. seealso::
  :ref:`cuda.lang.Pointer class documentation <data-pointer-cuda-lang-pointer>`

.. _data-vectors:

Vectors
-------

:meth:`Pointer.load` returns a :class:`Vector` when
``count`` is greater than ``1``. The default count of ``1`` returns one value.

For packed FP4 vectors, see :ref:`data-packed-fp4-storage`. Writing a modified
vector back to its source memory requires an explicit store.

.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

   VectorReduction

.. seealso::
  :ref:`Vector operations, indexing, and instruction generation <data-vector-cuda-lang-vector>`

.. _data-tensor-maps:

Tensor Maps
-----------

For a complete copy example and its synchronization protocol, see
:doc:`tma`.

A tensor-map descriptor describes how a multidimensional global array is accessed
by TMA operations. It captures the array's element type, logical shape, memory
layout, tile shape, interleave mode, and swizzle mode in a descriptor that can
be passed to low-level TMA intrinsics.

:func:`tensor_map_tiled` creates a tensor map from a global :class:`Array` in
ordinary Python or a :func:`host_entry`. The resulting immutable
:class:`TensorMap` value is accepted in the ``kernel_args`` tuple of
:func:`launch`. Inside a kernel, the value is a pointer to a
``tensor_map_descriptor``.

.. code-block:: python

   @cl.kernel
   def copy_kernel(tensor_map_ptr, output):
       # Use tensor_map_ptr with a TMA operation.
       ...

   tensor_map = cl.tensor_map_tiled(source, (32, 16), order=cl.AxisOrder.REVERSE)
   cl.launch(stream, grid, block, copy_kernel, (tensor_map, output))

Only tiled tensor map mode is supported today. Other TMA descriptor modes are
reserved for future support.

The returned value contains the descriptor bytes. The caller must keep the memory
referenced by the descriptor alive until all GPU operations using it have completed.
Kernel launches pass the descriptor bytes as a grid-constant kernel parameter.
The kernel receives a pointer with pointee dtype ``tensor_map_descriptor``.

.. seealso::
  :ref:`TensorMap operations <operations-tensor-map>`

.. toctree::
   :maxdepth: 2
   :hidden:

   /data/array
   /data/pointer
   /data/vector
   /data/tensor_map


.. _lang-data-data-types:

Data Types
----------

|cuda lang| and |cuda.tile| share the numeric data types and arithmetic
promotion rules described in :ref:`language-numeric-types`.

:func:`dtype_of` returns the data type of a scalar, pointer, or vector.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   dtype_of
