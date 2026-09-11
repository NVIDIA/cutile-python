.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _data-data-model:


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

Within |SIMT code|, arrays and pointers are associated with a CUDA
:class:`memory space <MemorySpace>`.
Memory spaces are represented by the :class:`MemorySpace` enum. The most common
memory spaces are:

- local memory, which is private to a single thread;
- shared memory, which is visible to all threads in a thread block (CTA) and
  can be mapped across CTAs in a cluster; and
- global memory, which is visible to all CTAs in a grid.


.. _data-arrays:

Arrays
------

A |cuda lang| :class:`Array` is an N-dimensional view of elements stored in a
CUDA memory space.

Global arrays are typically allocated by host code and passed to a kernel as
arguments. The kernel can index the array, take pointers to its elements, and
create views that refer to the same underlying data.

Local arrays are allocated per thread with :func:`local_array`. They must be
created in a ``with`` statement and have static shape. The local memory is valid
only within the ``with`` block.

Shared arrays are allocated with :func:`shared_array` and live until the end of
the kernel. Because shared array storage is not scoped to the call that creates
it, shared arrays must be declared at the beginning of the kernel and cannot be
declared inside control flow or helper functions.

Static shared arrays are allocated in statically allocated shared memory and
require a compile-time constant shape. Dynamic shared arrays are created with
``dynamic=True`` and are allocated in dynamically allocated shared memory. Their
shape may be dynamic, but currently the dynamic shape expression can reference
only an integer kernel parameter directly.

The following example uses a global array for the kernel result, a shared array
for communication between threads in the CTA, and a local array for per-thread
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
low-level load and store operations, including vector loads and stores. Use
:meth:`Array.pointer` to create a pointer to an element of an array.
With no argument, ``array.pointer()`` returns a pointer to the first element.
A scalar index or tuple of scalar indices returns a pointer to an element,
as in ``array.pointer(i)`` or ``array.pointer((i, j))``.

Pointer dtypes encode both the pointee type and the memory space. Use
:func:`pointer_dtype` to construct typed pointer dtypes, use
:func:`opaque_pointer_dtype` for opaque pointer dtypes, use
:func:`address_space_cast` to cast a pointer to another memory space, and use
:func:`map_shared_to_cluster` to map a shared-memory pointer from another CTA in
the same cluster.

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

.. seealso::
  :ref:`cuda.lang.Pointer class documentation <data-pointer-cuda-lang-pointer>`

.. _data-vectors:

Vectors
-------

A :class:`Vector` is a fixed-size collection of elements used by low-level
pointer operations. For example, :meth:`Pointer.load` returns a vector when a
``count`` is provided.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

   VectorReduction

.. seealso::
  :ref:`cuda.lang.Vector class documentation <data-vector-cuda-lang-vector>`

.. _data-tensor-maps:

Tensor Maps
-----------

A tensor-map descriptor describes how a multidimensional global array is accessed
by TMA operations. It captures the array's element type, logical shape, memory
layout, tile shape, interleave mode, and swizzle mode in a descriptor that can
be passed to low-level TMA intrinsics.

Create a tensor map from a global :class:`Array` with :func:`tensor_map_tiled`.
Call it from ordinary Python or from a :func:`host_entry`, then pass the
resulting immutable :class:`TensorMap` value in the ``kernel_args`` tuple of
:func:`launch`. Inside a kernel, the value is a pointer to a
``tensor_map_descriptor``.

.. code-block:: python

   @cl.kernel
   def copy_kernel(tensor_map_ptr, output):
       # Use source_map with a TMA operation.
       ...

   tensor_map = cl.tensor_map_tiled(source, (32, 16), order="F")
   cl.launch(stream, grid, block, copy_kernel, (tensor_map, output))

Only tiled tensor map mode is supported today. Other TMA descriptor modes are
reserved for future support.

The returned value contains the descriptor bytes. The caller must keep the memory
referenced by the descriptor alive until all GPU operations using it have completed.
Kernel launches copy bytes as grid constant, and the kernel receives
a pointer with pointee dtype ``tensor_map_descriptor``.

.. seealso::
  :ref:`TensorMap operations <operations-tensor-map>`

.. toctree::
   :maxdepth: 2
   :hidden:

   data/array
   data/pointer
   data/vector
   data/tensor_map


.. _data-data-types:

Data Types
----------

|cuda lang| shares the same data types and arithmetic promotion rules as
|cuTile|.

Use :func:`dtype_of` to get the data type of a scalar, pointer, or vector.

.. autosummary::
   :toctree: generated
   :nosignatures:

   dtype_of
