.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _lang-sync:

Thread Synchronization
======================

``cuda.lang`` provides barriers and atomics for coordinating memory
accesses between threads.
For example, a block barrier can order shared-memory writes before other
threads in the block read them, while device-scope release/acquire atomics can
coordinate access to global memory across blocks.

For memory spaces, allocation lifetimes, and array and pointer operations,
see :doc:`data`.

.. _lang-sync-block-sums:

Coordinating Threads within a Block
-----------------------------------

A common pattern has three phases: threads write shared data, synchronize,
then read data written by other threads. A full-block
:py:obj:`cl.barrier_sync_block() <cuda.lang.barrier_sync_block>` waits for the block's threads and orders shared
memory accesses across that boundary. :py:obj:`cl.syncthreads() <cuda.lang.syncthreads>` is the CUDA-style
convenience wrapper for a full-block barrier that every thread must reach at
the same call site. A conditional around ``syncthreads()`` must therefore be
uniform across the block. In the example below, the barrier follows the
per-thread bounds checks and is outside their conditional.

This example computes one sum per 128-thread block. Each thread stages one
value, with zeros for the partial final block. After the barrier, thread zero
reads the shared values and writes that block's sum:

.. testcode:: simt-block-sums
   :template: setup_only.py

   BLOCK_SIZE = 128

   @cl.kernel
   def block_sums(values, sums):
       scratch = cl.shared_array(BLOCK_SIZE, cl.float32)
       tx = cl.thread_index(0)
       block = cl.block_index(0)
       i = block * BLOCK_SIZE + tx

       if i < values.shape[0]:
           scratch[tx] = values[i]
       else:
           scratch[tx] = 0.0
       cl.barrier_sync_block()

       if tx == 0:
           total = 0.0
           for j in range(BLOCK_SIZE):
               total += scratch[j]
           sums[block] = total

   n = 1000
   values = torch.arange(n, dtype=torch.float32, device="cuda:0")
   blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
   sums = torch.empty(blocks, dtype=torch.float32, device="cuda:0")
   cl.launch(stream, (blocks,), (BLOCK_SIZE,), block_sums, (values, sums))
   torch.cuda.synchronize()
   expected = torch.stack([values[i:i + BLOCK_SIZE].sum()
                           for i in range(0, n, BLOCK_SIZE)])
   torch.testing.assert_close(sums, expected)
   print("Block sums passed.")

.. testoutput:: simt-block-sums

   Block sums passed.

The launch uses exactly ``BLOCK_SIZE`` threads per block, matching the shared
allocation. All threads participate in the barrier, including those that load
zeros. This serial accumulation is intended to illustrate coordination, rather
than provide an optimized reduction.

Atomics, Memory Order, and Scope
--------------------------------

When several threads update one location, an ordinary load followed by a store
can lose updates. :func:`atomic_rmw` performs a read-modify-write atomically and
returns the old value. :func:`atomic_load` and :func:`atomic_store` provide
atomic reads and writes through typed pointers.

Atomicity applies to the accessed value. Ordering of other accesses depends on
the chosen memory order and synchronization protocol; :py:obj:`cl.atomic_rmw <cuda.lang.atomic_rmw>` defaults
to acquire-release ordering and device scope. The two settings describe
different aspects of communication:

* :py:obj:`cl.MemoryOrder.RELAXED <cuda.lang.MemoryOrder>` provides atomicity without ordering unrelated
  accesses. Acquire and release ordering can establish visibility of other
  accesses through a matching communication operation.
* :py:obj:`cl.MemoryScope <cuda.lang.MemoryScope>` bounds the synchronization guarantees: block, cluster,
  device, or system. Communicating operations need compatible scopes that cover
  the communicating threads, and the operation and memory space must support
  those scopes.

For example, a producer can write a payload and then release-store a ready
flag. A consumer that acquire-loads the published flag value can then read
that payload. Both operations must access the same flag with scopes covering
both threads. The payload must remain unchanged while
the consumer reads it. See the `PTX memory synchronization rules
<https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-synchronization>`_.

For communication through one block's shared memory, block scope matches the
participants. The :ref:`normalization <lang-sync-normalization>` example
combines relaxed atomic additions with barriers to separate initialization,
accumulation, and reading.

.. _lang-sync-normalization:

Example: Normalization
----------------------

This example combines thread indexing from :doc:`execution` with block barriers
and relaxed atomic additions. It normalizes sixteen positive values by their sum.
The host launches exactly one block of four threads, with four values owned
by each thread. The total must be nonzero.

Each thread loads four values from global memory into a vector, atomically adds its
partial sum into one shared-memory location, then divides its loaded values by
the block sum and stores the normalized values back to global memory:

.. testcode:: simt-normalization
   :template: setup_only.py

   @cl.kernel
   def normalize_kernel(values, normalized):
       total = cl.shared_array(1, cl.float32)
       tx = cl.thread_index(0)
       offset = tx * 4

       if tx == 0:
           total[0] = 0.0
       cl.barrier_sync_block()

       loaded = values.pointer(offset).load(count=4, alignment=16)
       partial_sum = loaded[0] + loaded[1] + loaded[2] + loaded[3]
       cl.atomic_rmw(cl.AtomicOp.ADD, total.pointer(0), partial_sum,
                     memory_order=cl.MemoryOrder.RELAXED,
                     memory_scope=cl.MemoryScope.BLOCK)
       cl.barrier_sync_block()

       loaded = loaded / total[0]
       normalized[offset + 0] = loaded[0]
       normalized[offset + 1] = loaded[1]
       normalized[offset + 2] = loaded[2]
       normalized[offset + 3] = loaded[3]

   values = torch.tensor(
       [
           1.0, 1.0, 2.0, 4.0,
           1.0, 1.0, 2.0, 4.0,
           1.0, 1.0, 2.0, 4.0,
           1.0, 1.0, 2.0, 4.0,
       ],
       dtype=torch.float32,
       device="cuda:0",
   )
   normalized = torch.empty_like(values)
   cl.launch(stream,
             (1,),  # grid size
             (4,),  # block size
             normalize_kernel, (values, normalized))
   torch.cuda.synchronize()
   torch.testing.assert_close(normalized, values / values.sum())
   print(normalized.cpu().tolist())

.. testoutput:: simt-normalization

   [0.03125, 0.03125, 0.0625, 0.125, 0.03125, 0.03125, 0.0625, 0.125, 0.03125, 0.03125, 0.0625, 0.125, 0.03125, 0.03125, 0.0625, 0.125]


Thread Ownership and Vector Loads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thread ``tx`` owns elements ``tx * 4`` through ``tx * 4 + 3``. Each thread's
pointer load returns a four-element vector, so the four threads together
cover all sixteen inputs. Each vector belongs to one thread. The compiler
chooses the hardware instructions used for the load.

``alignment=16`` promises that each load address is aligned to sixteen bytes.
The input allocation is aligned, and each thread advances by four ``float32``
elements, preserving that alignment. See :ref:`data-pointers` for the pointer
interface.

Initialization and the First Barrier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``total`` is one shared-memory location per block. Thread zero initializes it;
the other threads must wait before updating it. The first
:py:obj:`cl.barrier_sync_block() <cuda.lang.barrier_sync_block>` makes that initialization available to every
thread before any atomic addition.

Accumulation and the Second Barrier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each thread computes a private partial sum, then atomically adds it to
``total[0]``. The atomic prevents concurrent additions from losing updates.
Its block scope matches the threads sharing this allocation. Relaxed ordering
is sufficient for these additions because the surrounding block barriers
provide the ordering between initialization, accumulation, and reading.

The second barrier ensures all additions have completed before any thread reads the
final total. Both barriers are outside the ``if tx == 0`` branch, so all four
threads participate.

Each thread then divides its private vector by the same total and writes its
four output elements with ordinary stores to disjoint regions. Floating-point atomic additions may occur in different orders;
for general inputs, these orders can produce different floating-point
rounding results.

Scope of the Example
~~~~~~~~~~~~~~~~~~~~

This kernel is written for exactly one block of four threads and contiguous
sixteen-element ``float32`` input and output arrays, with the input aligned as
described above. Increasing the grid size would create an independent
``total`` in each block, but every block would still access the same sixteen
elements. The output writes would overlap
and race. Larger inputs need a different work partition, such as per-block
partial sums followed by another kernel.
The :ref:`block-sum example <lang-sync-block-sums>` shows how to handle a
partial final block while keeping all threads involved in synchronization.

Barriers
--------

Barrier APIs differ in their participating threads:

* :func:`barrier_sync_warp` coordinates the warp lanes selected by its mask.
* :func:`barrier_sync_block` coordinates threads in a block. Its
  participation-count and barrier-id options support partial and named barriers.
* :func:`barrier_sync_cluster` coordinates blocks in a thread-block cluster
  on supporting GPUs.

Named block barriers have IDs 0 through 15, local to each block. An explicit
``number_of_threads`` counts participating threads and must be a positive
multiple of 32. Participation is accounted for at warp granularity. With no
explicit count, the barrier covers the block, including a partial final warp.
See the `PTX block barrier rules
<https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-barrier>`_.

TMA and Tensor Core operations have operation-specific completion and ordering
rules, described in :doc:`tma` and :doc:`tcgen05`.

For example, to read shared data from other blocks in a cluster, each block
first initializes its local shared array. All threads in the cluster then call
:py:obj:`cl.barrier_sync_cluster() <cuda.lang.barrier_sync_cluster>` so those writes are visible and all blocks have
started before remote reads begin. Threads use :func:`map_shared_to_cluster`
to obtain pointers into other blocks' arrays and read the data. After the
reads, all threads call :py:obj:`cl.barrier_sync_cluster() <cuda.lang.barrier_sync_cluster>` again, before any block
overwrites the data or exits. The second barrier prevents a block from
invalidating storage that another block is still reading.

For synchronization with asynchronous copies, see :doc:`tma`.
