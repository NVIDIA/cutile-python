.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _lang-tma:

TMA and Asynchronous Barriers
=============================

The Tensor Memory Accelerator (TMA) is available on NVIDIA Hopper and newer GPU
architectures, starting with compute capability 9.0. It transfers data
asynchronously in both directions between global and shared memory, allowing
kernels to overlap data movement with computation.

This chapter starts with a global-to-shared transfer into one shared-memory
buffer per block, building on the synchronization concepts in :doc:`sync`.
See the
`CUDA asynchronous-copy guide <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html>`_
for hardware support and alignment requirements.

Tensor Maps and Mbarriers
-------------------------

Two objects describe different parts of the operation:

* A :class:`TensorMap` describes the global array's layout and the shape of the
  region to copy. :func:`tensor_map_tiled` creates it on the host, and the
  descriptor is passed to the kernel. It encodes the source address and layout.
  The allocation must remain alive, and the source region must remain
  unchanged while a transfer reads it. See
  :ref:`data-tensor-maps` for the descriptor interface.
* An `mbarrier
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier>`_
  is a shared-memory synchronization object that tracks thread arrivals and
  outstanding transactions. For the TMA operation used here,
  transactions are counted in bytes. A global-to-shared
  TMA copy reports completion to this object. Threads wait for the expected
  arrivals and transactions to complete before reading the destination.

A block barrier waits for participating threads. An mbarrier can also account
for an asynchronous transfer that those threads have submitted.

Mbarrier Phases
---------------

The examples use the default version-0 mbarrier layout and disable copy-error
reporting. The phase rules below describe that layout.

A *phase* is one round of synchronization using an mbarrier. Initialization
starts the first phase with an expected arrival count. A phase completes when
both its pending arrivals and its outstanding transactions reach zero. The
mbarrier then automatically starts the next phase, resetting the pending
arrival count to the expected count. Transaction bytes must be registered for
each phase that tracks a transfer.

The *parity* identifies whether a phase is even or odd: it starts at zero and
alternates between zero and one as phases complete. A parity wait specifies
which phase's completion the thread is waiting for. The example below uses
only the first phase of each block's mbarrier, so it waits with parity zero.

A Complete Copy Example
-----------------------

Save the following example as ``tma_copy.py`` and run ``python tma_copy.py`` on
a supported GPU. Each block copies one column of an eight-row, thirty-two-column
:data:`float32` array through its own shared-memory buffer. Thread zero submits
the transfer; all eight threads wait for completion, then each writes one
element of that column to the output using an ordinary store. The final output
is checked against the input.

.. testcode:: tma-copy
   :skipif: torch.cuda.get_device_capability()[0] < 9

   import cuda.lang as cl
   import torch

   ROWS = 8
   COLS = 32

   @cl.kernel
   def tma_copy(source_map, output):
       staged = cl.shared_array(ROWS, cl.float32, alignment=128)
       ready = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
       tx = cl.thread_index(0)
       col = cl.block_index(0)

       if tx == 0:
           cl.mbarrier_initialize(ready, 1)
           cl.mbarrier_expect_transaction(ready, ROWS * 4)
       cl.barrier_sync_block()

       if tx == 0:
           cl.copy_async_bulk_tensor_global_to_shared(
               source_map, (0, col), staged.pointer(), ready
           )
           cl.mbarrier_arrive(ready)

       cl.mbarrier_wait_parity(ready, 0)
       output[tx, col] = staged[tx]

   if torch.cuda.get_device_capability()[0] < 9:
       raise RuntimeError("This example requires a GPU with TMA support (SM90+).")

   source = torch.arange(ROWS * COLS, dtype=torch.float32,
                         device="cuda:0").reshape(COLS, ROWS).t()
   output = torch.empty_like(source)
   source_map = cl.tensor_map_tiled(source, (ROWS, 1), order=cl.AxisOrder.KEEP)
   stream = torch.cuda.current_stream()
   cl.launch(stream, (COLS,), (ROWS,), tma_copy, (source_map, output))
   torch.cuda.synchronize()
   torch.testing.assert_close(output, source)
   print("TMA copy passed.")

.. testoutput:: tma-copy
   :skipif: torch.cuda.get_device_capability()[0] < 9

   TMA copy passed.

Describe the Layout
-------------------

The source has column-major storage: reshaping to ``(COLS, ROWS)`` and
transposing gives a ``(ROWS, COLS)`` view with element strides ``(1, ROWS)``.
Each column is contiguous in memory. TMA descriptor axis zero must describe
the innermost, contiguous dimension, which here is the row axis.
:class:`AxisOrder.KEEP <AxisOrder>` keeps descriptor axes in the same order as
array axes:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Array order
     - Descriptor order
   * - Axes
     - Row, column
     - Row, column
   * - Extents
     - ``(ROWS, COLS)``
     - ``(ROWS, COLS)``
   * - Copy shape
     - ``(ROWS, 1)``
     - ``(ROWS, 1)``
   * - Copy origin
     - Row 0, column ``col``
     - ``(0, col)``

Copy coordinates specify the origin in elements, using descriptor axis order.
Block ``col`` selects its column with origin ``(0, col)`` and copies
``ROWS`` elements into ``staged``. With swizzling disabled, those elements are
packed along the descriptor's innermost dimension, so ``staged[tx]`` corresponds
to ``source[tx, col]``.

The shared buffer and mbarrier use :func:`shared_array` with 128-byte and
8-byte alignment, respectively. The input allocation is aligned, its column
stride is 32 bytes, and each transfer contains 32 bytes. These choices satisfy the basic
non-swizzled tensor-copy alignment requirements. Different layouts and swizzle
modes need their own alignment checks; see :func:`tensor_map_tiled`.

Column-major storage also makes the copy's innermost extent eight ``float32``
elements, or 32 bytes. A single column in a row-major array would have an
innermost extent of one ``float32`` element, which does not meet the
16-byte requirement for a noninterleaved TMA box.

Copy Protocol
-------------

The example uses one mbarrier phase per block:

1. Thread zero calls :func:`mbarrier_initialize` on ``ready`` with one expected
   arrival, then :func:`mbarrier_expect_transaction` to register ``ROWS * 4``
   transaction bytes for the column's eight ``float32`` elements.
2. :func:`barrier_sync_block` publishes this setup before other threads use
   ``ready``.
   This follows the `PTX mbarrier initialization sequence
   <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init>`_.
3. Thread zero submits :func:`copy_async_bulk_tensor_global_to_shared`, then
   calls :func:`mbarrier_arrive` with its default release ordering. TMA reports
   the transaction bytes as the transfer completes.
4. All eight threads call :func:`mbarrier_wait_parity` with parity zero and
   acquire ordering before reading ``staged``.

In this protocol, transaction bytes are registered while an arrival is still
pending, preventing premature phase completion. A phase tracking several
transfers counts their total bytes and the arrivals actually issued.

Reuse and Overlap
-----------------

Each block uses one phase and immediately waits for completion. To overlap
data movement with computation, a pipelined kernel can submit a future tile
while consuming an earlier one, with separate buffers and completion tracking.

Successive phases of a reused barrier have alternating parity. A parity wait
may refer only to the current incomplete phase or the immediately preceding
completed phase. At least
one wait or test must observe completion of each phase before any arrival in
the next phase.
See the `PTX mbarrier lifecycle and phase rules
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier>`_.

A shared buffer can be overwritten after its consumers have finished reading
it. Repurposing an mbarrier's storage requires all thread and asynchronous uses
to finish, followed by :func:`mbarrier_invalidate` before the storage is reused.

Shared-to-global TMA copies use a different completion mechanism: bulk
async-groups. See :func:`copy_async_bulk_tensor_shared_to_global`,
:func:`copy_async_bulk_commit_group`, and :func:`copy_async_bulk_wait_group`.
Their proxy ordering and completion rules are described under
:ref:`lang-tma-proxy-fences`.


.. _lang-tma-proxy-fences:

Proxy Fences
------------

Memory Proxies
~~~~~~~~~~~~~~

A *memory proxy* labels the method an instruction uses to access memory.
Two instructions can access the same memory location through different
proxies. The proxies relevant to TMA are:

* The *generic proxy*, used by ordinary thread loads, stores, and atomics.
* The *async proxy*, used by TMA to read and write the data being copied.
* The *tensormap proxy*, used by TMA to read the tensor-map descriptor that
  describes the transfer.

PTX also distinguishes texture and surface accesses, fabric and read-only
accesses, and accesses through different virtual addresses that alias the same
memory. See the `PTX specification of proxies
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies>`_
for the different access methods and their ordering requirements.

A *proxy fence* establishes ordering between accesses through different
proxies. A thread barrier coordinates participating threads; communication
across proxies also needs the applicable proxy-ordering guarantees.

Why TMA Needs a Proxy Fence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The synchronization depends on the direction of the copy:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Access sequence
     - Required synchronization
   * - Generic shared-memory stores, then TMA reads
     - A generic/async proxy fence before the copy. When multiple
       threads write the buffer, also coordinate them before submission.
   * - TMA writes, then generic shared-memory reads
     - Observe transfer completion with an acquire mbarrier wait.

TMA bulk-copy completion includes an implicit proxy fence that makes the
results visible to the generic proxy once completion is observed. The
global-to-shared example observes completion by waiting on ``ready`` before
reading ``staged``. See the `PTX async-proxy completion rules
<https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy>`_.

Making Thread Stores Visible to TMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save the following example as ``tma_store.py`` and run it on a GPU with TMA
support. It writes one column per block using the same layout and launch
dimensions as the earlier example:

.. testcode:: tma-store
   :skipif: torch.cuda.get_device_capability()[0] < 9

   import cuda.lang as cl
   import torch

   ROWS = 8
   COLS = 32

   @cl.kernel
   def store_column(output_map):
       staged = cl.shared_array(ROWS, cl.float32, alignment=128)
       row = cl.thread_index(0)
       col = cl.block_index(0)

       staged[row] = cl.float32(col * ROWS + row)
       cl.fence_proxy_bidirectional(
           cl.FenceProxy.ASYNC,
           restriction=cl.FenceRestriction.shared_block(),
       )
       cl.barrier_sync_block()

       if row == 0:
           cl.copy_async_bulk_tensor_shared_to_global(
               staged.pointer(), output_map, (0, col)
           )
           cl.copy_async_bulk_commit_group()
           cl.copy_async_bulk_wait_group(0, read=True)

   if torch.cuda.get_device_capability()[0] < 9:
       raise RuntimeError("This example requires a GPU with TMA support (SM90+).")

   output = torch.full((COLS, ROWS), -1.0, dtype=torch.float32,
                        device="cuda:0").t()
   output_map = cl.tensor_map_tiled(output, (ROWS, 1), order=cl.AxisOrder.KEEP)
   stream = torch.cuda.current_stream()
   cl.launch(stream, (COLS,), (ROWS,), store_column, (output_map,))
   torch.cuda.synchronize()
   expected = torch.arange(ROWS * COLS, dtype=torch.float32,
                           device="cuda:0").reshape(COLS, ROWS).t()
   torch.testing.assert_close(output, expected)
   print("TMA store passed.")

.. testoutput:: tma-store
   :skipif: torch.cuda.get_device_capability()[0] < 9

   TMA store passed.

Each writer calls :func:`fence_proxy_bidirectional` after its store. With
:attr:`FenceProxy.ASYNC` and :meth:`FenceRestriction.shared_block`, this emits
``fence.proxy.async.shared::cta``. It orders accesses between the generic and
async proxies for the current block's shared memory. The ``shared::cta`` part
selects the memory space.

:func:`barrier_sync_block` ensures every writer has reached the fence before
thread zero submits the copy. Together, the bidirectional proxy fence and block
barrier order the stores before TMA reads across threads. Split release/acquire
proxy fences exposed by :func:`fence` require a matching synchronization
protocol. See the `PTX proxy-fence rules
<https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar-fence>`_.

Thread zero calls :func:`copy_async_bulk_commit_group`, then
:func:`copy_async_bulk_wait_group` with ``0`` and ``read=True``. This waits for
all its committed copies to finish reading their shared-memory sources and
tensor-map descriptors, so the shared buffer can be released when the block
exits. Global-memory writes may still be pending when the wait returns; host
synchronization waits for kernel completion before validating the output.
The `CUDA asynchronous-copy example
<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-tma-to-transfer-one-dimensional-arrays>`_
uses the same read-only wait.

With the default ``read=False``, the wait also establishes that destination
writes have completed and are visible to the issuing thread. Other threads
need subsequent synchronization before consuming those writes.
These are per-thread groups: the thread that submits and commits the copies
must perform the wait. See the `PTX bulk async-group wait
rules
<https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group>`_.
