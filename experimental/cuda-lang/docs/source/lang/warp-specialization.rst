.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _lang-warp-specialization:

Warp Specialization
===================

Warp specialization assigns different tasks to different warps in a thread
block. For the Gen05 GEMM in :doc:`tcgen05`, one warp can submit TMA copies while
another issues MMA instructions. Two shared-memory stages allow loading the
next operands while Gen05 Tensor Cores consume an earlier stage.

This chapter uses the same SM100 hardware, fixed matrix shapes, operand layout,
and FP32 output as the preceding example. It changes the work assignment and
buffer-reuse protocol. The kernel still launches one block of 128 threads.

Assigning Work to Warps
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Warp
     - Work
   * - Warp 0
     - Collectively allocate TMEM during setup. One elected lane then submits
       TMA copies as the producer.
   * - Warp 1
     - One elected lane waits for operands, issues MMA operations, and commits
       completion arrivals as the consumer.
   * - Warps 2 and 3
     - Wait for the main loop to finish.
   * - All four warps
     - Read and store their result rows after MMA completion. Warp 0 then
       collectively deallocates TMEM after all readers finish.

:py:obj:`cl.warp_index() <cuda.lang.warp_index>` selects a warp's role. Inside each role's warp-uniform
branch, :func:`elect_sync` selects one active lane to issue single-thread
operations. Every lane of that warp participates in the election. Allocation,
deallocation, and TMEM loads instead require their full participating warps.

A Stage Has Two Handoffs
------------------------

Each stage has its own A/B buffers and two mbarriers:

* ``loaded[stage]``: the producer registers the copy bytes and arrives; TMA
  completion makes the operands available to the consumer.
* ``reusable[stage]``: the consumer commits MMA completion to this barrier.
  Its hardware arrival tells the producer that the operand buffers can be
  overwritten for a later K slice.

The producer waits for reuse only after a stage has been used once. Initially,
each buffer is available for its first transfer.

A separate ``finished`` barrier signals completion of the accumulated result
before all four warps read TMEM.

A Complete Warp-Specialized Example
-----------------------------------

Save the following example as ``warp_specialized_gemm.py`` and run it on an
SM100 GPU.
It includes the same layout helpers and host setup as the first version and
runs independently.

.. testcode:: warp-specialized-gemm
   :skipif: torch.cuda.get_device_capability() != (10, 0)

   import cuda.lang as cl
   import torch

   M = N = 128
   K = 256
   BLOCK_K = 32
   MMA_K = 16
   THREADS = 128
   COPY_BYTES = (M + N) * BLOCK_K * 2
   TMEM_COLUMNS = N


   def operand_descriptor(pointer):
       return cl.Tcgen05SharedMemoryDescriptor(
           matrix_start_address=pointer,
           leading_dimension_byte_offset=16,
           stride_dimension_byte_offset=512,
           swizzle_mode=cl.SwizzleMode.SWIZZLE_64B,
       ).encode()


   def store_accumulators(tmem, c):
       row = cl.thread_index(0)
       warp = cl.warp_index()
       for column in cl.static_iter(range(0, N, 32)):
           address = cl.tcgen05_tmem_offset(
               tmem, lane_offset=warp * 32, column_offset=column
           )
           values = cl.tcgen05_load(
               cl.Tcgen05LoadStoreShape.SHAPE_32x32b,
               address, element_count=32, dtype=cl.float32,
           )
           cl.tcgen05_wait_load()
           c.pointer((row, column)).store(values, alignment=16)


   STAGES = 2


   @cl.kernel
   def gemm(a_map, b_map, c):
       a_smem = cl.shared_array((STAGES, M * BLOCK_K), cl.float16, alignment=512)
       b_smem = cl.shared_array((STAGES, N * BLOCK_K), cl.float16, alignment=512)
       loaded = cl.shared_array(STAGES, cl.mbarrier, alignment=8)
       reusable = cl.shared_array(STAGES, cl.mbarrier, alignment=8)
       finished = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
       tmem_slot = cl.shared_array(
           1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
       )
       tid = cl.thread_index(0)
       warp = cl.warp_index()

       if tid == 0:
           for stage in cl.static_iter(range(STAGES)):
               cl.mbarrier_initialize(loaded.pointer(stage), 1)
               cl.mbarrier_initialize(reusable.pointer(stage), 1)
           cl.mbarrier_initialize(finished, 1)
       if warp == 0:
           cl.tcgen05_allocate(tmem_slot.pointer(), TMEM_COLUMNS)
           cl.tcgen05_relinquish_allocation_permit()
       cl.barrier_sync_block()
       tmem = tmem_slot[0]

       if warp == 0 and cl.elect_sync():
           for k_tile in range(K // BLOCK_K):
               stage = k_tile % STAGES
               generation = k_tile // STAGES
               if k_tile >= STAGES:
                   cl.mbarrier_wait_parity(
                       reusable.pointer(stage), (generation - 1) % 2
                   )
               ready = loaded.pointer(stage)
               cl.mbarrier_expect_transaction(ready, COPY_BYTES)
               cl.copy_async_bulk_tensor_global_to_shared(
                   a_map, (k_tile * BLOCK_K, 0), a_smem.pointer((stage, 0)), ready
               )
               cl.copy_async_bulk_tensor_global_to_shared(
                   b_map, (k_tile * BLOCK_K, 0), b_smem.pointer((stage, 0)), ready
               )
               cl.mbarrier_arrive(ready)

       elif warp == 1 and cl.elect_sync():
           instruction = cl.Tcgen05InstructionDescriptor(
               d_type=cl.float32, a_type=cl.float16, b_type=cl.float16, m=M, n=N
           ).encode()
           for k_tile in range(K // BLOCK_K):
               stage = k_tile % STAGES
               generation = k_tile // STAGES
               cl.mbarrier_wait_parity(loaded.pointer(stage), generation % 2)
               cl.tcgen05_fence_after_thread_sync()
               a_desc = operand_descriptor(a_smem.pointer((stage, 0)))
               b_desc = operand_descriptor(b_smem.pointer((stage, 0)))
               for kk in cl.static_iter(range(BLOCK_K // MMA_K)):
                   cl.tcgen05_mma(
                       cl.Tcgen05MMAKind.F16, tmem,
                       cl.int64(a_desc + 2 * kk), cl.int64(b_desc + 2 * kk),
                       cl.int32(instruction),
                       accumulate=(k_tile != 0 or kk != 0),
                   )
               cl.tcgen05_commit(reusable.pointer(stage))
           cl.tcgen05_commit(finished)

       cl.barrier_sync_block()
       cl.mbarrier_wait_parity(finished, 0)
       cl.tcgen05_fence_after_thread_sync()
       store_accumulators(tmem, c)
       cl.tcgen05_fence_before_thread_sync()
       cl.barrier_sync_block()
       if warp == 0:
           cl.tcgen05_fence_after_thread_sync()
           cl.tcgen05_deallocate(tmem, TMEM_COLUMNS)


   if torch.cuda.get_device_capability() != (10, 0):
       raise RuntimeError("This example targets SM100a (compute capability 10.0).")

   torch.manual_seed(0)
   a = torch.randint(-2, 3, (M, K), device="cuda:0").to(torch.float16)
   b = torch.randint(-2, 3, (N, K), device="cuda:0").to(torch.float16)
   c = torch.empty((M, N), dtype=torch.float32, device="cuda:0")
   a_map = cl.tensor_map_tiled(
       a, (BLOCK_K, M), order=cl.AxisOrder.REVERSE,
       swizzle=cl.SwizzleMode.SWIZZLE_64B,
   )
   b_map = cl.tensor_map_tiled(
       b, (BLOCK_K, N), order=cl.AxisOrder.REVERSE,
       swizzle=cl.SwizzleMode.SWIZZLE_64B,
   )
   stream = torch.cuda.current_stream()
   cl.launch(stream, (1,), (THREADS,), gemm, (a_map, b_map, c))
   stream.synchronize()
   torch.testing.assert_close(c, a.float() @ b.float().T)
   print("Gen05 Tensor Core GEMM passed.")

.. testoutput:: warp-specialized-gemm
   :skipif: torch.cuda.get_device_capability() != (10, 0)

   Gen05 Tensor Core GEMM passed.

Following a Stage through Reuse
-------------------------------

The stage index is ``k_tile % STAGES``. Its generation is
``k_tile // STAGES``: each stage has its own phase sequence, independent of the
other stage. With two stages, the first six slices use:

.. list-table::
   :header-rows: 1

   * - K slice
     - Stage
     - Generation
     - Consumer's ``loaded`` parity
     - Producer's ``reusable`` parity
   * - 0
     - 0
     - 0
     - 0
     - Initial use: no wait
   * - 1
     - 1
     - 0
     - 0
     - Initial use: no wait
   * - 2
     - 0
     - 1
     - 1
     - 0
   * - 3
     - 1
     - 1
     - 1
     - 0
   * - 4
     - 0
     - 2
     - 0
     - 1
   * - 5
     - 1
     - 2
     - 0
     - 1

For each generation, the consumer waits with parity ``generation % 2``. Before
reusing that buffer, the producer waits for the preceding generation's MMA
completion with parity ``(generation - 1) % 2``. The producer registers new
transaction bytes for every transfer. The barrier automatically resets its
arrival count after each phase.

The reuse handshake also keeps the producer from advancing a stage past the
consumer's pending wait. The consumer must observe ``loaded`` before it can
issue MMA and commit ``reusable``; the producer must observe ``reusable`` before
it can fill that stage again. This respects the phase-window rules described
in :doc:`tma`.

Copy Completion and MMA Completion
----------------------------------

TMA completion permits the consumer to read the shared operands. MMA completion
permits the producer to overwrite them. These are different events: replacing
the ``reusable`` wait with a ``loaded`` wait would allow writes while MMA still
reads the same stage.

Only the consumer's elected lane issues MMA and its completion commits. The
sequence of same-shape MMA instructions that accumulate into the same TMEM
result is pipelined by the hardware for this issuing warp. Each commit tracks
preceding operations from its issuing thread. See the
`PTX tcgen05 pipeline rules
<https://docs.nvidia.com/cuda/parallel-thread-execution/#pipelined-tcgen05-instructions>`_.

The same-warp requirement matters: moving alternating MMA operations to
different warps would require an explicit completion and synchronization
protocol. The ``loaded`` and ``reusable`` handoffs order shared-memory accesses
to the operand buffers.

After the roles join at the block barrier, all threads wait on ``finished``
before loading the result. The join ensures the role loops have reached that
point; the mbarrier wait establishes asynchronous MMA completion. The result
helper and deallocation follow the same ordering as in :doc:`tcgen05`.

Keeping the Main Loop Independent
---------------------------------

There are no full-block barriers inside the producer and consumer loops. Each
role waits on the event it needs. A full-block barrier inside only one role
would wait for warps that do not execute that barrier.

This example uses 16 KiB of shared operand storage per stage, or 32 KiB across
two stages, plus barrier and address storage. While the consumer computes from
one stage, the producer can prepare the other.
