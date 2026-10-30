.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _lang-tcgen05:

Tensor Core (Gen5)
==================

Gen05 (fifth-generation) Tensor Cores expose the ``tcgen05`` instruction
interface. This chapter uses that interface through ``cuda.lang`` to compute
``C = A @ B.T`` with FP16 inputs and FP32 accumulation.
The allocation, descriptors, and synchronization described here are specific to Gen05.

The example builds on :doc:`tma`: TMA stages operands in shared memory, and an
asynchronous Gen05 MMA operation writes the result into *tensor memory* (TMEM).

The example targets SM100 GPUs, such as NVIDIA B200 (compute capability 10.0),
using the ``sm_100a`` target selected by :py:obj:`cl.launch <cuda.lang.launch>`.
Support for ``tcgen05`` depends on the target architecture and instruction
variant. Per-instruction target requirements appear in the Target ISA Notes
of the `PTX Tensor Core instruction reference
<https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions>`_.


From Global Memory to the Result
--------------------------------

The computation has four steps:

1. TMA copies a slice of A and B from global memory into shared memory.
2. The issuing thread waits for those copies, then submits MMA instructions.
3. MMA accumulates into TMEM and signals an mbarrier on completion.
4. Threads load the TMEM result into registers and store it to the output array.

Tensor Memory Layout
--------------------

TMEM is dedicated on-chip storage for ``tcgen05`` matrix operands and
accumulators. On the ``sm_100a`` target used here, a block's TMEM address space
has 128 rows, called *lanes*, and 512 columns. Each cell holds 32 bits.
The `PTX TMEM layout diagram
<https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tensor-memory-layout>`_
(ISA 9.4, CUDA 13.4.2, Figure 185) shows this organization and its addressing.

:func:`tcgen05_allocate` reserves columns across all 128 lanes. One column
therefore holds ``128 * 4 = 512`` bytes. The example allocates 128 columns
for its 128-by-128 FP32 accumulator, using 64 KiB. Its MMA configuration uses
`PTX Layout D
<https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-data-path-layout-d>`_:
accumulator row ``r`` occupies TMEM lane ``r``, with consecutive FP32 elements
in consecutive columns. Shared memory holds the input slices, mbarriers, and
the slot receiving the allocated TMEM address.

A TMEM address encodes a lane and a column. :func:`tcgen05_tmem_offset` advances
these coordinates; ``column_offset`` counts 32-bit cells. A *warpgroup* consists
of four consecutive warps. Register loads and stores have the following `access
ranges within each warpgroup
<https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-tensor-memory-ld-st-access-restrictions>`_:

.. _lang-tmem-warp-access-ranges:

.. list-table:: TMEM warp access ranges
   :header-rows: 1
   :widths: 50 50

   * - Warp index within a warpgroup
     - Accessible TMEM lanes
   * - 0
     - 0–31
   * - 1
     - 32–63
   * - 2
     - 64–95
   * - 3
     - 96–127

For example, warps 0 and 4 in the same block both have index 0 within their
respective warpgroups, so both can access TMEM lanes 0–31.
Each warp can access all allocated columns within its lane range.
:class:`Tcgen05LoadStoreShape` determines how :func:`tcgen05_load` distributes
those cells among the warp's threads and their returned register values.

A Complete GEMM Example
-----------------------

Save the following example as ``tensor_core_gemm.py`` and run
``python tensor_core_gemm.py`` on an SM100 GPU. It uses one block of 128 threads,
with ``M = N = 128`` and ``K = 256``. A and B are row-major arrays of shape
``(M, K)`` and ``(N, K)``.
The output is a row-major ``(M, N)`` FP32 array. These dimensions and layouts
are part of the example's contract.

Each iteration loads 32 elements along K. Thread zero issues the copies and
MMA instructions; all threads wait before the input buffers are reused. This
first version exposes the completion steps before introducing overlap.

.. testcode:: tensor-core-gemm
   :skipif: torch.cuda.get_device_capability() != (10, 0)

   import cuda.lang as cl
   import torch

   M, N, K = 128, 128, 256
   BLOCK_K, MMA_K = 32, 16
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


   @cl.kernel
   def gemm(a_map, b_map, c):
       a_smem = cl.shared_array(M * BLOCK_K, cl.float16, alignment=512)
       b_smem = cl.shared_array(N * BLOCK_K, cl.float16, alignment=512)
       loaded = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
       computed = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
       tmem_slot = cl.shared_array(
           1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
       )
       tid = cl.thread_index(0)
       warp = cl.warp_index()

       if tid == 0:
           cl.mbarrier_initialize(loaded, 1)
           cl.mbarrier_initialize(computed, 1)
       if warp == 0:
           cl.tcgen05_allocate(tmem_slot.pointer(), TMEM_COLUMNS)
           cl.tcgen05_relinquish_allocation_permit()
       cl.barrier_sync_block()
       tmem = tmem_slot[0]

       instruction = cl.Tcgen05InstructionDescriptor(
           d_type=cl.float32, a_type=cl.float16, b_type=cl.float16, m=M, n=N
       ).encode()
       a_desc = operand_descriptor(a_smem.pointer())
       b_desc = operand_descriptor(b_smem.pointer())

       for k_tile in range(K // BLOCK_K):
           if tid == 0:
               cl.mbarrier_expect_transaction(loaded, COPY_BYTES)
               cl.copy_async_bulk_tensor_global_to_shared(
                   a_map, (k_tile * BLOCK_K, 0), a_smem.pointer(), loaded
               )
               cl.copy_async_bulk_tensor_global_to_shared(
                   b_map, (k_tile * BLOCK_K, 0), b_smem.pointer(), loaded
               )
               cl.mbarrier_arrive(loaded)
           cl.mbarrier_wait_parity(loaded, k_tile % 2)

           if tid == 0:
               cl.tcgen05_fence_after_thread_sync()
               for kk in cl.static_iter(range(BLOCK_K // MMA_K)):
                   cl.tcgen05_mma(
                       cl.Tcgen05MMAKind.F16, tmem,
                       cl.int64(a_desc + 2 * kk), cl.int64(b_desc + 2 * kk),
                       cl.int32(instruction),
                       accumulate=(k_tile != 0 or kk != 0),
                   )
               cl.tcgen05_commit(computed)
           cl.mbarrier_wait_parity(computed, k_tile % 2)
           cl.barrier_sync_block()

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

.. testoutput:: tensor-core-gemm
   :skipif: torch.cuda.get_device_capability() != (10, 0)

   Gen05 Tensor Core GEMM passed.

Tensor Memory Allocation and Participation
------------------------------------------

Different instructions have different participation requirements:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Operation
     - Participants in this example
   * - :py:obj:`tcgen05_allocate <cuda.lang.tcgen05_allocate>` and :py:obj:`tcgen05_deallocate <cuda.lang.tcgen05_deallocate>`
     - All 32 threads of warp zero, executing the same collective instruction.
   * - :py:obj:`tcgen05_relinquish_allocation_permit <cuda.lang.tcgen05_relinquish_allocation_permit>`
     - All 32 threads of warp zero, executing the same collective instruction.
   * - TMA submission, :py:obj:`tcgen05_mma <cuda.lang.tcgen05_mma>`, and :py:obj:`tcgen05_commit <cuda.lang.tcgen05_commit>`
     - Thread zero. A single thread issues each matrix operation.
   * - :py:obj:`tcgen05_load <cuda.lang.tcgen05_load>`
     - All threads of each participating warp, with a warp-uniform TMEM address.

Allocation writes the TMEM address into ``tmem_slot`` in shared memory. The
block barrier makes that address available to the other threads. The address
is represented by a pointer in :py:obj:`MemorySpace.TENSOR <cuda.lang.MemorySpace>`; use
:func:`tcgen05_tmem_offset` to select TMEM lanes and columns.

After the allocation, warp zero calls
:func:`tcgen05_relinquish_allocation_permit` on behalf of the whole block.
No thread in the block may allocate more TMEM after this call. The existing
allocation remains usable until :func:`tcgen05_deallocate` releases it after
all readers finish. Explicitly deallocate TMEM before the kernel exits.

These examples use the default ``CTAGroup.CTA_1``. All ``tcgen05`` operations
with a CTA-group argument in a kernel must use the same group. A two-block
operation needs the additional participation and synchronization described in
the `PTX tensor-memory allocation rules
<https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-dealloc-tcgen05-relinquish-alloc-permit>`_.

Describing the Operand Layout
-----------------------------

Three descriptors have different jobs:

* The host-created tensor maps describe the global arrays and the TMA copy
  shape. ``order=cl.AxisOrder.REVERSE`` selects descriptor axes ``(K, M)`` for
  A and ``(K, N)`` for B, making K the contiguous dimension.
* :py:obj:`Tcgen05SharedMemoryDescriptor <cuda.lang.Tcgen05SharedMemoryDescriptor>` describes how MMA reads each shared buffer.
* :py:obj:`Tcgen05InstructionDescriptor <cuda.lang.Tcgen05InstructionDescriptor>` selects the matrix shape and element types
  for the MMA operation.

A TMA tile contains 32 FP16 values per row, or 64 bytes. Both the tensor maps
and the shared-memory descriptors select :py:obj:`SWIZZLE_64B <cuda.lang.SwizzleMode>`. For this K-major
layout, a swizzle atom holds eight rows of 64 bytes, so the stride between
successive eight-row groups is 512 bytes. The buffers request 512-byte alignment
to start on those atom boundaries, with the descriptor's default base offset
of zero. For this swizzled K-major layout, the leading-dimension field is unused and
assumed to be one in its encoded units; the example sets
``leading_dimension_byte_offset=16``, which encodes that value. These fields
describe the Gen05 shared-memory layout.

MMA reads the swizzled shared buffer through the descriptor. See :class:`Tcgen05SharedMemoryDescriptor` for
layout diagrams and the rules for choosing these fields.

Each FP16 MMA instruction consumes 16 values along K. Two instructions cover a
32-element input slice. Shared-memory descriptor addresses are encoded in
16-byte units, so advancing 16 FP16 values (32 bytes) adds two to the encoded
address. The first instruction uses ``accumulate=False`` to initialize the
result; every subsequent instruction includes the existing accumulator.

Waiting for the Right Operation
-------------------------------

``loaded`` tracks the two TMA copies. It expects one thread arrival and
``COPY_BYTES`` bytes: ``(128 + 128) * 32 * 2 = 16,384``. The two transfers satisfy
the byte count, and thread zero supplies the arrival after submitting them.

``computed`` tracks MMA completion. :func:`tcgen05_commit` arranges a hardware
arrival after the preceding MMA operations issued by that thread complete.
That hardware arrival supplies the barrier's one expected arrival and allows
the input buffers to be reused after the wait.

Both objects are reused across K slices, so their parity is ``k_tile % 2``.
The block barrier at the end of each iteration ensures all threads have
observed the completed phase before thread zero can advance the next one.

The ``tcgen05`` instruction family has additional ordering operations.
:func:`tcgen05_fence_after_thread_sync` orders subsequent ``tcgen05`` accesses
after the preceding synchronization. After the MMA completion wait, it orders
the TMEM loads after that wait. A TMEM load is itself asynchronous:
:func:`tcgen05_wait_load` completes it before its register values are stored.
``tcgen05_commit`` with an mbarrier wait observes MMA, copy, and shift
completion; ``tcgen05_wait_load`` completes TMEM-to-register loads.
The before/after-thread-sync fences around the final block barrier order the
TMEM readers before deallocation.

See the `PTX tcgen05 synchronization patterns
<https://docs.nvidia.com/cuda/parallel-thread-execution/#canonical-synchronization-patterns>`_
for completion and ordering requirements.

Writing the Result
------------------

``store_accumulators`` assigns each of the four warps 32 TMEM lanes, with
``lane_offset=warp * 32``. With the load shape used here, each thread retrieves
32 columns of its row at a time. Four loads cover all 128 columns. The resulting
vectors are private register values, which the helper writes to the global
output array.

The block finishes these reads before warp zero deallocates TMEM. Host stream
synchronization then waits for the kernel, and the example compares the result
against ``a.float() @ b.float().T``. The small integer-valued inputs make the
initial correctness check easy to interpret; other inputs require numerical
tolerances appropriate to FP16 multiplication and FP32 accumulation.

Continue with :doc:`warp-specialization` to let separate warps submit TMA
copies and MMA operations using two shared-memory buffers.
