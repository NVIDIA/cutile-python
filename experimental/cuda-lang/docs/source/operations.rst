.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang


Operations
==========

This reference includes host utilities and SIMT kernel operations. In ``Args``,
``const`` marks scalar arguments that must be known at compile time. Entries also
describe required thread participation and the synchronization needed before
using results or reusing storage.
Runtime addresses, counts, and descriptors must satisfy the documented
requirements even when the compiler cannot check their values.

.. _api-participation:

.. list-table:: Participation labels
   :header-rows: 1
   :widths: 25 75

   * - Label
     - Requirement
   * - Per-thread
     - Each calling thread performs its own operation.
   * - Single-thread issue
     - One call submits the whole operation. Each intended operation has one
       issuer; other threads may submit distinct operations. The entry
       specifies the operation's scope where applicable.
   * - Warp aligned
     - Threads in each participating warp execute the same instruction.
       A branch guarding the call must be uniform across the warp.
       Entries specify when all 32 lanes are required.
   * - Masked warp
     - Each caller belongs to the mask. All non-exited lanes selected by the
       mask execute the matching call with the same mask.
   * - Block aligned
     - All threads in the block execute the same instruction. A branch
       guarding the call must be uniform across the block.

``Completion`` uses **Synchronous** for ordinary execution by the calling
thread, or **Asynchronous** with the mechanism used to observe completion.
Barriers specify which participants or phase they wait for. ``Ordering``
describes visibility and synchronization between threads or proxies, including
the synchronization consumers need before reading stored data.


Execution
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

   kernel
   launch


.. _operations-array-creation:

Array Creation
--------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   local_array
   shared_array


.. _operations-pointer-utilities:

Pointer Utilities
-----------------
:meth:`Array.pointer` returns the base pointer or a pointer to a specified
array element.

.. autosummary::
   :toctree: generated
   :nosignatures:

   load
   store
   is_pointer_dtype
   pointer_dtype
   opaque_pointer_dtype
   address_space_cast
   map_shared_to_cluster
   map_shared_to_block_pair_leader
   shared_cluster_leader_bit_mask

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

   PointerInfo

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

   MemorySpace


.. _operations-type-casts:

Type Casts
----------
.. autosummary::
   :toctree: generated
   :nosignatures:

   bitcast


SIMT Model
----------
.. autosummary::
   :toctree: generated
   :nosignatures:

    thread_index
    thread_count
    block_index
    block_count
    cluster_index
    cluster_count
    block_in_cluster_index
    block_in_cluster_count
    lane_index
    lane_count
    warp_index
    warp_count
    environment_register
    full_mask
    elect_sync


Atomics
-------
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_with_members.rst

    MemoryOrder
    MemoryScope
    AtomicOp

.. autosummary::
   :toctree: generated
   :nosignatures:

    atomic_load
    atomic_store

    atomic_rmw


.. _lang-operations-math:

Math
----
.. autosummary::
   :toctree: generated
   :nosignatures:

    add
    sub
    mul
    wrapping_add
    wrapping_sub
    wrapping_mul
    fma
    truediv
    reciprocal
    floordiv
    mod
    integer_remainder
    divmod
    pow
    minimum
    maximum
    where
    negative
    abs
    ceil
    floor
    exp
    exp2
    log
    log2
    sqrt
    rsqrt
    sin
    cos
    sincos
    tan
    sinh
    cosh
    tanh
    atan2
    isnan
    isinf
    isfinite
    isnormal
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_not
    greater
    greater_equal
    less
    less_equal
    equal
    not_equal

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_with_members.rst

    RoundingMode
    SaturationMode

.. autosummary::
   :toctree: generated
   :nosignatures:

    cdiv


Warp Shuffle
------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    shuffle_sync

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    ShuffleKind


Warp Vote
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

    vote_all_sync
    vote_any_sync
    vote_uniform_sync
    vote_ballot_sync


Warp Matrix Load and Store
--------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    load_matrix
    store_matrix

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    MatrixLoadShape
    MatrixStoreShape
    MatrixLoadSourceFormat


.. _operations-tensor-map:

TensorMap
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

    tensor_map_tiled
    TensorMap

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    SwizzleMode
    AxisOrder


Async Copy
----------
Global to shared copies require SM80 or newer. Each thread issues and commits its own copies.

.. autosummary::
   :toctree: generated
   :nosignatures:

    copy_async_global_to_shared
    copy_async_commit_group
    copy_async_wait_groups


TensorMap Async Copy
--------------------
TMA requires SM90 or newer; individual modes and modifiers can require a
more specific target. See :doc:`lang/tma` for complete copy and synchronization
examples. The API entries below describe submission and completion separately.

.. autosummary::
   :toctree: generated
   :nosignatures:

    copy_async_bulk_tensor_global_to_shared
    copy_async_bulk_tensor_shared_to_global
    copy_async_bulk_commit_group
    copy_async_bulk_wait_group


Cache Control
-------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    prefetch
    prefetch_uniform
    prefetch_tensor_map
    create_range_cache_policy
    create_fractional_cache_policy

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    CacheHint
    CachePolicy
    PrefetchLevel


Synchronization
---------------
:py:obj:`syncwarp <cuda.lang.syncwarp>` and the :py:obj:`syncthreads <cuda.lang.syncthreads>` family provide CUDA C++ style convenience functions.
The underlying barrier APIs also expose named and partial block barriers,
cluster barriers, and shared-memory mbarriers.

.. autosummary::
   :toctree: generated
   :nosignatures:

    syncwarp
    syncthreads
    syncthreads_count
    syncthreads_and
    syncthreads_or

    barrier_sync_warp
    barrier_sync_block
    barrier_sync_block_aligned
    barrier_arrive_block
    barrier_arrive_block_aligned
    barrier_reduce_block
    barrier_reduce_block_aligned
    barrier_arrive_cluster
    barrier_arrive_cluster_aligned
    barrier_wait_cluster
    barrier_wait_cluster_aligned
    barrier_sync_cluster
    barrier_sync_cluster_aligned

Basic mbarrier operations require SM80 or newer. Transaction counting, cluster
scope, relaxed ordering, and the ``try_wait`` and ``wait`` families require SM90
or newer. Explicit layouts require SM90 or newer and a toolchain supporting the
PTX 9.3 ``.layout`` qualifier. See the `PTX mbarrier Target ISA Notes
<https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>`_.

.. autosummary::
   :toctree: generated
   :nosignatures:

    mbarrier_initialize
    mbarrier_invalidate
    mbarrier_arrive
    mbarrier_arrive_expect_transaction
    mbarrier_expect_transaction
    mbarrier_complete_transaction
    mbarrier_test_wait
    mbarrier_test_wait_parity
    mbarrier_try_wait
    mbarrier_try_wait_parity
    mbarrier_wait
    mbarrier_wait_parity
    copy_async_mbarrier_arrive
    copy_async_mbarrier_arrive_no_increment

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    MbarrierScope
    BarrierReductionKind


Memory Fence
------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    memory_barrier
    fence
    fence_proxy_bidirectional

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_with_members.rst

    FenceProxy
    FenceRestriction


TensorCore (Gen5)
-----------------

Gen5 target support depends on the instruction and its modifiers. The
`PTX Gen5 reference
<https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions>`_
lists per-instruction requirements in its Target ISA Notes.
The example in :doc:`lang/tcgen05` targets ``sm_100a``.

.. autosummary::
   :toctree: generated
   :nosignatures:

    tcgen05_allocate
    tcgen05_deallocate
    tcgen05_tmem_offset
    tcgen05_commit
    tcgen05_load
    tcgen05_copy
    tcgen05_store
    tcgen05_mma
    tcgen05_mma_block_scale
    tcgen05_mma_weight_stationary
    tcgen05_wait_load
    tcgen05_wait_store
    tcgen05_fence_before_thread_sync
    tcgen05_fence_after_thread_sync
    tcgen05_shift_down
    tcgen05_relinquish_allocation_permit

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    Tcgen05SharedMemoryDescriptor
    Tcgen05InstructionDescriptor
    Tcgen05Mxf8f6f4InstructionDescriptor
    Tcgen05Mxf4InstructionDescriptor
    CTAGroup
    Tcgen05MMAKind
    Tcgen05MMABlockScaleKind
    Tcgen05MMAScaleVectorSize
    Tcgen05MMACollectorBBuffer
    Tcgen05MMACollectorOp
    Tcgen05LoadStoreShape
    Tcgen05CopyMulticast
    Tcgen05CopyShape
    Tcgen05CopySourceFormat


Cluster Launch Control
----------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    cluster_launch_control_try_cancel
    cluster_launch_control_is_canceled
    cluster_launch_control_get_first_block_index

Programmatic Dependent Launch
-----------------------------
The ``programmatic_dependent_launch=True`` keyword argument to :func:`launch`
enables Programmatic Dependent Launch (PDL) for the dependent kernel.

.. autosummary::
   :toctree: generated
   :nosignatures:

    grid_dependency_control_wait
    grid_dependency_control_launch_dependents


Utility
-------
.. autosummary::
   :toctree: generated
   :nosignatures:

    nanosleep


.. _lang-operations-classes:

Classes
-------
.. autosummary::
   :nosignatures:

   Array
   Pointer
   Vector

.. toctree::
   :hidden:

   data/array
   data/pointer
   data/vector
   data/tensor_map


.. _lang-operations-annotations:

Parameter Annotations
---------------------

These ``cuda.lang`` annotations are aliases of the corresponding ``cuda.tile``
objects. See :ref:`language-parameter-annotations` for their semantics at kernel
and compiled host function boundaries. The linked definitions document their
arguments and shared behavior.

Constant Annotations
~~~~~~~~~~~~~~~~~~~~

.. py:data:: Constant

   Alias of :py:data:`cuda.tile.Constant`. Marks a parameter as a compile-time
   constant.

Array Annotations
~~~~~~~~~~~~~~~~~

.. py:class:: ArrayAnnotation

   Alias of :py:class:`cuda.tile.ArrayAnnotation`. Controls array indexing types
   and specialization of shape and stride dimensions.

Scalar Annotations
~~~~~~~~~~~~~~~~~~

.. py:data:: ScalarInt64

   Alias of :py:data:`cuda.tile.ScalarInt64`. Makes an integer parameter use
   ``int64``.
