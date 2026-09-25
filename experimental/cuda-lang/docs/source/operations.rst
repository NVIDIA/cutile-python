.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang


Operations
==========


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
Use :meth:`Array.pointer` to get the base pointer or a pointer to a specified
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
   map_shared_to_leader_block
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


.. _operations-simt-model:

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
    full_mask
    elect_sync


Atomics
-------
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    MemoryOrder
    MemoryScope
    AtomicOp

.. autosummary::
   :toctree: generated
   :nosignatures:

    atomic_load
    atomic_store

    atomic_rmw


.. _operations-math:

Math
----
.. currentmodule:: cuda.lang._stub.math
.. autosummary::
   :toctree: generated
   :nosignatures:

    add
    sub
    mul
    fma
    truediv
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
.. currentmodule:: cuda.lang

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    RoundingMode
    SaturationMode

.. autosummary::
   :toctree: generated
   :nosignatures:

    cdiv


Warp shuffle
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


Warp vote
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

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    SwizzleMode


TensorMap Async Copy
--------------------
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

    CachePolicy
    PrefetchLevel


Synchronization
---------------
``syncwarp`` and the ``syncthreads`` family provide CUDA C++ style convenience functions.
They call more explicit underlying barrier APIs which offer richer interfaces. Use the
barrier APIs directly for features such as named or partial block barriers, cluster barriers,
and shared-memory mbarriers.

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
    barrier_arrive_block
    barrier_reduce_block
    barrier_arrive_cluster
    barrier_wait_cluster
    barrier_sync_cluster

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


TensorCore (Gen5)
-----------------
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
To use Programmatic Dependent Launch (PDL), launch the dependent kernel with
the ``programmatic_dependent_launch=True`` keyword argument to :func:`launch`.

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


.. _operations-classes:

Classes
-------
.. autosummary::
   :nosignatures:

   Array
   Pointer
   Vector
   TensorMap

.. toctree::
   :hidden:

   data/array
   data/pointer
   data/vector
   data/tensor_map
