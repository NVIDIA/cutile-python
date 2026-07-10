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
.. autosummary::
   :toctree: generated
   :nosignatures:

   is_pointer_dtype
   pointer_dtype
   opaque_pointer_dtype
   address_space_cast
   bitcast
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

.. autosummary::
   :toctree: generated
   :nosignatures:

    atomic_add
    atomic_sub
    atomic_and
    atomic_or
    atomic_xor
    atomic_min
    atomic_max
    atomic_inc
    atomic_dec
    atomic_xchg
    atomic_cas


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
    truediv
    floordiv
    mod
    pow
    minimum
    maximum
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
    tan
    sinh
    cosh
    tanh
    atan2
    isnan
    isinf
    isfinite
    isnormal
.. currentmodule:: cuda.lang

.. autosummary::
   :toctree: generated
   :nosignatures:

    cdiv


Warp shuffle
------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    shfl_sync
    shfl_up_sync
    shfl_down_sync
    shfl_xor_sync


.. _operations-tensor-map:

TensorMap
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

    tensor_map_tiled
    copy_async_bulk_tensor_global_to_shared
    copy_async_bulk_tensor_shared_to_global

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


Synchronization
---------------
.. autosummary::
   :toctree: generated
   :nosignatures:

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
    fence_sc_cluster
    fence_mbarrier_initialize
    fence_sync_restrict
    fence_proxy
    fence_proxy_acquire
    fence_proxy_release
    fence_proxy_sync_restrict


TensorCore (Gen5)
-----------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    tcgen05_allocate
    tcgen05_deallocate
    tcgen05_commit
    tcgen05_load
    tcgen05_copy
    tcgen05_store
    tcgen05_mma
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
    SwizzleMode
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

    clusterlaunchcontrol_try_cancel
    clusterlaunchcontrol_is_canceled
    clusterlaunchcontrol_get_first_block_index

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
