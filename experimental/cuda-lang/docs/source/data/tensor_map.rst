.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _data-tensor-map-descriptor-pointers:

Tensor-map descriptor values and pointers
=========================================

:func:`tensor_map_tiled` returns an immutable :class:`TensorMap` value containing
the descriptor bytes. The caller must keep the memory referenced by the
descriptor alive until all kernels using it have completed.

Passing a descriptor from host to kernel copies the descriptor into device
as a grid constant and passes a pointer to the descriptor as the kernel argument.

In device code, the pointer can be used directly for an async copy. Loading
from this pointer produces an opaque descriptor value
that can be stored into another descriptor pointer. For example::

    shared_ptr = cl.shared_array(1, cl.tensor_map_descriptor, alignment=128).pointer()
    if cl.thread_index(0) == 0:
        shared_ptr[0] = tmap_ptr.load()
    cl.barrier_sync_block_aligned()
