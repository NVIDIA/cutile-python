.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

Interoperability
================

.. _interoperability-machine-representation:

.. _tile-simt-interoperability:

Interoperability with SIMT
--------------------------

Inter-Kernel
~~~~~~~~~~~~

Inter-kernel interoperability lets separate tile and SIMT kernels share
source files, binaries, and data. This includes:

- Writing tile and SIMT kernels in the same source file.
- Linking tile and SIMT kernels into the same binary.
- Passing the same kinds of arrays to both tile and SIMT kernels.

You can launch separate SIMT and tile kernels on the same stream and pass the
same GPU allocations between them. See :doc:`/integration/index` for supported
third-party arrays and streams.

Intra-Kernel
~~~~~~~~~~~~

A tile kernel can call a SIMT function through tile/SIMT interoperability,
combining tile computations and SIMT implementations within the same kernel.
There is an explicit call boundary between the two programming models.

Calling SIMT functions from tile kernels is currently a preview feature and
requires the separate ``cuda-tile-preview`` package. More details will be coming soon.


JAX FFI
-------

cuTile kernels can be launched from JAX-traced graphs via
:func:`cuda.tile.jax.cutile_call`, which threads buffers, scalar, and
tuple arguments through the JAX FFI call site so the kernel runs as a
regular op inside a ``jax.jit``\-compiled graph.

See :func:`cuda.tile.jax.cutile_call` for the full argument convention,
along with :class:`cuda.tile.jax.OutputPlaceholder` and
:class:`cuda.tile.jax.InputOutput` for declaring outputs and in-place
updates.
