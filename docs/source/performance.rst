.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.tile

Performance Tuning
==================

Several performance tuning techniques are available in cuTile:

* architecture-specific configuration values, using :class:`ByTarget`;
* load/store hints such as ``latency`` and ``allow_tma``.


Architecture-specific configuration
-----------------------------------

.. autoclass:: ByTarget
   :members:
   :exclude-members: __init__

See :ref:`tile-kernels` for the full description of kernel configuration
parameters such as ``num_ctas``, ``occupancy`` and ``opt_level``. Any of
these options may be given as a :class:`ByTarget` value to specialize them
for different GPU architectures.

Load/store performance hints
----------------------------

The :func:`load` and :func:`store` operations accept optional keyword
arguments that can influence how memory traffic is scheduled and lowered:

* ``latency`` (``int`` or ``None``) – A hint indicating how heavy the DRAM
  traffic will be for this operation. It shall be an integer between 1 (low)
  and 10 (high). A large value typically fits the cases when DRAM traffic is
  high, and will likely result in a larger prefetch depth of the memory operation.

* ``allow_tma`` (``bool`` or ``None``) – If ``True``, the load or store may be
  lowered to use TMA (Tensor Memory Accelerator) when the target architecture
  supports it. If ``False``, TMA will not be used for this operation.
  By default, TMA is allowed.

These hints are optional: kernels will compile and run without specifying
them, but providing them can help the compiler make better code-generation
decisions for a particular memory-access pattern.


Example
~~~~~~~
.. literalinclude:: ../../test/test_load_store.py
    :start-after: example-begin
    :end-before: example-end


.. _autotuning:

Autotuning
----------

:func:`tune.exhaustive_search` provides a convenient way to measure kernel performance
on a finite space of configurations and return the best configuration.

.. autofunction:: cuda.tile.tune.exhaustive_search


To achieve consistent result with tuning, it is best to fix GPU clock and memory clock.

Enable persistent mode::

    nvidia-smi -i <GPU_ID> -pm 1

Query supported clocks::

    nvidia-smi -i <GPU_ID> --query-supported-clocks=graphics,memory --format=csv | head

Fix graphics and memory clocks::

    nvidia-smi -i <GPU_ID> -lgc <MIN_CLOCK>,<MAX_CLOCK>
    nvidia-smi -i <GPU_ID> -lmc <MIN_CLOCK>,<MAX_CLOCK>
