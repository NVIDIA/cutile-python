.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

Feature Gaps
============

``cuda.tile`` and ``cuda.lang`` share language semantics, with differences in
supported kernel arguments and public tooling. The table summarizes current
availability.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Feature
     - ``cuda.tile``
     - ``cuda.lang``
   * - Lists of arrays as kernel arguments
     - Supported
     - Not supported
   * - Kernel export / ahead-of-time compilation
     - Supported
     - Not supported
   * - Compiled host functions (``host_entry``)
     - Not supported
     - Supported
   * - Autotuning API (``tune``)
     - Supported
     - Not supported

Unsupported features may become available in future releases.

Lists of Arrays
---------------

Tile kernels accept non-empty Python lists whose arrays have the same rank
and dtype. The list view supports ``len`` and runtime integer indexing, and
its structure is immutable. See :doc:`/tile/data` for the argument interface.

SIMT kernels and compiled host entries currently do not support list arguments.
Both models accept tuples of arrays with a compile-time-known structure and
constant tuple indices. See :ref:`language-entry-arguments` for the entry
argument types and their conversion rules.

Kernel Export and AOT Compilation
---------------------------------

:py:func:`cuda.tile.compilation.export_kernel` exports kernels as cubin files
or Tile IR bytecode, using explicit argument signatures and constraints.
The :doc:`/tile/compilation` chapter describes these formats and calling
conventions.

``cuda.lang`` currently exposes JIT kernel execution through
:py:func:`cuda.lang.launch`; a public kernel export/AOT API is not available.

Compiled Host Functions
-----------------------

:py:obj:`cuda.lang.host_entry` compiles Python functions into CPU code that can
launch SIMT kernels. Host functions and the kernels they launch have separate
specializations. See :ref:`lang-compiled-host-functions` for the interface
and specialization rules.

Autotuning
----------

:py:func:`cuda.tile.tune.exhaustive_search` measures kernel performance over
a finite configuration space and reports measurements and the best configuration.
The :doc:`/tile/performance` chapter describes the tuning API and compiler hints.
``cuda.lang`` currently has no corresponding public autotuning API.
