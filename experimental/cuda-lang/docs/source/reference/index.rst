.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

API Reference
=============

The namespaces have distinct execution and data-access APIs. For example,
:py:obj:`cuda.lang.launch <cuda.lang.launch>` accepts an explicit thread-block size, while
:py:obj:`cuda.tile.launch <cuda.tile.launch>` lets the compiler choose the thread mapping.

The operation catalogs below describe each namespace's APIs.
The :doc:`/lang/index` and
:doc:`/tile/index` explain each programming model, and the
:doc:`/getting-started/index` includes runnable examples of both.

.. toctree::
   :maxdepth: 1

   cuda.lang API </operations>
   cuda.tile API </tile/operations>
   feature-gaps
