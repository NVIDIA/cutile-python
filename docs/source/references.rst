.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. |grid| replace:: :ref:`grid <grid>`
.. |grids| replace:: :ref:`grids <grid>`
.. |Grid| replace:: :ref:`Grid <grid>`
.. |Grids| replace:: :ref:`Grids <grid>`

.. |block| replace:: :ref:`block <block>`
.. |Block| replace:: :ref:`Block <block>`
.. |blocks| replace:: :ref:`blocks <block>`
.. |Blocks| replace:: :ref:`Blocks <block>`

.. |host code| replace:: :ref:`host code <host code>`
.. |Host code| replace:: :ref:`Host code <host code>`

.. |SIMT code| replace:: :ref:`SIMT code <SIMT code>`
.. |SIMT codes| replace:: :ref:`SIMT codes <SIMT code>`

.. |tile code| replace:: :ref:`tile code <tile code>`
.. |Tile code| replace:: :ref:`Tile code <tile code>`

.. |execution space| replace:: :ref:`execution space <execution-execution-spaces>`
.. |Execution space| replace:: :ref:`Execution space <execution-execution-spaces>`
.. |execution spaces| replace:: :ref:`execution spaces <execution-execution-spaces>`
.. |Execution spaces| replace:: :ref:`Execution spaces <execution-execution-spaces>`

.. |tile function| replace:: :ref:`tile function <execution-tile-functions>`
.. |Tile function| replace:: :ref:`Tile function <execution-tile-functions>`
.. |tile functions| replace:: :ref:`tile functions <execution-tile-functions>`
.. |Tile functions| replace:: :ref:`Tile functions <execution-tile-functions>`

.. |tile kernel| replace:: :ref:`tile kernel <execution-tile-kernels>`
.. |Tile kernel| replace:: :ref:`Tile kernel <execution-tile-kernels>`
.. |tile kernels| replace:: :ref:`tile kernels <execution-tile-kernels>`
.. |Tile kernels| replace:: :ref:`Tile kernels <execution-tile-kernels>`
.. |kernel| replace:: :ref:`kernel <execution-tile-kernels>`
.. |Kernel| replace:: :ref:`Kernel <execution-tile-kernels>`
.. |kernels| replace:: :ref:`kernels <execution-tile-kernels>`
.. |Kernels| replace:: :ref:`Kernels <execution-tile-kernels>`

.. |constant embed| replace:: :ref:`constant embed <execution-constant-embedding>`
.. |Constant embed| replace:: :ref:`Constant embed <execution-constant-embedding>`
.. |constant embedding| replace:: :ref:`constant embedding <execution-constant-embedding>`
.. |Constant embedding| replace:: :ref:`Constant embedding <execution-constant-embedding>`
.. |constant embedded| replace:: :ref:`constant embedded <execution-constant-embedding>`
.. |Constant embedded| replace:: :ref:`Constant embedded <execution-constant-embedding>`

.. |loosely typed| replace:: :ref:`loosely typed <execution-constant-expressions-objects>`
.. |loosely typed numeric constant| replace:: :ref:`loosely typed numeric constant <execution-constant-expressions-objects>`
.. |loosely typed numeric constants| replace:: :ref:`loosely typed numeric constants <execution-constant-expressions-objects>`

.. |data model| replace:: :ref:`data model <data-data-model>`
.. |Data model| replace:: :ref:`Data model <data-data-model>`

.. |global array| replace:: :ref:`global array <data-array-cuda-tile-array>`
.. |Global array| replace:: :ref:`Global array <data-array-cuda-tile-array>`
.. |global arrays| replace:: :ref:`global arrays <data-array-cuda-tile-array>`
.. |Global arrays| replace:: :ref:`Global arrays <data-array-cuda-tile-array>`
.. |array| replace:: :ref:`array <data-array-cuda-tile-array>`
.. |Array| replace:: :ref:`Array <data-array-cuda-tile-array>`
.. |arrays| replace:: :ref:`arrays <data-array-cuda-tile-array>`
.. |Arrays| replace:: :ref:`Arrays <data-array-cuda-tile-array>`

.. |tile| replace:: :ref:`tile <data-tiles-and-scalars>`
.. |Tile| replace:: :ref:`Tile <data-tiles-and-scalars>`
.. |tiles| replace:: :ref:`tiles <data-tiles-and-scalars>`
.. |Tiles| replace:: :ref:`Tiles <data-tiles-and-scalars>`

.. |data type| replace:: :ref:`data type <data-data-types>`
.. |Data type| replace:: :ref:`Data type <data-data-types>`
.. |data types| replace:: :ref:`data types <data-data-types>`
.. |Data types| replace:: :ref:`Data types <data-data-types>`
.. |dtype| replace:: :ref:`dtype <data-data-types>`
.. |Dtype| replace:: :ref:`Dtype <data-data-types>`
.. |dtypes| replace:: :ref:`dtypes <data-data-types>`
.. |Dtypes| replace:: :ref:`Dtypes <data-data-types>`

.. |numeric data type| replace:: :ref:`numeric data type <data-numeric-arithmetic-data-types>`
.. |Numeric data type| replace:: :ref:`Numeric data type <data-numeric-arithmetic-data-types>`
.. |numeric data types| replace:: :ref:`numeric data types <data-numeric-arithmetic-data-types>`
.. |Numeric data types| replace:: :ref:`Numeric data types <data-numeric-arithmetic-data-types>`
.. |numeric dtype| replace:: :ref:`numeric dtype <data-numeric-arithmetic-data-types>`
.. |Numeric dtype| replace:: :ref:`Numeric dtype <data-numeric-arithmetic-data-types>`
.. |numeric dtypes| replace:: :ref:`numeric dtypes <data-numeric-arithmetic-data-types>`
.. |Numeric dtypes| replace:: :ref:`Numeric dtypes <data-numeric-arithmetic-data-types>`

.. |arithmetic data type| replace:: :ref:`arithmetic data type <data-numeric-arithmetic-data-types>`
.. |Arithmetic data type| replace:: :ref:`Arithmetic data type <data-numeric-arithmetic-data-types>`
.. |arithmetic data types| replace:: :ref:`arithmetic data types <data-numeric-arithmetic-data-types>`
.. |Arithmetic data types| replace:: :ref:`Arithmetic data types <data-numeric-arithmetic-data-types>`
.. |arithmetic dtype| replace:: :ref:`arithmetic dtype <data-numeric-arithmetic-data-types>`
.. |Arithmetic dtype| replace:: :ref:`Arithmetic dtype <data-numeric-arithmetic-data-types>`
.. |arithmetic dtypes| replace:: :ref:`arithmetic dtypes <data-numeric-arithmetic-data-types>`
.. |Arithmetic dtypes| replace:: :ref:`Arithmetic dtypes <data-numeric-arithmetic-data-types>`

.. |type promotion| replace:: :ref:`type promotion <data-arithmetic-promotion>`

.. |tile space| replace:: :ref:`tile space <data-element-tile-space>`
.. |Tile space| replace:: :ref:`Tile space <data-element-tile-space>`
.. |tile spaces| replace:: :ref:`tile spaces <data-element-tile-space>`
.. |Tile spaces| replace:: :ref:`Tile spaces <data-element-tile-space>`

.. |scalar| replace:: :ref:`scalar <data-tiles-and-scalars>`
.. |Scalar| replace:: :ref:`Scalar <data-tiles-and-scalars>`
.. |scalars| replace:: :ref:`scalars <data-tiles-and-scalars>`
.. |Scalars| replace:: :ref:`Scalars <data-tiles-and-scalars>`

.. |Rounding Modes| replace:: :ref:`Rounding Modes <data-rounding-modes>`
.. |Padding Modes| replace:: :ref:`Padding Modes <data-padding-modes>`

.. |machine representation| replace:: :ref:`machine representation <interoperability-machine-representation>`
.. |Machine representation| replace:: :ref:`Machine representation <interoperability-machine-representation>`
.. |machine representations| replace:: :ref:`machine representations <interoperability-machine-representation>`
.. |Machine representations| replace:: :ref:`Machine representations <interoperability-machine-representation>`

.. |operations| replace:: :ref:`operations <operations-operations>`

.. |loading and storing| replace:: :ref:`loading and storing <operations-load-store>`

.. |factory| replace:: :ref:`factory <operations-factory>`
.. |Factory| replace:: :ref:`Factory <operations-factory>`

.. |memory order| replace:: :ref:`memory order <memory-model-memory-order>`
.. |Memory Order| replace:: :ref:`Memory Order <memory-model-memory-order>`
.. |memory scope| replace:: :ref:`memory scope <memory-model-memory-scope>`
.. |Memory Scope| replace:: :ref:`Memory Scope <memory-model-memory-scope>`

.. |Python Array API| replace:: `Python Array API <https://data-apis.org/array-api/latest/>`__
.. |Numpy broadcasting rule| replace:: `Numpy broadcasting rule <https://numpy.org/devdocs/user/basics.broadcasting.html>`__
.. |DLPack| replace:: `DLPack <https://dmlc.github.io/dlpack/latest/>`__
.. |CUDA Array Interface| replace:: `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`__
.. |NumPy| replace:: `NumPy <https://numpy.org/doc/stable/>`__
.. |CuPy| replace:: `CuPy <https://docs.cupy.dev/en/stable/>`__
.. |PyTorch| replace:: `PyTorch <https://pytorch.org/docs/stable/>`__
.. |IEEE 754-2019| replace:: `IEEE 754-2019 <https://standards.ieee.org/standard/754-2019.html>`__
