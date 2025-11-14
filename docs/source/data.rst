.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.tile

Data Model
==========

cuTile is an array-based programming model.
The fundamental data structure is multidimensional arrays with elements of a single homogeneous type.
cuTile Python does not expose pointers, only arrays.

An array-based model was chosen because:

- Arrays know their bounds, so accesses can be checked to ensure safety and correctness.
- Array-based load/store operations can be efficiently lowered to speed-of-light hardware mechanisms.
- Python programmers are used to array-based programming frameworks such as NumPy.
- Pointers are not a natural choice for Python.

Within |tile code|, only the types described in this section are supported.

Global Arrays
-------------

.. autoclass:: Array
   :no-members:
   :no-index:

   .. seealso::
      :ref:`Complete cuda.tile.Array class documentation <data/array:cuda.tile.Array>`

.. toctree::
   :maxdepth: 2
   :hidden:

   data/array

Tiles
-----

.. autoclass:: Tile
   :no-members:
   :no-index:

   .. seealso::
      :ref:`Complete cuda.tile.Tile class documentation <data/tile:cuda.tile.Tile>`

.. toctree::
   :maxdepth: 2
   :hidden:

   data/tile

Element & Tile Space
--------------------

.. image:: /_static/images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__dark_background.svg
   :class: only-dark

.. image:: /_static/images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__light_background.svg
   :class: only-light

.. image:: /_static/images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__dark_background.svg
   :class: only-dark

.. image:: /_static/images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__light_background.svg
   :class: only-light

The *element space* of an array is the multidimensional space of elements contained in that array,
stored in memory according to a certain layout (row major, column major, etc).

The *tile space* of an array is the multidimensional space of tiles into that array of a certain
tile shape.
A tile index ``(i, j, ...)`` with shape ``S`` refers to the elements of the array that belong to the
``(i+1)``-th, ``(j+1)``-th, ... tile.

When accessing the elements of an array using tile indices, the multidimensional memory layout of the array is used.
To access the tile space with a different memory layout, use the `order` parameter of load/store operations.

Shape Broadcasting
------------------

*Shape broadcasting* allows |tiles| with different shapes to be combined in arithmetic operations.
When performing operations between |tiles| of different shapes, smaller |tile| is automatically
extended to match the shape of the larger one, following these rules:

- |Tiles| are aligned by their trailing dimensions.
- If the corresponding dimensions have the same size or one of them is 1, they are compatible.
- If one |tile| has fewer dimensions, its shape is padded with 1s on the left.

Broadcasting follows the same semantics as |NumPy|, which makes code more concise and readable
while maintaining computational efficiency.

Data Types
----------

.. autoclass:: cuda.tile.DType()
   :members:

.. include:: generated/includes/numeric_dtypes.rst

Numeric & Arithmetic Data Types
-------------------------------
A *numeric* data type represents numbers. An *arithmetic* data type is a numeric data type
that supports general arithmetic operations such as addition, subtraction, multiplication,
and division.


Arithmetic Promotion
--------------------

Binary operations can be performed on two |tile| or |scalar| operands of different |numeric dtypes|.

When both operands are |loosely typed numeric constants|, then the result is also
a loosely typed constant: for example, ``5 + 7`` is a loosely typed integral constant 12,
and ``5 + 3.0`` is a loosely typed floating-point constant 8.0.

If any of the operands is not a |loosely typed numeric constant|, then both are *promoted*
to a common dtype using the following process:

- Each operand is classified into one of the three categories:
  *boolean*, *integral* or *floating-point*.
  The categories are ordered as follows: *boolean* < *integral* < *floating-point*.
- If either operand is a |loosely typed numeric constant|, a concrete dtype is picked for it:
  integral constants are treated as `int32`, and floating-point constants as treated as `float32`.
- If one of the two operands has a higher category than the other, then its concrete dtype
  is chosen as the common dtype.
- If both operands are of the same category, but one of them is a |loosely typed numeric constant|,
  then the other operand's dtype is picked as the common dtype.
- Otherwise, the common dtype is computed according to the table below.

.. rst-class:: compact-table

.. include:: generated/includes/dtype_promotion_table.rst


Scalars
-------

A *scalar* is a single immutable value of a specific |data type|.

|Scalars| can be used in |host code| and |tile code|.
They can be |kernel| parameters.

.. toctree::
   :maxdepth: 2
   :hidden:

   data/scalar

Tuples
------

Tuples can be used in |tile code|. They cannot be |kernel| parameters.

Rounding Modes
--------------

.. autoclass:: cuda.tile.RoundingMode
   :members:
   :undoc-members:
   :member-order: bysource

Padding Modes
-------------

.. autoclass:: cuda.tile.PaddingMode
   :members:
   :undoc-members:
   :member-order: bysource