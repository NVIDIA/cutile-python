.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.tile

.. _data-data-model:

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

A *global array* (or *array*) is a container of elements of a specific |dtype|
arranged in a logical multidimensional space.

Array's *shape* is a tuple of integer values, each denoting the length of
the corresponding dimension.
The length of the shape tuple equals the arrays's number of dimensions.
The product of shape values equals the total logical number of elements in the array.

Arrays are stored in global memory using a *strided memory layout*: in addition to a shape,
an array also has an equally sized tuple of *strides*. Strides determine the mapping of logical
array indices to physical memory locations. For example, for a 3-dimensional `float32` array
with strides `(s1, s2, s3)`, the memory address of the element at the logical index
`(i1, i2, i3)` will be:

.. code-block::

    base_addr + 4 * (s1 * i1 + s2 * i2 + s3 * i3),

where ``base_addr`` is the base address of the array and `4` is the byte size of a single `float32`
element.

New arrays can only be allocated by the host, and passed to the tile kernel as arguments.
|Tile code| can only create new views of existing arrays, for example using
:meth:`Array.slice`. Like in Python, assigning an array object to another variable does not copy
the underlying data, but creates another reference to the array object.

Any object that implements the |DLPack| interface or the |CUDA Array Interface|
can be passed to the kernel as an argument. Example: |CuPy| arrays and |PyTorch| tensors.

If two or more array arguments are passed to the kernel, their memory storage must not overlap.
Otherwise, behavior is undefined.

Array's shape can be queried using the :py:attr:`Array.shape` attribute, which
returns a tuple of `int32` scalars. These scalars are non-constant, runtime values.
Using `int32` makes the tile code more performant at the cost of limiting the maximum
representable shape at 2,147,483,647 elements. This limitation will be lifted in the future.


.. seealso::
  :ref:`cuda.tile.Array class documentation <data-array-cuda-tile-array>`

.. toctree::
   :maxdepth: 2
   :hidden:

   data/array


.. _data-tiles-and-scalars:

Tiles and Scalars
-----------------
A *tile* is an immutable multidimensional collection of elements of a specific |dtype|.

Tile's *shape* is a tuple of integer values, each denoting the length of the corresponding dimension.
The length of the shape tuple equals the tile's number of dimensions.
The product of shape values equals the total number of elements in the tile.

The shape of a tile must be known at compile time. Each dimension of a tile must be a power of 2.

Tile's dtype and shape can be queried with the ``dtype`` and ``shape`` attributes, respectively.
For example, if ``x`` is a `float32` tile, the expression ``x.dtype`` will return
a compile-time constant equal to :py:data:`cuda.tile.float32`.

A zero-dimensional tile is called a *scalar*. Such tile has exactly one element. The shape
of a scalar is the empty tuple `()`. Numeric literals like `7` or `3.14` are treated as
constant scalars, i.e. zero-dimensional tiles.

Since scalars are tiles, they slightly differ in behavior from Python's ``int``/``float`` objects.
For example, they have ``dtype`` and ``shape`` attributes:

.. code-block:: python

    a = 0
    # The following line will evaluate to cuda.tile.int32 in cuTile,
    # but would raise an AttributeError in Python:
    a.dtype

Tiles can only be used in |tile code|, not host code.
The contents of a tile do not necessarily have a physical representation in memory.
Non-scalar tiles can be created by loading from |global arrays| using functions such as
:py:func:`cuda.tile.load` and :py:func:`cuda.tile.gather` or with |factory| functions
such as :py:func:`cuda.tile.zeros`.

Tiles can also be stored into global arrays using functions such as :py:func:`cuda.tile.store`
or :py:func:`cuda.tile.scatter`.

Only scalars (i.e. 0-dimensional tiles) can be used as |kernel| parameters.

Scalar constants are |loosely typed| by default, for example, a literal ``2`` or
a constant attribute like ``Tile.ndim``, ``Tile.shape``, or ``Array.ndim``.

.. seealso::
  :ref:`cuda.tile.Tile class documentation <data-tile-cuda-tile-tile>`

.. toctree::
   :maxdepth: 2
   :hidden:

   data/tile


.. _data-element-tile-space:

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
When performing operations between |tiles| of different shapes, the smaller |tile| is automatically
extended to match the shape of the larger one, following these rules:

- |Tiles| are aligned by their trailing dimensions.
- If the corresponding dimensions have the same size or one of them is 1, they are compatible.
- If one |tile| has fewer dimensions, its shape is padded with 1s on the left.

Broadcasting follows the same semantics as |NumPy|, which makes code more concise and readable
while maintaining computational efficiency.

.. _data-data-types:

Data Types
----------

.. autoclass:: cuda.tile.DType()
   :members:

.. include:: generated/includes/numeric_dtypes.rst

.. _data-numeric-arithmetic-data-types:

Numeric & Arithmetic Data Types
-------------------------------
A *numeric* data type represents numbers. An *arithmetic* data type is a numeric data type
that supports general arithmetic operations such as addition, subtraction, multiplication,
and division.


.. _data-arithmetic-promotion:

Arithmetic Promotion
--------------------

Binary operations can be performed on two |tile| or |scalar| operands of different |numeric dtypes|.

When both operands are |loosely typed numeric constants|, then the result is also
a loosely typed constant: for example, ``5 + 7`` is a loosely typed integral constant 12,
and ``5 + 3.0`` is a loosely typed floating-point constant 8.0.

If any of the operands is not a |loosely typed numeric constant|, then both are *promoted*
to a common dtype using the following process:

- Each operand is classified into one of the three categories:
  *boolean*, *integral*, or *floating-point*.
  The categories are ordered as follows: *boolean* < *integral* < *floating-point*.
- If either operand is a |loosely typed numeric constant|, a concrete dtype is picked for it:
  integral constants are treated as `int32`, `int64`, or `uint64`, depending on the value;
  floating-point constants are treated as `float32`.
- If one of the two operands has a higher category than the other, then its concrete dtype
  is chosen as the common dtype.
- If both operands are of the same category, but one of them is a |loosely typed numeric constant|,
  then the other operand's dtype is picked as the common dtype.
- Otherwise, the common dtype is computed according to the table below.

.. rst-class:: compact-table

.. include:: generated/includes/dtype_promotion_table.rst

Tuples
------

Tuples can be used in |tile code|. They cannot be |kernel| parameters.

.. _data-rounding-modes:

Rounding Modes
--------------

.. autoclass:: cuda.tile.RoundingMode()
   :members:
   :undoc-members:
   :member-order: bysource

.. _data-padding-modes:

Padding Modes
-------------

.. autoclass:: cuda.tile.PaddingMode()
   :members:
   :undoc-members:
   :member-order: bysource
