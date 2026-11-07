.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.tile

.. _data-data-model:

Data Model
==========

cuTile is an array-based programming model.
The fundamental data structure is the multidimensional array whose elements share a single
homogeneous type.

An array-based model was chosen because:

- Arrays know their bounds, so accesses can be checked for safety and correctness.
- Array metadata lets the compiler select memory instructions, including TMA
  transfers when the target architecture and access pattern support them.
- Python programmers are already familiar with array-based frameworks such as NumPy.

Within |tile code|, only the types described in this section are supported.

A typical tile kernel loads tiles from global arrays, computes with those
values, and stores the resulting tiles back into global arrays. Tiles support
elementwise arithmetic, broadcasting, shape manipulation, reductions, scans,
and matrix multiplication; each operation produces new tile values. See
:doc:`operations` for the operation catalog.


.. _data-global-arrays:

Global Arrays
-------------

A *global array* (or *array*) is a container of elements of a specific |dtype|
arranged in a logical multidimensional space.

An array's *shape* is a tuple of integers, each denoting the length of the
corresponding dimension. The length of the shape tuple equals the array's number
of dimensions, and the product of its values equals the total number of logical
elements in the array.

Arrays are stored in global memory using a *strided memory layout*: in addition to a shape,
each array has an equally sized tuple of *strides* that maps logical indices to physical memory
locations. For example, for a 3-dimensional `float32` array with strides `(s1, s2, s3)`, the
memory address of the element at index `(i1, i2, i3)` is:

.. code-block::

    base_addr + 4 * (s1 * i1 + s2 * i2 + s3 * i3),

where ``base_addr`` is the base address of the array and ``4`` is the byte size of a ``float32``
element.

New arrays can only be allocated by the host and passed to the tile kernel as arguments.
|Tile code| can only create new views of existing arrays, for example via
:meth:`Array.slice`. Assigning an array to another variable creates another
reference to the same array.

Any object that implements the |DLPack| interface or the |CUDA Array Interface|
can be passed as a kernel argument --- for example, |CuPy| arrays and |PyTorch| tensors.

When launching from Python with :py:func:`cuda.tile.launch`, distinct array arguments
must not overlap in memory, including arrays passed inside tuples. Otherwise, the
behavior is undefined. For ahead-of-time compilation, the permitted aliasing is
specified by :py:class:`cuda.tile.compilation.ArrayConstraint`.

An array's shape and strides can be queried with :py:attr:`Array.shape` and
:py:attr:`Array.strides`. Their scalar values use ``int32`` by default.
:py:data:`cuda.tile.IndexedWithInt64` or
:py:obj:`index_dtype=ct.int64 <cuda.lang.int64>` in an
:py:class:`cuda.tile.ArrayAnnotation` enables indexing for arrays whose shape
or stride values exceed the range of ``int32``.

Shape and stride values can be runtime values or compile-time constants.
``static_shape_dims`` and ``static_stride_dims`` in
:py:class:`cuda.tile.ArrayAnnotation` specialize selected dimensions to their
launch-time values. The dispatcher can also infer constant dimensions, such as
shape or stride values equal to one.


.. seealso::
  :ref:`cuda.tile.Array class documentation <data-array-cuda-tile-array>`

.. _data-tiles-and-scalars:

Tiles and Scalars
-----------------
A *tile* is an immutable multidimensional collection of elements of a specific |dtype|.

A tile's *shape* is a tuple of integers, each denoting the length of the corresponding dimension.
The length of the shape tuple equals the tile's number of dimensions, and the product of its values
equals the total number of elements in the tile.

The shape of a tile must be known at compile time. Each dimension of a tile must be a power of 2.

A tile's dtype and shape can be queried with the ``dtype`` and ``shape`` attributes, respectively.
For example, if ``x`` is a `float32` tile, ``x.dtype`` returns a compile-time constant
equal to :py:data:`cuda.tile.float32`.

A zero-dimensional tile is called a *scalar*. A scalar has exactly one element and its shape is
the empty tuple `()`. Numeric literals like `7` or `3.14` are treated as constant scalars,
i.e. zero-dimensional tiles.

Because scalars are tiles, they differ slightly from Python's ``int``/``float`` objects.
For example, they have ``dtype`` and ``shape`` attributes:

.. code-block:: python

    a = 0
    # The following line will evaluate to cuda.tile.int32 in cuTile,
    # but would raise an AttributeError in Python:
    a.dtype

Tiles are values used in |tile code|. The compiler chooses how to represent
them and distribute their elements across threads.
Tiles are created by loading from |global arrays| using functions such as
:py:func:`cuda.tile.load` and :py:func:`cuda.tile.gather`, or with |factory| functions
such as :py:func:`cuda.tile.zeros`.

Tiles can be stored back into global arrays using functions such as :py:func:`cuda.tile.store`
and :py:func:`cuda.tile.scatter`.

Scalar constants are |loosely typed| by default, for example, a literal ``2`` or
a constant attribute like :py:obj:`Tile.ndim <cuda.tile.Tile.ndim>`, :py:obj:`Tile.shape <cuda.tile.Tile.shape>`, or :py:obj:`Array.ndim <cuda.tile.Array.ndim>`.

.. seealso::
  :ref:`cuda.tile.Tile class documentation <data-tile-cuda-tile-tile>`

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

The *element space* of an array is the multidimensional space of its elements, stored in memory
according to a given layout (row-major, column-major, etc.).

The *tile space* of an array is the multidimensional space of tiles of a given tile shape within
that array. A tile index ``(i, j, ...)`` with shape ``S`` refers to the elements belonging to the
``(i+1)``-th, ``(j+1)``-th, ... tile.

When accessing array elements via tile indices, the array's multidimensional memory layout is used.
The ``order`` parameter of load/store operations selects a different tile-space layout.

For example, :py:obj:`ct.load(array, index=(i,), shape=(128,)) <cuda.tile.load>` loads a tile starting
at element ``i * 128``. Its index is in tile space, whereas a SIMT expression
such as ``array[i]`` accesses an individual element.

Loading and Storing Tiles
-------------------------

:py:func:`cuda.tile.load` and :py:func:`cuda.tile.store` check array boundaries
by default. Bounds checking handles partial edge tiles. A load's explicit
``padding_mode`` defines the padded values used by a computation. See the
:doc:`vector-add quickstart </getting-started/tile>` for an example.

:py:func:`cuda.tile.gather` and :py:func:`cuda.tile.scatter` provide
element-based indexing for less regular access patterns.

.. _data-tiled-views:

Tiled Views
-----------

A *tiled view* represents the |tile space| of a |global array|.

A tiled view's *num_tiles* is a tuple of integers, each denoting the number of tiles along
the corresponding dimension. The length of the *num_tiles* tuple equals the tile space's number
of dimensions, and the product of its values equals the total number of tiles in the tile space.

A tile in the tiled view can be loaded or stored by its tile index.

By default, consecutive tiles along each axis are adjacent with no overlap or gaps: the origin of
each successive tile advances by ``tile_shape[i]`` elements along axis *i*. Specifying
``traversal_steps`` to :meth:`Array.tiled_view` changes the advance per step to
``traversal_steps[i]``, producing overlapping tiles when ``traversal_steps[i] < tile_shape[i]``
or gapped tiles when ``traversal_steps[i] > tile_shape[i]``.

.. seealso::
  :ref:`cuda.tile.TiledView class documentation <data-tiled-view-cuda-tile-tiled-view>`

  :meth:`Array.tiled_view`

Shape Broadcasting
------------------

*Shape broadcasting* allows |tiles| with different shapes to be combined in arithmetic operations.
When an operation involves |tiles| of different shapes, the smaller |tile| is automatically
extended to match the larger one, following these rules:

- |Tiles| are aligned by their trailing dimensions.
- If the corresponding dimensions have the same size or one of them is 1, they are compatible.
- If one |tile| has fewer dimensions, its shape is padded with 1s on the left.

Broadcasting follows the same semantics as |NumPy|, keeping code concise and readable
while maintaining computational efficiency.

Numeric data types, arithmetic promotion, and rounding modes are shared with
``cuda.lang``. See :ref:`language-numeric-types`.

Tuples
------

Tuples can be constructed and used in |tile code| and passed as |kernel|
arguments. Tuple arguments may be nested and contain supported argument types,
such as scalars and arrays. Their structure is known at compile time; their
elements need not be constants. Tuple indexing requires a compile-time constant
index.

Parameter annotations can describe individual elements, for example
``tuple[ct.Array, ct.Constant[int]]``. A parameter annotated with
:py:obj:`ct.Constant[tuple] <cuda.tile.Constant>` makes the whole tuple constant, so every element must be
representable as a compile-time constant.

For ahead-of-time export, tuple parameters use
:py:class:`cuda.tile.compilation.TupleConstraint` and require the
``cutile_python_v2`` calling convention. See :ref:`compilation-callconv`.

Lists of Arrays
---------------

A non-empty Python list of arrays can be passed as a kernel argument. All
elements must have the same rank and dtype. In |tile code|, this becomes a
list view supporting ``len`` and integer indexing, including runtime indices.
The list structure is immutable; its arrays support loads and stores.
Creating or mutating ordinary Python lists in |tile code| is unsupported.

Launching kernels with list-of-array arguments during CUDA graph capture is
currently unsupported.

:py:class:`cuda.tile.ListAnnotation` configures the element arrays'
indexing types and static dimensions.

.. _data-padding-modes:

Padding Modes
-------------

.. autoclass:: cuda.tile.PaddingMode()
   :members:
   :undoc-members:
   :member-order: bysource
