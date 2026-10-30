.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.tile

.. _operations-operations:

Operations
==========

.. _operations-load-store:

Load/Store
----------
.. autosummary::
   :toctree: generated
   :nosignatures:

   bid
   num_blocks
   num_tiles
   load
   store
   load_advanced_indexing
   store_advanced_indexing
   gather
   scatter


.. _operations-factory:

Factory
-------
.. autosummary::
   :toctree: generated
   :nosignatures:

   arange
   astile
   full
   ones
   zeros


.. _operations-shape-dtype:

Shape & DType
-------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   cat
   broadcast_to
   expand_dims
   reshape
   permute
   transpose
   astype
   bitcast
   pack_to_bytes
   unpack_from_bytes


.. _operations-reduction:

Reduction
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sum
   max
   min
   prod
   argmax
   argmin
   reduce


.. _operations-scan:

Scan
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

   cumsum
   cumprod
   scan


.. _operations-matmul:

Matmul
------
.. autosummary::
   :toctree: generated
   :nosignatures:

   mma
   mma_scaled
   matmul

.. _operations-selection:

Selection
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

   where
   extract
   insert


.. _operations-math:

Math
----
.. autosummary::
   :toctree: generated
   :nosignatures:

   add
   sub
   mul
   wrapping_add
   wrapping_sub
   wrapping_mul
   truediv
   floordiv
   cdiv
   pow
   atan2
   mod
   divmod
   minimum
   maximum
   negative
   abs
   isnan

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
   floor
   ceil


.. _operations-bitwise:

Bitwise
-------
.. autosummary::
   :toctree: generated
   :nosignatures:

   bitwise_and
   bitwise_or
   bitwise_xor
   bitwise_lshift
   bitwise_rshift
   bitwise_not


.. _operations-comparison:

Comparison
----------
.. autosummary::
   :toctree: generated
   :nosignatures:

   greater
   greater_equal
   less
   less_equal
   equal
   not_equal

.. _operations-atomic:

Atomic
------
.. autosummary::
   :toctree: generated
   :nosignatures:

   atomic_cas
   atomic_xchg
   atomic_add
   atomic_max
   atomic_min
   atomic_and
   atomic_or
   atomic_xor

.. _operations-utility:

Utility
-------
.. autosummary::
   :toctree: generated
   :nosignatures:

   printf
   print
   assert_
   assume_divisible_by
   grid_dependency_control_launch_dependents
   grid_dependency_control_wait


.. _operations-metaprogramming:

Metaprogramming Support
-----------------------

See :ref:`language-static-evaluation` for compile-time evaluation semantics and
loop-unrolling examples.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ensure_constant
   static_assert
   static_eval
   static_exception
   static_iter

.. _operations-classes:

Classes
-------

.. autoclass:: cuda.tile.DType()
   :members:

.. autosummary::
   :nosignatures:

   Array
   Tile
   TiledView
   Slice

.. toctree::
   :hidden:

   data/array
   data/tile
   data/tiled_view
   data/slice


.. _tile-operations-annotations:

Parameter Annotations
---------------------

These ``cuda.tile`` annotations control how Python arguments are converted at
a kernel boundary. See :ref:`language-parameter-annotations` for the language
rules and :ref:`lang-operations-annotations` for the ``cuda.lang`` annotation APIs.

Constant Annotations
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../../test/test_constant.py
   :language: python
   :dedent:
   :start-after: example-begin imports
   :end-before: example-end imports

.. literalinclude:: ../../../../../test/test_constant.py
   :language: python
   :dedent:
   :start-after: example-begin constant
   :end-before: example-end constant

.. autoclass:: cuda.tile.ConstantAnnotation

.. autodata:: cuda.tile.Constant

Array Annotations
~~~~~~~~~~~~~~~~~

.. autoclass:: cuda.tile.ArrayAnnotation

.. autodata:: cuda.tile.IndexedWithInt64

Scalar Annotations
~~~~~~~~~~~~~~~~~~

.. autodata:: cuda.tile.ScalarInt64

List Annotations
~~~~~~~~~~~~~~~~

.. autoclass:: cuda.tile.ListAnnotation


.. _operations-enums:

Enums
-----
.. autosummary::
   :nosignatures:

   ~cuda.lang.RoundingMode
   PaddingMode


.. _operations-tuning:

Autotuning
----------
.. autosummary::
   :nosignatures:

   tune.exhaustive_search

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/dataclass_no_init.rst

   tune.TuningResult
   tune.Measurement

.. autosummary::
   :nosignatures:

   kernel.replace_hints
   compiler_timeout


.. _operations-jax:

JAX FFI
-------
.. autosummary::
   :toctree: generated
   :nosignatures:

   jax.cutile_call

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/dataclass_no_init.rst

   jax.OutputPlaceholder
   jax.InputOutput
