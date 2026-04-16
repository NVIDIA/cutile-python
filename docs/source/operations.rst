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
   gather
   scatter


.. _operations-factory:

Factory
-------
.. autosummary::
   :toctree: generated
   :nosignatures:

   arange
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
   matmul

.. _operations-selection:

Selection
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

   where
   extract


.. _operations-math:

Math
----
.. autosummary::
   :toctree: generated
   :nosignatures:

   add
   sub
   mul
   truediv
   floordiv
   cdiv
   pow
   mod
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


.. _operations-metaprogramming:

Metaprogramming Support
-----------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   static_assert
   static_eval
   static_iter


.. _operations-classes:

Classes
-------
.. autosummary::
   :nosignatures:

   Array
   TiledView

.. toctree::
   :hidden:

   data/array
   data/tiled_view


.. _operations-enums:

Enums
-----
.. autosummary::
   :nosignatures:

   RoundingMode
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
