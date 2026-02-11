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


Scan
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

   cumsum
   cumprod
   scan


Matmul
------
.. autosummary::
   :toctree: generated
   :nosignatures:

   mma
   matmul


Selection
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

   where
   extract


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


Utility
-------
.. autosummary::
   :toctree: generated
   :nosignatures:

   printf
   assert_
