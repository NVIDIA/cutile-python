.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

Execution Model
===============

Abstract Machine
----------------

.. _grid:

A *tile kernel* is executed by logical thread |blocks| organized in a 1D,
2D, or 3D *grid*.

.. _block:

Each *block* runs on a subset of a GPU defined by the underlying compiler implementation.
Every |block| executes the body of the |kernel|:
scalar operations run serially on a single thread, while array operations
run collectively in parallel across all threads of the |block|.

Tile programs express |block|-level parallelism only with no exposure to
individual threads within the block.

Explicit synchronization or communication within a |block| is not permitted,
but is allowed between different |blocks|.

A |block| defines the unit of execution and a |tile| defines unit of data,
which shall not be confused. A single block may operate on multiple |tiles|
with different shapes originating from different |global arrays|.


.. _execution-execution-spaces:

Execution Spaces
----------------

cuTile code runs on one or more *targets*. A target is an execution environment
defined by its hardware resources and programming model.

.. _host code:
.. _SIMT code:
.. _tile code:

The set of targets where a construct can be used is called its *execution space*.
cuTile defines three execution spaces:

- *Host code* --- all CPU targets.
- *SIMT code* --- all CUDA SIMT targets.
  (Historically called *device code*; we avoid that term to prevent ambiguity.)
- *Tile code* --- all CUDA tile targets.

Some constructs span multiple execution spaces. For example,
:py:func:`~cuda.tile.cdiv` is usable in both host code and tile code.

A function whose decorator explicitly specifies its execution space is called
an *annotated function*.

.. _execution-tile-functions:

Tile Functions
--------------

.. autoclass:: cuda.tile.function

.. _tile-kernels:
.. _execution-tile-kernels:

Tile Kernels
------------

.. autoclass:: cuda.tile.kernel
   :members: replace_hints

.. autofunction:: cuda.tile.launch

Python Subset
-------------

|Tile code| supports a subset of the Python language.
There is no Python runtime within |tile code|.

Only Python features explicitly listed in this document are supported.
Many features --- such as exceptions, and coroutines --- are not
supported today.

Object Model & Lifetimes
~~~~~~~~~~~~~~~~~~~~~~~~

All objects created within |tile code| are immutable.
Any operation that would conceptually modify an object instead creates and returns a new object.
Attributes cannot be added to objects dynamically.

Global |arrays| are views that can read and write global device memory, but the views themselves
are also immutable.

The caller of a |kernel| must ensure that:

- Array arguments obey the aliasing requirements described under
  :ref:`data-global-arrays`.
- All |arrays| remain valid until the |kernel| completes execution.

Control Flow
~~~~~~~~~~~~

Python control flow statements (``if``, ``for``, ``while``, etc.) are usable in |tile code|
and can be arbitrarily nested.

Current limitations
^^^^^^^^^^^^^^^^^^^

|Tile code| imposes additional restrictions on control flow:

* ``step`` must be strictly positive.

  Negative-step ranges such as ``range(10, 0, -1)`` are not supported today.
  Passing a negative step indirectly via a variable may cause undefined
  behavior.

* The bounds, the step and the induction variable of a ``for`` loop all share one
  integer type.

  The common type is at least ``int32`` and widens to ``int64`` if any bound or step
  is ``int64``. This also applies to the implicit lower bound and step in ``range(n)``.
  A literal bound too large for ``int32``, such as ``range(3_000_000_000)``, widens
  the loop to ``int64`` as well.

Tile Parallelism
----------------

When a |block| executes a function that takes |tiles| as parameters, it may
parallelize evaluation across the |block|'s execution resources.

Constantness
------------

.. _execution-constant-expressions-objects:

Constant Expressions & Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some facilities require parameters whose values are known at compile time.
*Constant expressions* produce *constant objects* suitable for such parameters.
Constant expressions are:

- A literal object.
- Integer arithmetic expressions where all the operands are literal objects.
- A local object or parameter that is assigned from a literal object or constant expression.
- A global object that is defined at the time of compilation or launch.

By default, numeric constants are *loosely typed*: integer constants have
infinite precision and floating-point constants are stored in IEEE 754 double
precision, until used in a context that requires a specific-width type.

A *strictly typed* constant is created by calling a dtype constructor,
e.g. ``ct.int16(5)``. Combining a strictly typed constant with a loosely typed
constant yields a strictly typed constant:
``ct.int16(5) + 2`` produces a strictly typed ``int16`` constant 7.

Combining two strictly typed constants also produces a strictly typed constant,
with the regular |type promotion| rules applied.
For example, ``ct.int16(5) + ct.int32(7)`` produces a strictly typed ``int32``
constant 12.


.. _execution-constant-embedding:

Constant Embedding
~~~~~~~~~~~~~~~~~~

If a |kernel| parameter is *constant embedded*, then:

- Every use of the parameter behaves as if replaced by its literal value.
- A distinct |machine representation| of the |kernel| is generated for each unique value of the parameter. Note: The |kernel| is compiled once per unique value, even if JIT caching is enabled.

Type Annotations
----------------

Kernel parameter type annotations use ``typing.Annotated`` metadata to control
constant embedding, array shape and index metadata, the integer dtype of scalar
parameters, and list element types.

Ordinary type hints such as ``int``, ``float``, or ``ct.int32`` do not enforce
or convert parameter types. Use an explicit conversion such as ``ct.int32(x)``
when a specific scalar dtype is needed. In ``ct.Constant[int]``, ``ct.Constant``
marks the argument as constant; the inner ``int`` does not validate or convert it.

Tuple annotations describe the argument structure and can apply supported
metadata to individual elements, for example ``tuple[ct.Array, ct.Constant[int]]``.
Ordinary element type hints still do not enforce or convert element types.
Annotations on helper parameters, local variables, and return values do not
control kernel argument conversion.

Constant Annotations
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../test/test_constant.py
   :language: python
   :dedent:
   :start-after: example-begin imports
   :end-before: example-end imports

.. literalinclude:: ../../test/test_constant.py
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
