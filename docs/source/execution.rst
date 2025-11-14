.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

Execution Model
===============

Abstract Machine
----------------

.. _grid:

A *tile kernel* is executed by logical thread |blocks| that are organized in
a 1D, 2D, or 3D *grid*.

.. _block:

Each *block* is executed by a subset of a GPU, which is decided by the
implementation, not the programmer.
Each |block| executes the body of the |kernel|.
Scalar operations are executed serially by a single thread of the |block|,
and array operations are collectively executed in parallel by all threads of
the |block|.

Tile programs explicitly describe for |block|-level parallelism, but not
thread-level parallelism.
Threads cannot be explicitly identified or manipulated in tile programs.

Explicit synchronization or communication within a |block| is not
permitted, but it is allowed between different |blocks|.

It is important to not confuse |blocks| (units of execution) with
|tiles| (units of data).
A block may work with multiple different |tiles| with
differing shapes originating from differing |global arrays|.

.. Comparison with SIMT
.. --------------------
.. 
.. TODO

Execution Spaces
----------------

cuTile code is executed on one or more *targets*, which are distinct execution environments that
are distinguished by different hardware resources or programming models.

A function is *usable* if it can be called.
A type or object is *usable* if its attributes are accessible (can be read and written) and its
methods are callable.

.. _host code:
.. _SIMT code:
.. _tile code:

Some functions, types, and objects are only usable on certain *targets*.
The set of *targets* that such a construct is usable on is called its *execution space*.

- *Host code* is the execution space that includes all CPU targets.
- *SIMT code* is the execution space that includes all CUDA SIMT targets.
     Note: This has historically been called device code, but we avoid this term to prevent ambiguity.
- *Tile code* is the execution space that includes all CUDA tile targets.

Functions can have decorators that explicitly specify their execution space.
These are called *annotated functions*.

Tile Functions
--------------

.. autoclass:: cuda.tile.function

Tile Kernels
------------

.. autoclass:: cuda.tile.kernel

.. autofunction:: cuda.tile.launch

Python Subset
-------------

|Tile code| supports a subset of the Python language.
Within |tile code|, there is no Python runtime.

Only Python features explicitly enumerated in this document are supported.
Many features, such as lambdas, exceptions, and coroutines are not supported today.

Object Model & Lifetimes
~~~~~~~~~~~~~~~~~~~~~~~~

All objects created within |tile code| are immutable.
Any operation that conceptually modifies an object or its attributes shall create and return a new
object.
Attributes shall not be dynamically added to objects.

The only mutable objects that can be used in |tile code| are |arrays|, which must be passed in as
|kernel| parameters.
The caller of a |kernel| shall ensure that any |arrays| passed to the |kernel| shall not be
destroyed until the |kernel| has finished execution.

Control Flow
~~~~~~~~~~~~

Python control flow statements (``if``, ``for``, ``while``, etc.) shall be usable in |tile code|.
They can be arbitrarily nested.

Tile Parallelism
----------------

When a |block| executes a function that takes |tiles| as parameters, it may parallelize the
evaluation of the function across the |block|'s execution resources.
Unless otherwise specified, the execution shall complete before the function returns.

Constantness
------------

Constant Expressions & Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some facilities require certain parameters to be an object that is known statically at compilation time.
*Constant expressions* produce *constant objects* suitable for such parameters. Constant expressions are:

- A literal object.
- Integer arithmetic expressions where all the operands are literal objects.
- A local object or parameter that is assigned from a literal object or constant expression.
- A global object that is defined at the time of compilation or launch.

By default, numeric constants are *loosely typed*: until used in a context that requires
a type of specific width, integer constants have infinite precision, and floating-point
constants are stored in the IEEE 754 double precision format.

A *strictly typed* constant can be created by calling a dtype object as a constructor,
e.g. ``ct.int16(5)`` creates a strictly typed ``int16`` constant. When a strictly typed constant
is combined with a loosely typed constant, the result is a strictly typed constant.
For example ``ct.int16(5) + 2`` will create a strictly typed ``int16`` constant 7.

Combining two strictly typed constants creates a new strictly typed constant. In this case,
the regular |type promotion| rules apply.
For example, ``ct.int16(5) + ct.int32(7)`` will create a strictly typed ``int32`` constant 12.


Constant Embedding
~~~~~~~~~~~~~~~~~~

If a parameter to a |kernel| is *constant embedded*, then:

- All uses of the parameter shall act as if they were replaced by the literal value of the parameter.
- There shall be a distinct machine representation of the |kernel| for each different value of the parameter that the |kernel| is invoked with. Note: The |kernel| shall be compiled once for each different value of the parameter, even if JIT caching is enabled.
- The |machine representation| of the parameter shall be 0 bytes.

Constant Type Hints
~~~~~~~~~~~~~~~~~~~

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
