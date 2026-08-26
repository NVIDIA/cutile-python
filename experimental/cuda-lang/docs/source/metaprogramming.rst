.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _metaprogramming-metaprogramming:

Metaprogramming
===============

|cuda lang| uses compile-time values and Python evaluation to specialize
generated SIMT code.

Compile-Time Constants
----------------------

Some facilities require values that are known at compile time. Literals and
expressions derived entirely from compile-time constants remain known to the
compiler. A value that depends on run-time thread state is dynamic.

Annotate a kernel parameter with ``cl.Constant[T]`` when its launch-time value
must be embedded in the compiled kernel. A distinct kernel is compiled for each
unique value of a constant-embedded parameter.

Static Evaluation
-----------------

``cl.static_eval`` evaluates an expression with Python semantics during
compilation. It can inspect compile-time properties of dynamic values and
select among symbolic expressions, but it cannot perform run-time GPU
operations.

Static Iteration
----------------

``cl.static_iter`` evaluates an iterable during compilation and expands the
loop body once for each item.

Static Assertions
-----------------

``cl.static_assert`` checks a condition during compilation.
``cl.ensure_constant`` requires a value to be known at compile time and returns
that value.

Static Functions
----------------

The ``@cl.static_def`` decorator marks a reusable helper function to be
evaluated with static-evaluation semantics.
