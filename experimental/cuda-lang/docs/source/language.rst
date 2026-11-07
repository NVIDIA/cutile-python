.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _language:

Language Guide
==============

|cuda.lang| and |cuda.tile| share a consistent language design. This guide uses
|cuda.lang| examples to explain the shared concepts. Shared language APIs,
including ``static_eval``, ``static_def``, ``static_iter``, and
``ensure_constant``, are available through both namespaces.

The language has two evaluation modes with distinct Python semantics:

* **Default mode** compiles a Python subset for runtime execution, including
  control flow and device operations. Constant folding propagates compile-time
  values through scalar expressions.
* **Static evaluation** runs ordinary Python during compilation, including
  branches, loops, and helper calls. ``static_eval``
  and ``@static_def`` explicitly request these semantics.

Constants enter static evaluation as ordinary Python values; runtime values
enter as symbolic objects. Static evaluation can return constants or supported
symbolic expressions for default mode to compile. See
:ref:`language-metaprogramming` for examples and restrictions.

.. _language-execution-spaces:

.. _execution-execution-spaces:

Execution Spaces
----------------

CUDA Languages code runs on one or more *targets*. A target is an execution environment
defined by its hardware resources and programming model.

.. _host code:
.. _SIMT code:
.. _tile code:

The set of targets where a construct can be used is called its *execution space*.
CUDA Languages defines three execution spaces:

- *Host code* --- all CPU targets.
- *SIMT code* --- all CUDA SIMT targets.
- *Tile code* --- all CUDA tile targets.

Some constructs span multiple execution spaces. For example,
:py:func:`cl.cdiv <cuda.lang.cdiv>` is usable in both host code and SIMT code.

A function whose decorator explicitly specifies its execution space is called
an *annotated function*.

Default Mode
------------

The body of a :py:obj:`@cl.kernel <cuda.lang.kernel>` or :py:obj:`@cl.host_entry <cuda.lang.host_entry>` function starts in default
mode. This mode supports a subset of Python, where statements have runtime
semantics. A kernel runs on the GPU; a host entry runs as compiled CPU code.
See :ref:`language-supported-python-features` for the available Python subset.

.. testcode:: language-runtime-print
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def print_values(values, n):
       i = cl.thread_index(0)
       if i < n:
           print(values[i])
   # end-snippet

   values = torch.tensor([7, 9], dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (2,), print_values, (values, 1))

.. testoutput:: language-runtime-print

   7

The ``if`` tests a runtime value, and array indexing loads GPU memory.
``print(...)`` is a runtime print. Each GPU thread checks its index. Threads
with ``i < n`` load and print an array element at runtime.


.. _language-captured-values:

Capturing Python Values
~~~~~~~~~~~~~~~~~~~~~~~

A :py:obj:`@cl.kernel <cuda.lang.kernel>` or :py:obj:`@cl.host_entry <cuda.lang.host_entry>` function can reference values from module
globals or an enclosing Python function. Values that the compiler can represent
as constants in default mode are captured when the function is compiled.

You can use this to write a factory that creates kernels with different
configuration values:

.. testcode:: language-captured-values
   :template: language_example.py

   # begin-snippet
   def make_scale_kernel(factor):
       @cl.kernel
       def scale_values(values, out):
           i = cl.thread_index(0)
           out[i] = values[i] * factor

       return scale_values

   double_values = make_scale_kernel(2)
   triple_values = make_scale_kernel(3)
   # end-snippet

   values = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device="cuda")
   out = torch.empty_like(values)
   cl.launch(stream, (1,), (4,), double_values, (values, out))
   assert out.cpu().tolist() == [2, 4, 6, 8]
   cl.launch(stream, (1,), (4,), triple_values, (values, out))
   assert out.cpu().tolist() == [3, 6, 9, 12]

The factory runs in ordinary Python and returns a kernel with its own captured
``factor``. When launched through :py:obj:`cl.launch <cuda.lang.launch>`, ``double_values`` compiles with
the constant ``2``, and ``triple_values`` compiles with the constant ``3``.
The array loads, multiplication, and stores still have runtime semantics.
The same capture rules apply when a :py:obj:`@cl.host_entry <cuda.lang.host_entry>` is compiled.

Capture requires both a value known during compilation and a representation
as a constant in default mode. Examples include Python numbers, tuples of such
values, and frozen dataclasses whose fields meet the same requirements.
GPU tensors enter compiled code as runtime :py:obj:`Array <cuda.lang.Array>` arguments.
Captured values remain fixed in already compiled code.

.. _language-constants:

Constants and Constant Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A constant is a value known during compilation. In default mode:

* A :py:obj:`cl.Constant <cuda.lang.Constant>` parameter annotation on :py:obj:`@cl.kernel <cuda.lang.kernel>` or :py:obj:`@cl.host_entry <cuda.lang.host_entry>`
  marks its argument as a compile-time constant. See
  :ref:`language-parameter-annotations` for the supported annotation rules.
* Literals such as ``3``, ``2.0``, and ``True`` are constants.
* Values :ref:`captured from outside the compiled function
  <language-captured-values>` are constants.
* Assigning a constant to a local variable preserves its constantness.
* Arithmetic operations on scalar constants produce constants too.

For example, these assignments inside a kernel all produce constants:

.. testcode:: language-constant-folding
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def check_constants(out):
       width = 4
       height = 8
       area = width * height  # Compile-time constant: 32.
       out[0] = cl.ensure_constant(area)
   # end-snippet

   out = torch.empty(1, dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (1,), check_constants, (out,))
   assert out.cpu().item() == 32

:py:obj:`cl.ensure_constant(value) <cuda.tile.ensure_constant>` asserts that a value is constant and returns
its compile-time constant value. Compilation fails for runtime values;
:py:obj:`ensure_constant <cuda.tile.ensure_constant>` does not make them constant.

An ``if`` whose condition is constant selects a branch during compilation and
discards the other branch. :py:obj:`ensure_constant <cuda.tile.ensure_constant>` enforces this
requirement on the condition:

.. testcode:: language-constant-branch
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def choose_value(out, use_large: cl.Constant[bool]):
       if cl.ensure_constant(use_large):
           out[0] = 100
       else:
           out[0] = 10
   # end-snippet

   out = torch.empty(1, dtype=torch.int32, device="cuda")
   for flag, expected in ((True, 100), (False, 10)):
       cl.launch(stream, (1,), (1,), choose_value, (out, flag))
       assert out.cpu().item() == expected

Only the selected store is compiled. The store still runs on the GPU; there is
no runtime branch. A plain ``if use_large:`` also eliminates the unused branch
because ``use_large`` is constant.

.. _language-constant-types:

Loosely Typed and Strictly Typed Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _execution-constant-expressions-objects:

Constantness and numeric typing are separate properties.

By default, numeric constants are *loosely typed*: integer constants have
infinite precision and floating-point constants are stored in IEEE 754 double
precision, until used in a context that requires a specific-width type.

A *strictly typed* constant is created by calling a dtype constructor,
e.g. :py:obj:`cl.int16(5) <cuda.lang.int16>`. Combining a strictly typed constant with a loosely typed
constant yields a strictly typed constant:
``cl.int16(5) + 2`` produces a strictly typed ``int16`` constant 7.

Combining two strictly typed constants also produces a strictly typed constant,
with the regular |type promotion| rules applied.
For example, ``cl.int16(5) + cl.int32(7)`` produces a strictly typed ``int32``
constant 12.

See :ref:`language-numeric-types` for numeric dtypes and arithmetic promotion.

.. _language-numeric-types:

Data Types and Promotion
------------------------

``cuda.lang`` and ``cuda.tile`` use common dtype definitions and arithmetic promotion rules.
The reference below lists the names each namespace exposes. The reference below
lists public names from both namespaces; examples use ``cuda.lang``, imported
as ``cl``. See :ref:`language-constant-types` for loosely and strictly typed
constants.

.. _data-data-types:

Data Types
~~~~~~~~~~

A dtype describes the type of a value or an array's elements. Numeric dtypes
are available as objects such as :py:obj:`cl.int32 <cuda.lang.int32>` and :py:obj:`cl.float32 <cuda.lang.float32>`. Each has a
``name`` and a ``bitwidth``:

.. code-block:: python

   cl.float32.name      # "float32"
   cl.float32.bitwidth  # 32

.. include:: /generated/includes/numeric_dtypes.rst

.. _data-numeric-arithmetic-data-types:

Numeric & Arithmetic Data Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A *numeric* data type represents numbers. An *arithmetic* data type is a numeric data type
that supports general arithmetic operations such as addition, subtraction, multiplication,
and division.

.. _data-packed-fp4-storage:

Packed FP4 Storage
~~~~~~~~~~~~~~~~~~

``cuda.lang`` provides packed FP4 pairs as
:py:data:`cl.float4_e2m1fn_x2 <cuda.lang.float4_e2m1fn_x2>`.
``cuda.tile`` provides individual FP4 elements as
:py:data:`ct.float4_e2m1fn <cuda.tile.float4_e2m1fn>`.

.. autodata:: cuda.lang.float4_e2m1fn_x2
   :no-value:

The packed dtype's ``bitwidth`` is ``8``.
Array shapes, strides, pointer offsets, and vector element counts use packed
pairs. For example, ``cl.shared_array(32, cl.float4_e2m1fn_x2)`` holds 64 FP4
values in 32 bytes.

Packed values support loads, stores, and :py:func:`cl.bitcast <cuda.lang.bitcast>`
to and from :py:data:`cl.uint8 <cuda.lang.uint8>`. Arithmetic and numeric
conversions are currently unsupported. A PyTorch byte tensor can be passed as
``storage.view(torch.float4_e2m1fn_x2)``; its shape and strides are preserved
when entering a SIMT kernel.

MMA instruction descriptors use this dtype to select E2M1 operands. Their
M, N, and K dimensions count individual FP4 values. The `PTX operand layouts
<https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-packing-formats>`_
specify the packing and padding required by each MMA family.


.. _data-arithmetic-promotion:

Arithmetic Promotion
~~~~~~~~~~~~~~~~~~~~

Arithmetic operations on numeric operands of different dtypes use the
following promotion rules.

When both operands are |loosely typed numeric constants|, the result is also
a loosely typed constant: ``5 + 7`` is a loosely typed integral constant 12,
and ``5 + 3.0`` is a loosely typed floating-point constant 8.0.

If any of the operands is not a |loosely typed numeric constant|, both are *promoted*
to a common dtype as follows:

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

.. include:: /generated/includes/dtype_promotion_table.rst

.. _data-rounding-modes:

Rounding Modes
~~~~~~~~~~~~~~

:class:`~cuda.lang.RoundingMode` selects the rounding behavior of floating-point
operations. Its API reference describes the available modes.

.. _language-helper-functions:

Helper Functions
----------------

An ordinary Python helper called from default mode is compiled with the same
runtime semantics as its caller. No extra decorator or annotation is required:

.. code-block:: python

   def add_pair(x, y):
       return x + y

   @cl.kernel
   def add_values(a, b, out):
       i = cl.thread_index(0)
       out[i] = add_pair(a[i], b[i])

The compiler inlines ``add_pair`` at the call site. Each thread adds its input
values when the kernel runs. Helpers must use the same supported Python subset as kernels.
Loops support repetitions whose count depends on runtime data; see
:ref:`language-recursion` for the limits on recursive helpers.

:py:obj:`@cl.function <cuda.lang.function>` explicitly marks a helper's execution spaces. Its
defaults are ``simt=True, host=False, tile=False``: the helper can be called from
SIMT code, while direct Python calls and tile compilation are disabled.
``host=True`` permits direct Python calls, and ``simt=False`` disables SIMT
compilation. ``tile=True`` also permits calls from ``cuda.tile`` code.
:py:obj:`@cl.function(tile=True, simt=True) <cuda.lang.function>` supports calls
from both device compilers.

.. autofunction:: cuda.lang.function

Static Evaluation
-----------------

``cl.static_eval(expression)`` evaluates Python during compilation and
returns its result to default mode. For example,
``cl.static_eval(sorted((3, 1, 2))[0])`` returns the constant ``1``.
See :ref:`language-metaprogramming` for compile-time helpers, loop unrolling,
symbolic values, and their restrictions.

.. _execution-constant-embedding:

.. _language-jit-compilation:

JIT Compilation
---------------

CUDA languages use just-in-time (JIT) compilation: they compile a kernel when it
is first launched and a :py:obj:`@cl.host_entry <cuda.lang.host_entry>` function when it is first called. Each
compiled version is a **specialization** for its argument types and compile-time
constants. Later calls with matching arguments reuse that version.
Specializing a parameter by its constant value is also called *constant embedding*.

Changing a constant argument selects a specialization for the new
value. If that specialization has not been compiled yet, it is compiled on
demand. Returning to a previously used constant value can reuse its cached
specialization.

An array's rank (number of dimensions) and element dtype are compile-time
metadata. Changing either requires a matching specialization, even though the
array's contents remain runtime data. Supported annotations can add further
specialization properties, such as constant shape dimensions.

:ref:`lang-specialization-visibility` describes compilation messages, logs, and
cache inspection.

.. _language-entry-arguments:

Compilation Entry Points
~~~~~~~~~~~~~~~~~~~~~~~~

Kernels and :py:obj:`@cl.host_entry <cuda.lang.host_entry>` functions are
compilation entry points. Arguments passed from Python must have a supported
representation for compiled code. :ref:`language-supported-python-features`
describes the value types available within compiled code.
The |cuda.lang| entry points accept:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Python value
     - ``@cl.kernel`` via ``cl.launch``
     - ``@cl.host_entry``
   * - ``bool``, ``int``, ``float``
     - Scalar
     - Scalar
   * - CUDA array, such as a PyTorch tensor or CuPy array
     - ``Array`` view
     - ``Array`` view
   * - Plain tuple
     - Supported elements, recursively
     - Supported elements, recursively
   * - Frozen dataclass record
     - Supported fields, recursively
     - Supported fields, recursively
   * - ``TensorMap``
     - Read-only descriptor pointer
     - Currently unsupported
   * - CUDA stream object
     - Unsupported as a kernel argument
     - Stream value

Tuple lengths, record types, and field structure are known during compilation;
their scalar and array contents can remain runtime data. Named tuples and other
tuple subclasses are currently unsupported as entry arguments.

Numeric arguments become runtime scalars unless annotated with
:py:obj:`Constant <cuda.lang.Constant>`. Strings, ``None``, enum members, and
dtype objects require ``Constant`` annotations when passed from Python.
Frozen record arguments, ``TensorMap`` arguments, stream objects, and these
nonnumeric constant arguments currently require a build with development
features enabled (``ENABLE_DEV_FEATURES=ON``).

Helper functions can be referenced from a compiled body through captured names,
and passed between inlined helpers. Entry arguments cannot contain
functions, including parameters annotated with ``Constant``. Pointers and
vectors are constructed inside compiled code. A compiled host function can
also create a ``TensorMap`` and pass it to a kernel; see
:ref:`data-tensor-map-descriptor-pointers`.

The scalar, array, and tuple conversions also apply to |cuTile|. Tile kernels
additionally accept :doc:`lists of arrays </tile/data>`; lists are currently
unsupported at either |cuda.lang| entry boundary. Tiles are constructed inside
tile kernels. See :doc:`/reference/feature-gaps` for other differences.

.. _language-parameter-annotations:

.. _converting-python-arguments:

Parameter Annotations
~~~~~~~~~~~~~~~~~~~~~

When Python launches a :py:obj:`@cl.kernel <cuda.lang.kernel>` through :py:obj:`cl.launch <cuda.lang.launch>`, or calls a
:py:obj:`@cl.host_entry <cuda.lang.host_entry>`, arguments cross from the Python interpreter into compiled
code. Supported Python values are converted to |cuda.lang| values. For example,
a GPU tensor becomes an :py:obj:`Array <cuda.lang.Array>` view, and a numeric argument becomes a scalar.
Only supported parameter annotations, such as :py:obj:`cl.Constant <cuda.lang.Constant>`, affect this
conversion at a compilation boundary: :py:obj:`@cl.kernel <cuda.lang.kernel>` or :py:obj:`@cl.host_entry <cuda.lang.host_entry>`.

:py:obj:`cl.Constant <cuda.lang.Constant>` marks an argument as a compile-time constant:

.. testcode:: language-constant-argument
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def add_offset(values, offset: cl.Constant[int]):
       cl.static_eval(print("Compiling offset =", offset))
       i = cl.thread_index(0)
       values[i] = values[i] + offset

   values = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (4,), add_offset, (values, 3))
   assert values.cpu().tolist() == [4, 5, 6, 7]
   cl.launch(stream, (1,), (4,), add_offset, (values, 5))
   assert values.cpu().tolist() == [9, 10, 11, 12]
   cl.launch(stream, (1,), (4,), add_offset, (values, 3))
   assert values.cpu().tolist() == [12, 13, 14, 15]
   # end-snippet

.. testoutput:: language-constant-argument

   Compiling offset = 3
   Compiling offset = 5

Here, ``offset`` is known during compilation, while the array elements are
runtime values. The compiler creates a separate specialization for each distinct
``offset`` value. The compile-time prints show two compilations; the final
launch reuses the specialization for ``offset=3``.
:py:obj:`Constant <cuda.lang.Constant>` can also be used on :py:obj:`@cl.host_entry <cuda.lang.host_entry>` parameters.
See the :ref:`cuda.lang <lang-operations-annotations>` and
:ref:`cuda.tile <tile-operations-annotations>` references for parameter annotation APIs.
Other supported annotations are shared with |cuTile|:

* :py:data:`cl.ScalarInt64 <cuda.lang.ScalarInt64>`
  makes an integer parameter use ``int64``.
* :py:class:`cl.ArrayAnnotation <cuda.lang.ArrayAnnotation>`
  is used with ``typing.Annotated`` to control array indexing types and mark
  selected shape or stride dimensions as compile-time constants.

Both namespaces expose these annotations.

.. note::

   Ordinary scalar type hints, such as ``x: int``, ``x: float``, and
   ``x: cl.int32``, do not enforce or convert types. Explicit conversions such as
   :py:obj:`cl.int32(x) <cuda.lang.int32>` produce values of a particular scalar type.

   In :py:obj:`cl.Constant[int] <cuda.lang.Constant>`, :py:obj:`cl.Constant <cuda.lang.Constant>` marks the argument as constant.
   The inner ``int`` is a regular Python type hint.

   Tuple annotations describe the argument structure and can apply supported
   metadata to individual elements, for example
   ``tuple[cl.Array, cl.Constant[int]]``. The same scalar type-hint rules apply
   to tuple elements.

   Supported annotations affect compilation only on parameters at a
   compilation boundary. Helper parameter annotations, local variable
   annotations, and return annotations serve as Python type hints.

Separate Host and Kernel Specializations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JIT compilation of a :py:obj:`@cl.host_entry <cuda.lang.host_entry>` and the kernels it launches happens
separately. Calling the host entry specializes the CPU function for its own
arguments. Each :py:obj:`cl.launch <cuda.lang.launch>` inside it independently selects or compiles a GPU
kernel specialization for the arguments passed to that launch. One host
specialization can therefore launch several kernel specializations.

For a :py:obj:`cl.launch <cuda.lang.launch>` inside compiled host code, the kernel parameter's annotation
and the argument's constantness determine whether its value specializes the
kernel:

* **Parameter annotated with** :py:obj:`cl.Constant <cuda.lang.Constant>`: the argument is treated as a
  kernel constant. Each distinct value selects a specialization, even if the
  value is runtime data in the host entry.
* **Unannotated parameter, constant argument in the host entry:** the argument
  remains constant in the kernel and specializes it by value. This includes
  literals, constant expressions, and the host entry's :py:obj:`cl.Constant <cuda.lang.Constant>` parameters.
* **Unannotated parameter, runtime argument in the host entry:** the argument
  stays a runtime value. Calls with different values reuse the same kernel
  specialization when its type and other specialization properties match.

At an ordinary Python launch boundary, an unannotated scalar becomes a runtime
argument, including a literal such as ``1``. Inside :py:obj:`@cl.host_entry <cuda.lang.host_entry>`, the same
literal is known during compilation and remains constant when passed to an
unannotated kernel parameter.

This example passes one argument of each kind:

.. testcode:: language-jit-compilation
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def affine(values, out, scale: cl.Constant[int], offset, bias):
       cl.static_eval(print("Compiling affine with scale", scale))
       fixed_offset = cl.ensure_constant(offset)
       out[0] = values[0] * scale + fixed_offset + bias

   @cl.host_entry
   def run_affine(stream, values, out, scale, bias):
       cl.static_eval(print("Compiling run_affine"))
       cl.launch(stream, (1,), (1,), affine, (values, out, scale, 1, bias))

   values = torch.tensor([3], dtype=torch.float32, device="cuda")
   out = torch.empty_like(values)
   run_affine(stream, values, out, 2, 3)  # Compile host and kernel for scale=2.
   # end-snippet
   assert out.cpu().item() == 10
   # begin-snippet

   run_affine(stream, values, out, 2, 5)  # Change bias: reuse both versions.
   # end-snippet
   assert out.cpu().item() == 12
   # begin-snippet

   run_affine(stream, values, out, 4, 5)  # Change scale: compile another kernel.
   # end-snippet
   assert out.cpu().item() == 18
   # begin-snippet

   run_affine(stream, values, out, 2, 3)  # Reuse the cached kernel for scale=2.
   # end-snippet
   assert out.cpu().item() == 10

.. testoutput:: language-jit-compilation

   Compiling run_affine
   Compiling affine with scale 2
   Compiling affine with scale 4

The :py:obj:`static_eval(print(...)) <cuda.tile.static_eval>` calls print only during compilation. The
messages show one host compilation and two kernel compilations. The final call
reuses the specialization for ``scale=2`` and prints no compilation message.

``scale`` and ``bias`` are runtime parameters of ``run_affine``. Changing either
value reuses the host specialization, as long as the argument types and other
specialization properties match. At the kernel boundary:

* ``scale`` becomes constant because ``affine`` annotates it with :py:obj:`cl.Constant <cuda.lang.Constant>`.
  Changing it from ``2`` to ``4`` selects a different kernel specialization.
* ``offset`` is constant because the host entry passes the literal ``1``, even
  though the kernel parameter has no annotation. :py:obj:`ensure_constant <cuda.tile.ensure_constant>` checks this.
* ``bias`` remains runtime data. Changing it from ``3`` to ``5`` reuses the same
  kernel specialization.

Constant inheritance applies to unannotated kernel parameters. An explicit
supported scalar annotation such as :py:obj:`cl.ScalarInt64 <cuda.lang.ScalarInt64>` treats the argument as a
runtime value, even when it is constant in the host entry.




.. _lang-compiled-host-functions:

Compiled Host Functions
-----------------------

Compiled host functions are currently supported for ``cuda.lang`` only.
Compiled host support for ``cuda.tile`` will be added in the future.

:py:obj:`cuda.lang.host_entry <cuda.lang.host_entry>` compiles host-side functions into CPU code.
A decorated function is callable from Python, and its body can launch SIMT kernels with
:py:obj:`cl.launch <cuda.lang.launch>`.

Host-function specializations are separate from those of the GPU kernels they
launch. See :ref:`language-jit-compilation` for the specialization rules and a
complete example, including how constants cross a compiled-host launch boundary.

.. autoclass:: cuda.lang.host_entry

.. _language-supported-python-features:

Supported Python Features
-------------------------

The following value types and language features are available in default mode.
:ref:`language-entry-arguments` describes which values can be supplied from
Python when entering compiled code.

Supported Value Types
~~~~~~~~~~~~~~~~~~~~~

* Boolean, integer, and floating-point scalars.
* Python ``enum.Enum`` members.
* String constants and formatted strings for supported uses such as printing,
  and ``None`` where an API accepts it.
* Tuples whose elements are supported values.
* Frozen dataclass instances whose fields are supported values. Their classes
  are defined outside the compiled function with ``@dataclass(frozen=True)``.
* Function and closure values, as described under :ref:`language-helper-functions`.
* GPU arrays and CUDA streams.
* |cuda.lang| only: typed pointers, vectors, and tensor-map descriptors.
* |cuda.tile| only: tiles.
* |cuda.tile| also accepts non-empty Python lists of arrays as kernel arguments.
  These become list views with same-rank, same-dtype array elements, supporting
  ``len`` and integer indexing over an immutable list structure.

Creating or mutating ordinary Python lists, dictionaries, sets, and iterators
is unsupported in default mode. Supported immutable data structures include
tuples and frozen dataclasses. Arrays support loads and stores to GPU memory.

Dictionary support is limited to read-only dictionaries formed by collecting
keyword arguments into a helper's ``**kwargs`` parameter. These support lookup
by compile-time string keys, ``get``, membership tests, and forwarding with
``helper(**kwargs)``. Their values may be runtime values.

Supported Language Syntax
~~~~~~~~~~~~~~~~~~~~~~~~~

* Local variables, assignment, augmented assignment such as ``x += 1``, and
  nested tuple packing and unpacking. Local variable type annotations are
  accepted but have no effect on compilation; see
  :ref:`language-parameter-annotations`.
* Arithmetic, bitwise operations, comparisons, Boolean expressions, and
  conditional expressions such as ``a if condition else b``. Integer inputs
  support ``divmod``.
* ``if``/``else``, ``while``, and ``for`` loops over ``range``. A range's step
  must be positive. Ordinary loops support ``break`` and ``continue``;
  ``for-else`` and ``while-else`` are unsupported.
* Calls to supported |cuda.lang| or |cuda.tile| APIs, helper functions,
  closures, and supported dataclass methods.
* Nested function and lambda definitions. Helpers and closures can return
  values; kernels can only use ``return`` without a value.
* Variadic helper parameters (``*args`` and ``**kwargs``) and argument
  unpacking (``helper(*args, **kwargs)``). A helper's ``**kwargs`` parameter is
  a read-only dictionary.
* Tuple concatenation with ``+``, ``len(tuple)``, and membership tests with
  ``in`` and ``not in``. Tuple indexing requires a compile-time constant index.
  Tuple comprehensions use :py:obj:`cl.static_iter <cuda.tile.static_iter>` for every iterable, for example
  ``tuple(i * 2 for i in cl.static_iter(range(3)))``; any filter condition must
  also be known at compile time.
* Frozen dataclass construction, field access, and ``dataclasses.replace()``
  to create a copy with changed fields.
* Enum equality and inequality comparisons, and enum construction from a
  compile-time constant value.
* Attribute access and indexing supported by the value's type, such as
  ``array.dtype`` and ``array[i]``.
* ``with`` statements for supported context managers, such as
  ``with cl.local_array(...) as scratch:``, including user-defined context
  managers created with ``@contextlib.contextmanager``.
* Runtime ``print``, including tuples, frozen dataclasses, and supported
  f-string formatting.

Ordinary ``raise`` is unsupported because GPU code cannot raise Python
exceptions. ``raise cl.static_exception(...)`` raises an exception during
compilation, as shown under :ref:`language-static-evaluation`.

Imports, class definitions, and ``try``/``except`` are also unsupported in a
compiled function body. Import modules and define classes outside the function,
and use static evaluation for Python computations needed during compilation.

.. _language-metaprogramming:

Compile-Time Programming
------------------------

The following sections cover compile-time Python computations, symbolic
expressions, and helper-function restrictions in more detail.

.. _language-static-evaluation:

Static Evaluation
~~~~~~~~~~~~~~~~~

Static evaluation runs ordinary Python during compilation to compute
configuration values and build or select symbolic expressions.

.. list-table:: Mode Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Default mode
     - Static evaluation
   * - Supported syntax and mutable Python objects
     - Python subset; creating or mutating ordinary Python lists and
       dictionaries is unsupported.
     - Full Python; mutable Python objects can be created and used during
       compilation.
   * - ``print(...)``
     - Prints when the compiled code runs.
     - Python prints during compilation.
   * - ``if``, ``for``, ``while``
     - Compiles to runtime control flow; constant conditions may simplify it.
     - Python chooses branches and runs loops during compilation.
   * - Helper function calls
     - Inlines the helper with default-mode semantics.
     - Executes the helper as ordinary Python during compilation.
   * - Device operations: loads, stores, :py:obj:`cl.syncthreads() <cuda.lang.syncthreads>`
     - Allowed in kernels; execute on the GPU at runtime.
     - Cannot issue device loads, stores, or synchronization.

The ``static_*`` constructs provide explicit ways to request compile-time work:

* :py:obj:`cl.static_eval(expression) <cuda.tile.static_eval>` evaluates one Python expression during
  compilation and returns its result to default mode.
* ``for item in cl.static_iter(iterable):`` evaluates the iterable and unrolls
  the loop during compilation. Its body retains default-mode runtime semantics.
  Currently, ``break``, ``continue``, and ``return`` are unsupported in this
  unrolled loop.
* ``raise cl.static_exception(exception)`` constructs and raises an exception
  during compilation.

:ref:`@cl.static_def <language-static-def>` gives a helper compile-time
semantics at every call.

Evaluate Python Expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:obj:`static_eval <cuda.tile.static_eval>` takes a single expression. A Python
helper called from that expression can contain multiple statements.

For example, Python's ``sorted()`` function and lists are unsupported in default
mode. :py:obj:`static_eval <cuda.tile.static_eval>` evaluates the sort during compilation and returns the value
needed by the kernel. This helper also uses an ordinary Python ``if``,
``print``, and ``raise``:

.. testcode:: language-static-eval
   :template: language_example.py

   # begin-snippet
   def choose_smallest(values):
       if not values:
           raise ValueError("values must not be empty")
       print("Sorting", values)
       return sorted(values)[0]

   @cl.kernel
   def write_smallest(out):
       smallest = cl.static_eval(choose_smallest((3, 1, 2)))
       out[0] = smallest  # The store runs on the GPU; smallest is constant 1.
   # end-snippet

   out = torch.empty(1, dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (1,), write_smallest, (out,))
   assert out.cpu().item() == 1

.. testoutput:: language-static-eval

   Sorting (3, 1, 2)

During compilation, Python prints ``Sorting (3, 1, 2)`` and computes ``1``.
Passing an empty tuple would raise ``ValueError`` during compilation. Without
:py:obj:`static_eval <cuda.tile.static_eval>`, the helper's sorting and ordinary exception raising are
unsupported. The temporary Python list exists only during compilation.

Unroll Loops
^^^^^^^^^^^^

:py:obj:`static_iter <cuda.tile.static_iter>` generates a sequence of runtime operations:

.. testcode:: language-static-iter
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def write_values(out):
       for i, value in cl.static_iter(enumerate([10, 20])):
           out[i] = value
   # end-snippet

   out = torch.empty(2, dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (1,), write_values, (out,))
   assert out.cpu().tolist() == [10, 20]

This generates two stores, equivalent to ``out[0] = 10`` and ``out[1] = 20``.
The iterable is evaluated and the loop variables are bound during compilation.
The stores execute on the GPU. Arguments supplied from Python can be
made available during compilation through
:ref:`language-parameter-annotations`.

Partial unrolling combines a runtime outer loop with a fixed-size inner
expansion:

.. testcode:: language-partial-unroll
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def sum_rows(values, out):
       row = cl.thread_index(0)
       total = cl.int32(0)
       for base in range(0, values.shape[1], 4):
           for offset in cl.static_iter(range(4)):
               column = base + offset
               if column < values.shape[1]:
                   total += values[row, column]
       out[row] = total

   values = torch.arange(44, dtype=torch.int32, device="cuda").reshape(4, 11)
   out = torch.empty(4, dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (4,), sum_rows, (values, out))
   torch.testing.assert_close(out, values.sum(dim=1, dtype=torch.int32))
   # end-snippet

Each thread sums one row. The outer loop advances through the runtime row
length in groups of four. During compilation, ``static_iter(range(4))`` expands
the inner body with ``offset`` equal to ``0``, ``1``, ``2``, and ``3``.
The bounds checks, loads, and additions execute on the GPU; the checks handle
a final group with fewer than four elements.

You can also use other Python iterator factories inside :py:obj:`static_iter <cuda.tile.static_iter>`:

.. testcode:: language-static-enumerate
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def copy_tensors(tensors, outputs):
       index = cl.thread_index(0)
       for t, o in cl.static_iter(zip(tensors, outputs, strict=True)):
           o[index] = t[index]
   # end-snippet

   tensors = (
       torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda"),
       torch.tensor([5, 6, 7, 8], dtype=torch.float32, device="cuda"),
   )
   outputs = tuple(torch.empty_like(t) for t in tensors)
   cl.launch(stream, (1,), (4,), copy_tensors, (tensors, outputs))
   for actual, expected in zip(outputs, tensors):
       torch.testing.assert_close(actual, expected)

``tensors`` and ``outputs`` are tuples of arrays of the same length.
Their tuple structure is known during compilation, while the array contents
are runtime values. The compiler unrolls one load and store for each input/output
array pair. The loop body still has runtime semantics: GPU threads copy the
array elements when the kernel runs.

Raise Compile-Time Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A compile-time check can stop compilation with a useful message:

.. testcode:: language-static-exception
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def checked_kernel(n: cl.Constant[int]):
       if n <= 0:
           raise cl.static_exception(ValueError("n must be positive"))
   # end-snippet

   from unittest import TestCase
   from cuda.tile import StaticException

   cl.launch(stream, (1,), (1,), checked_kernel, (1,))
   with TestCase().assertRaisesRegex(StaticException, "n must be positive"):
       cl.launch(stream, (1,), (1,), checked_kernel, (0,))

Working with Runtime Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a default-mode value enters static evaluation, a constant becomes its
corresponding Python value. A runtime value becomes a **symbolic value**: a
Python object representing a runtime value whose contents are not yet known.
You can inspect available type metadata and build supported symbolic expressions.
Returning a symbolic value brings the expression back into default mode.

.. testcode:: language-symbolic-expressions
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def adjust(values, out, operation: cl.Constant[str]):
       x = values[0]
       y = cl.static_eval({"increment": x + 1, "double": x * 2}[operation])
       out[0] = y
   # end-snippet

   values = torch.tensor([3], dtype=torch.float32, device="cuda")
   out = torch.empty_like(values)
   for operation, expected in (("increment", 4), ("double", 6)):
       cl.launch(stream, (1,), (1,), adjust, (values, out, operation))
       assert out.cpu().item() == expected

Here, Python builds a dictionary of symbolic expressions and selects one using
the constant ``operation`` string. ``"increment"`` selects ``x + 1``;
``"double"`` selects ``x * 2``. The dictionary exists only during compilation,
and the selected arithmetic runs on the GPU after the load. This dictionary
literal needs :py:obj:`static_eval <cuda.tile.static_eval>` because dictionary literals are unsupported in
default mode. Limited read-only dictionaries are available through helper
``**kwargs`` parameters, as described under
:ref:`language-supported-python-features`.

Static evaluation builds symbolic expressions for default mode to compile
into runtime code. Device operations such as :py:obj:`cl.syncthreads() <cuda.lang.syncthreads>`,
memory loads and stores, and control flow depending on runtime values must
be expressed in default mode.

Helper Functions across Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ordinary Python helper inherits the mode of its caller. No extra decorator
or annotation is required:

.. testcode:: language-helper-modes
   :template: language_example.py

   # begin-snippet
   def report_value(x):
       print("value:", x)

   @cl.kernel
   def use_helper(values):
       x = values[0]
       report_value(x)                  # GPU prints the value at runtime.
       cl.static_eval(report_value(x))  # Python prints a symbol at compile time.
   # end-snippet

   values = torch.tensor([3], dtype=torch.float32, device="cuda")
   cl.launch(stream, (1,), (1,), use_helper, (values,))

.. testoutput:: language-helper-modes

   value: <scalar[float32]>
   value: 3.000000

The regular call compiles ``report_value`` with default-mode runtime semantics. When
the kernel executes, each GPU thread prints the value loaded from ``values[0]``.
The :py:obj:`static_eval <cuda.tile.static_eval>` call runs the same helper as Python during compilation.
It prints the symbolic representation of ``x``: for a ``float32`` input array,
the message is ``value: <scalar[float32]>``. The actual value becomes available
only at runtime.

A helper called in default mode must use the supported Python subset;
a helper defined outside the compiled function and called in static evaluation
can use full Python, subject to the compile-time restrictions above. Helpers
defined inside default-mode code become
:ref:`symbolic closures <language-symbolic-closures>` when passed into static
evaluation.

.. _language-recursion:

Inlining and Recursion
~~~~~~~~~~~~~~~~~~~~~~

Helper calls in default mode are **inlined during compilation**: the compiler
inserts the helper's body at each call site. Operations in that body still have
runtime semantics.

If a recursive call's stopping condition depends on a runtime value, the
compiler cannot determine when to stop inlining. Compilation reaches the
recursion limit and raises ``RecursionLimitError``:

**Does not compile.**

.. testcode:: language-recursion
   :template: language_example.py

   # begin-snippet
   def advance(state, depth):
       if depth > 0:
           return advance(state + 1, depth - 1)
       return state

   @cl.kernel
   def example(n):
       result = advance(0, n)

   # cl.launch(..., example, (5,)) raises RecursionLimitError
   # end-snippet

   from unittest import TestCase
   from cuda.lang._exception import RecursionLimitError

   with TestCase().assertRaises(RecursionLimitError):
       cl.launch(stream, (1,), (1,), example, (3,))

   @cl.kernel
   def constant_depth(out):
       out[0] = advance(0, 3)

   out = torch.empty(1, dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (1,), constant_depth, (out,))
   assert out.cpu().item() == 3

Here, ``n`` is a runtime value, so the compiler keeps expanding the recursive
branch even if ``n`` would be small when the kernel runs. A ``for`` or ``while``
loop supports a repetition count that depends on a runtime value.

Recursion with a compile-time constant depth, such as ``advance(0, 3)``, can
compile: constant folding selects the branch at each step and eventually
reaches the stopping condition. The depth must still fit within the compiler's
recursion limit. Calls made in static evaluation follow ordinary Python call
semantics and are subject to Python's own recursion limit.

.. _language-static-def:

Compile-Time Helpers with static_def
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cuda.lang.static_def

:py:obj:`@cl.static_def <cuda.lang.static_def>` is syntactic sugar for giving an entire helper function
static-evaluation semantics. Calling it from default mode implicitly enters
static evaluation:

.. testcode:: language-static-def
   :template: language_example.py

   # begin-snippet
   @cl.static_def
   def smallest(values):
       return sorted(values)[0]

   @cl.kernel
   def write_smallest_with_helper(out):
       out[0] = smallest((3, 1, 2))
   # end-snippet

   out = torch.empty(1, dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (1,), write_smallest_with_helper, (out,))
   assert out.cpu().item() == 1

The helper sorts using Python during compilation and returns the constant
``1``.

.. _language-symbolic-closures:

Symbolic Closures
~~~~~~~~~~~~~~~~~

Nested functions or lambdas defined inside default-mode code cross into static
evaluation as **symbolic closures**. These are opaque function handles: Python
can pass them around, store them in containers, select one, and return it to
default mode.
Calling a symbolic closure inside :py:obj:`static_eval <cuda.tile.static_eval>` or a :py:obj:`@cl.static_def <cuda.lang.static_def>` helper
is **disallowed by design**: it would be ambiguous whether its body should have
runtime or compile-time semantics.

Static evaluation can operate on a symbolic closure without changing the
semantics of its body. For example, a Python dictionary can select a callback
by name. One of these callbacks captures a runtime value from the enclosing
kernel:

.. testcode:: language-closure-selection
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def choose_callback(values, out, bias, operation: cl.Constant[str]):
       def double(x):
           return x * 2

       def add_bias(x):
           return x + bias

       fn = cl.static_eval({"double": double, "add_bias": add_bias}[operation])
       out[0] = fn(values[0])
   # end-snippet

   values = torch.tensor([3], dtype=torch.float32, device="cuda")
   out = torch.empty_like(values)
   for operation, bias, expected in (
       ("double", 2.0, 6),
       ("add_bias", 2.0, 5),
       ("add_bias", -1.0, 2),
   ):
       cl.launch(stream, (1,), (1,), choose_callback, (values, out, bias, operation))
       assert out.cpu().item() == expected

``add_bias`` captures the runtime parameter ``bias``. Static evaluation selects
the function using the constant ``operation``. The selected closure is returned
to default mode and called there; its
arithmetic executes on the GPU. Calling it inside static evaluation is not
allowed:

**Does not compile.**

.. testcode:: language-symbolic-closure-error
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def unsupported_closure_call(out):
       def twice(x):
           return x * 2

       fn = cl.static_eval(twice)      # Allowed: returns the symbolic closure.
       out[0] = cl.static_eval(fn(3))  # Compilation error: cannot call it here.
   # end-snippet

   from unittest import TestCase
   from cuda.lang._exception import StaticEvalError

   out = torch.empty(1, dtype=torch.int32, device="cuda")
   with TestCase().assertRaisesRegex(StaticEvalError, "cannot be called inside static_eval"):
       cl.launch(stream, (1,), (1,), unsupported_closure_call, (out,))

   @cl.kernel
   def call_closure_in_default_mode(out):
       def twice(x):
           return x * 2

       fn = cl.static_eval(twice)
       out[0] = fn(3)

   cl.launch(stream, (1,), (1,), call_closure_in_default_mode, (out,))
   assert out.cpu().item() == 6

Calling the closure in default mode, for example ``out[0] = fn(3)``, compiles
its body with default-mode semantics. A helper defined outside the compiled
function enters static evaluation as an ordinary Python callable and executes
with Python semantics during compilation.


Known Implementation Gaps
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _language-python-lambdas:

Captured Python Lambdas
^^^^^^^^^^^^^^^^^^^^^^^

A Python lambda defined outside default mode cannot currently be captured and
called as a default-mode helper. The compiler cannot reliably recover its
abstract syntax tree (AST) from the Python function object. The same limitation
applies to lambdas returned from :py:obj:`static_eval <cuda.tile.static_eval>` or a :py:obj:`@cl.static_def <cuda.lang.static_def>` helper.
A returned lambda can cross as a function value, but calling it in default mode
currently fails compilation:

**Does not compile.**

.. testcode:: language-python-lambda-error
   :template: language_example.py

   # begin-snippet
   double = lambda x: x * 2

   @cl.kernel
   def unsupported_captured_lambda(values, out):
       out[0] = double(values[0])  # Compilation error: cannot recover lambda AST.

   @cl.kernel
   def unsupported_returned_lambda(values, out):
       fn = cl.static_eval(lambda x: x * 2)
       out[0] = fn(values[0])  # Compilation error: cannot compile this lambda.
   # end-snippet

   from unittest import TestCase
   from cuda.lang._exception import InternalError

   values = torch.tensor([3], dtype=torch.float32, device="cuda")
   out = torch.empty_like(values)
   for kernel in (unsupported_captured_lambda, unsupported_returned_lambda):
       with TestCase().assertRaises(InternalError):
           cl.launch(stream, (1,), (1,), kernel, (values, out))

   @cl.kernel
   def local_lambda(values, out):
       fn = lambda x: x * 2
       out[0] = fn(values[0])

   cl.launch(stream, (1,), (1,), local_lambda, (values, out))
   assert out.cpu().item() == 6

Captured Python functions defined with ``def`` can be compiled as default-mode
helpers. Lambdas defined directly inside default-mode code are supported because
their AST is already
available to the compiler. Lambdas created and called entirely within static
evaluation follow ordinary Python semantics.
