.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

.. _language:

Language Guide
==============

CUDA languages in Python, such as |cuda.lang| and |cuda.tile|, share a
consistent language design. This guide uses |cuda.lang| examples to explain
the shared concepts.

**TL;DR**

* **Two explicit modes** separate using Python for runtime and compile-time
  programming: default mode and static evaluation.
* **Default mode** compiles a *Python subset* for runtime execution. The
  ``static_*`` constructs explicitly request compile-time work.
  Constant folding propagates constantness through scalar expressions.
* **Static evaluation** runs *full Python* for compile-time metaprogramming.
  It supports symbolic computation but cannot directly emit runtime
  instructions or control flow that depends on symbolic runtime values.
* **Values can cross modes.** Constants from default mode enter static evaluation
  as ordinary Python values; runtime values enter as symbolic values. Constants
  and symbolic expressions can return to default mode if they can be converted
  to values representable in default mode.

.. list-table:: Mode comparison
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
     - Python chooses branches and runs loops using compile-time values.
   * - Ordinary helper calls
     - Inlines the helper with default-mode semantics.
     - Executes the helper as ordinary Python during compilation.
   * - Device operations: loads, stores, ``cl.syncthreads()``
     - Allowed in kernels; execute on the GPU at runtime.
     - Cannot issue device loads, stores, or synchronization.

Default Mode
------------

The body of a ``@cl.kernel`` or ``@cl.host_entry`` function starts in default
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
Python itself does not execute these operations during compilation.


.. _language-parameter-annotations:

Converting Python Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When Python launches a ``@cl.kernel`` through ``cl.launch``, or calls a
``@cl.host_entry``, arguments cross from the Python interpreter into compiled
code. Supported Python values are converted to |cuda.lang| values. For example,
a GPU tensor becomes an ``Array`` view, and a numeric argument becomes a scalar.
Only supported parameter annotations, such as ``cl.Constant``, affect this
conversion at a compilation boundary: ``@cl.kernel`` or ``@cl.host_entry``.

Use ``cl.Constant`` to mark an argument as a compile-time constant:

.. testcode:: language-constant-argument
   :template: language_example.py

   # begin-snippet
   @cl.kernel
   def add_offset(values, offset: cl.Constant[int]):
       i = cl.thread_index(0)
       values[i] = values[i] + offset
   # end-snippet

   values = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (4,), add_offset, (values, 3))
   assert values.cpu().tolist() == [4, 5, 6, 7]
   cl.launch(stream, (1,), (4,), add_offset, (values, 5))
   assert values.cpu().tolist() == [9, 10, 11, 12]

Here, ``offset`` is known during compilation, while the array elements are
runtime values. The compiler creates a separate specialization for each distinct
``offset`` value. ``Constant`` can also be used on ``@cl.host_entry`` parameters.
Other supported annotations are shared with |cuTile|:

* `cl.ScalarInt64
  <https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.ScalarInt64>`_
  makes an integer parameter use ``int64``.
* `cl.ArrayAnnotation
  <https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.ArrayAnnotation>`_
  is used with ``typing.Annotated`` to control array indexing types and mark
  selected shape or stride dimensions as compile-time constants.

These links use the ``cuda.tile`` names; the same annotations are available
through ``cuda.lang``.

.. note::

   Ordinary scalar type hints do not enforce or convert types. For example,
   ``x: int``, ``x: float``, or ``x: cl.int32`` does not enforce or convert
   ``x`` to that type. Use an explicit conversion such as ``cl.int32(x)``
   when you need a particular scalar type.

   In ``cl.Constant[int]``, ``cl.Constant`` marks the argument as constant.
   The inner ``int`` is a regular Python type hint; it does not validate or
   convert the argument's type.

   Tuple annotations describe the argument structure and can apply supported
   metadata to individual elements, for example
   ``tuple[cl.Array, cl.Constant[int]]``. Ordinary element type hints still do
   not enforce or convert element types.

   Annotations on helper parameters, local variables, and return values do not
   affect compilation. In particular, ``cl.Constant`` only marks an argument
   constant when used on a parameter at a compilation boundary.



.. _language-captured-values:

Capturing Python Values
~~~~~~~~~~~~~~~~~~~~~~~

A ``@cl.kernel`` or ``@cl.host_entry`` function can reference values from module
globals or an enclosing Python function. Values that the compiler can represent
as constants in default mode are captured when the function is compiled. They
do not need a ``cl.Constant`` annotation, because they are captured values
rather than function arguments.

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
``factor``. When launched through ``cl.launch``, ``double_values`` compiles with
the constant ``2``, and ``triple_values`` compiles with the constant ``3``.
The array loads, multiplication, and stores still have runtime semantics.
The same capture rules apply when a ``@cl.host_entry`` is compiled.

Capture requires both a value known during compilation and a representation
as a constant in default mode. Examples include Python numbers, tuples of such
values, and frozen dataclasses whose fields meet the same requirements.
A GPU tensor can be passed as a runtime ``Array`` argument, but cannot be
captured as a constant. Captured values are not read again at runtime, so
changing an outer Python variable does not update already compiled code.

.. _language-constants:

Constants and Constant Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A constant is a value known during compilation. In default mode:

* A ``cl.Constant`` parameter annotation on ``@cl.kernel`` or ``@cl.host_entry``
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

Use ``cl.ensure_constant(value)`` to assert that a value is constant and return
its compile-time constant value. Compilation fails for runtime values;
``ensure_constant`` does not make them constant.

An ``if`` whose condition is constant selects a branch during compilation and
discards the other branch. You can use ``ensure_constant`` to enforce such
behavior:

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

Constantness and numeric typing are separate properties. The rules for loosely
typed and strictly typed constants can be found in cuTile's documentation:
`Constant Expressions & Objects
<https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-constant-expressions-objects>`_
for the definitions, and
`Arithmetic Promotion
<https://docs.nvidia.com/cuda/cutile-python/data.html#data-arithmetic-promotion>`_
for how operations combine operand types.

For the current limitation when passing sized integer constants into static
evaluation, see :ref:`language-sized-integer-constants`.


.. _language-static-evaluation:

Static Evaluation
-----------------

Static evaluation runs ordinary Python during compilation to compute
configuration values and build or select symbolic expressions. It supports
full Python with compile-time semantics. Control flow, ``print``, and ``raise``
all follow ordinary
Python semantics: Python executes branches and loops, prints messages, and
raises exceptions during compilation. A plain ``raise`` inside static evaluation
does not need ``static_exception``. Python lists, dictionaries, sets, and
iterators can also be created and used here.

The ``static_*`` constructs provide explicit ways to request compile-time work:

* ``cl.static_eval(expression)`` evaluates one Python expression during
  compilation and returns its result to default mode.
* ``for item in cl.static_iter(iterable):`` evaluates the iterable and unrolls
  the loop during compilation. Its body retains default-mode runtime semantics.
  Currently, ``break``, ``continue``, and ``return`` are unsupported in this
  unrolled loop.
* ``raise cl.static_exception(exception)`` constructs and raises an exception
  during compilation.

For a helper that should always use compile-time semantics, use
:ref:`@cl.static_def <language-static-def>`.

Evaluate Python Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``static_eval`` takes a single expression. For computations requiring multiple
statements, call a Python helper function from that expression.

For example, Python's ``sorted()`` function and lists are unsupported in default
mode. Use ``static_eval`` to sort during compilation and return just the value
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
``static_eval``, the helper's sorting and ordinary exception raising are
unsupported. The temporary Python list exists only during compilation.

Unroll Loops
~~~~~~~~~~~~

Use ``static_iter`` to generate a short sequence of runtime operations:

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
The stores execute on the GPU.

You can also use other Python iterator factories inside ``static_iter``:

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

Pass ``tensors`` and ``outputs`` as tuples of arrays of the same length.
Their tuple structure is known during compilation, while the array contents
are runtime values. The compiler unrolls one load and store for each input/output
array pair. The loop body still has runtime semantics: GPU threads copy the
array elements when the kernel runs.

Raise Compile-Time Exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
literal needs ``static_eval`` because dictionary literals are unsupported in
default mode. Limited read-only dictionaries are available through helper
``**kwargs`` parameters, as described under
:ref:`language-supported-python-features`.

Static evaluation does not emit runtime code (compiler IR). It can
build symbolic expressions for default mode to compile later, but it cannot
insert arbitrary runtime operations such as ``cl.syncthreads()``,
memory loads or stores, or control flow that depends on runtime values.

.. _language-helper-functions:

Helper Functions
----------------
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
The ``static_eval`` call runs the same helper as Python during compilation.
It prints the symbolic representation of ``x``: for a ``float32`` input array,
the message is ``value: <scalar[float32]>``. The actual value becomes available
only at runtime.

A helper called in default mode must use the supported Python subset;
a helper defined outside the compiled function and called in static evaluation
can use full Python, subject to the compile-time restrictions above. Helpers
defined inside default-mode code become
:ref:`symbolic closures <language-symbolic-closures>` when passed into static
evaluation.

Use ``@cl.function`` to explicitly mark a helper's execution spaces. Its
defaults are ``simt=True, host=False, tile=False``: the helper can be called from
SIMT code, while direct Python calls and tile compilation are disabled.
Set ``host=True`` to permit direct Python calls, or ``simt=False`` for a helper
that is only compiled for the host. Set ``tile=True`` to also permit calls from
``cuda.tile`` code. Use ``@cl.function(tile=True, simt=True)`` for a helper shared
by both device compilers.

Inlining and Recursion
~~~~~~~~~~~~~~~~~~~~~~

Helper calls in default mode are **inlined during compilation**: the compiler
inserts the helper's body at each call site. Operations in that body still have
runtime semantics. Inlining does not execute the helper as ordinary Python or
create runtime recursive calls.

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
   from cuda.tile._exception import RecursionLimitError

   with TestCase().assertRaises(RecursionLimitError):
       cl.launch(stream, (1,), (1,), example, (3,))

   @cl.kernel
   def constant_depth(out):
       out[0] = advance(0, 3)

   out = torch.empty(1, dtype=torch.int32, device="cuda")
   cl.launch(stream, (1,), (1,), constant_depth, (out,))
   assert out.cpu().item() == 3

Here, ``n`` is a runtime value, so the compiler keeps expanding the recursive
branch even if ``n`` would be small when the kernel runs. Use a ``for`` or
``while`` loop when the number of repetitions depends on a runtime value.

Recursion with a compile-time constant depth, such as ``advance(0, 3)``, can
compile: constant folding selects the branch at each step and eventually
reaches the stopping condition. The depth must still fit within the compiler's
recursion limit. Calls made in static evaluation follow ordinary Python call
semantics and are subject to Python's own recursion limit.

.. _language-static-def:

Compile-Time Helpers with static_def
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``@cl.static_def`` is syntactic sugar for giving an entire helper function
static-evaluation semantics. Calling it from default mode implicitly enters
static evaluation, so you do not need to wrap each call in ``cl.static_eval``:

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
Calling a symbolic closure inside ``static_eval`` or a ``@cl.static_def`` helper
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
the function using the constant ``operation``; it does not evaluate ``bias``.
The selected closure is returned to default mode and called there; its
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
   from cuda.tile import TileStaticEvalError

   out = torch.empty(1, dtype=torch.int32, device="cuda")
   with TestCase().assertRaisesRegex(TileStaticEvalError, "cannot be called inside static_eval"):
       cl.launch(stream, (1,), (1,), unsupported_closure_call, (out,))

   @cl.kernel
   def call_closure_in_default_mode(out):
       def twice(x):
           return x * 2

       fn = cl.static_eval(twice)
       out[0] = fn(3)

   cl.launch(stream, (1,), (1,), call_closure_in_default_mode, (out,))
   assert out.cpu().item() == 6

Call the closure in default mode instead, for example ``out[0] = fn(3)``.
If its body needs to execute as Python during compilation, define the helper
outside the compiled function so it crosses as an ordinary Python callable.


.. _language-jit-compilation:

JIT Compilation
---------------

CUDA languages use just-in-time (JIT) compilation: they compile a kernel when it
is first launched and a ``@cl.host_entry`` function when it is first called. Each
compiled version is a **specialization** for its argument types and compile-time
constants. Later calls with matching arguments reuse that version.

Changing a constant argument, such as the ``offset`` in
:ref:`language-parameter-annotations`, selects a specialization for the new
value. If that specialization has not been compiled yet, it is compiled on
demand. Returning to a previously used constant value can reuse its cached
specialization.

An array's rank (number of dimensions) and element dtype are compile-time
metadata. Changing either requires a matching specialization, even though the
array's contents remain runtime data. Supported annotations can add further
specialization properties, such as constant shape dimensions.

Separate Host and Kernel Specializations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JIT compilation of a ``@cl.host_entry`` and the kernels it launches happens
separately. Calling the host entry specializes the CPU function for its own
arguments. Each ``cl.launch`` inside it independently selects or compiles a GPU
kernel specialization for the arguments passed to that launch. One host
specialization can therefore launch several kernel specializations.

For a ``cl.launch`` inside compiled host code, the kernel parameter's annotation
and the argument's constantness determine whether its value specializes the
kernel:

* **Parameter annotated with** ``cl.Constant``: the argument is treated as a
  kernel constant. Each distinct value selects a specialization, even if the
  value is runtime data in the host entry.
* **Unannotated parameter, constant argument in the host entry:** the argument
  remains constant in the kernel and specializes it by value. This includes
  literals, constant expressions, and the host entry's ``cl.Constant`` parameters.
* **Unannotated parameter, runtime argument in the host entry:** the argument
  stays a runtime value. Changing its value does not cause a new kernel
  specialization; changes to its type or other specialization properties still
  can.

At an ordinary Python launch boundary, an unannotated scalar becomes a runtime
argument, including a literal such as ``1``. Inside ``@cl.host_entry``, the same
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

The ``static_eval(print(...))`` calls print only during compilation. The
messages show one host compilation and two kernel compilations. The final call
reuses the specialization for ``scale=2`` and prints no compilation message.

``scale`` and ``bias`` are runtime parameters of ``run_affine``. Changing either
value reuses the host specialization, as long as the argument types and other
specialization properties match. At the kernel boundary:

* ``scale`` becomes constant because ``affine`` annotates it with ``cl.Constant``.
  Changing it from ``2`` to ``4`` selects a different kernel specialization.
* ``offset`` is constant because the host entry passes the literal ``1``, even
  though the kernel parameter has no annotation. ``ensure_constant`` checks this.
* ``bias`` remains runtime data. Changing it from ``3`` to ``5`` reuses the same
  kernel specialization.

Constant inheritance applies to unannotated kernel parameters. An explicit
supported scalar annotation such as ``cl.ScalarInt64`` treats the argument as a
runtime value, even when it is constant in the host entry.




.. _language-supported-python-features:

Supported Python Features
-------------------------

The following value types and language features are available in default mode.

Supported Value Types
~~~~~~~~~~~~~~~~~~~~~

* Boolean, integer, and floating-point scalars.
* Python ``enum.Enum`` members.
* String constants and formatted strings for supported uses such as printing,
  and ``None`` where an API accepts it.
* Tuples whose elements are supported values.
* Frozen dataclass instances whose fields are supported values. Define the
  class outside the compiled function using ``@dataclass(frozen=True)``.
* Function and closure values, as described under :ref:`language-helper-functions`.
* GPU arrays and CUDA streams.
* |cuda.lang| only: typed pointers, vectors, and tensor-map descriptors.
* |cuda.tile| only: tiles.
* |cuda.tile| also accepts non-empty Python lists of arrays as kernel arguments.
  These become list views with same-rank, same-dtype array elements, supporting
  ``len`` and integer indexing. The list structure cannot be modified.

Creating or mutating ordinary Python lists, dictionaries, sets, and iterators
is unsupported in default mode. The list-of-arrays argument conversion above
does not provide general Python list semantics. Use immutable data structures
such as tuples or frozen dataclasses instead. Arrays can still load and store
GPU memory.

Dictionary support is limited to read-only dictionaries formed by collecting
keyword arguments into a helper's ``**kwargs`` parameter. These support lookup
by compile-time string keys, ``get``, membership tests, and forwarding with
``helper(**kwargs)``. Their values may be runtime values; the dictionary cannot
be mutated.

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
  Tuple comprehensions use ``cl.static_iter`` for every iterable, for example
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
exceptions. Use ``raise cl.static_exception(...)`` for a compile-time exception,
as shown under :ref:`language-static-evaluation`.

Imports, class definitions, and ``try``/``except`` are also unsupported in a
compiled function body. Import modules and define classes outside the function,
and use static evaluation for Python computations needed during compilation.


Known Implementation Gaps
-------------------------

.. _language-sized-integer-constants:

Sized Integer Constants
~~~~~~~~~~~~~~~~~~~~~~~

The sized integer type of a constant is currently not preserved when crossing
from default mode into static evaluation. For example, a constant created as
``cl.int16(5)`` is passed into ``static_eval`` as the Python integer ``5``, losing
its ``int16`` type. A returned Python integer becomes a loosely typed constant.
Arithmetic performed on the converted Python integer follows Python's
arbitrary-precision semantics.
If you need a particular dtype afterward, apply its constructor in default
mode, for example ``cl.int16(cl.static_eval(...))``. This restores the result's
dtype, but does not preserve fixed-width arithmetic inside static evaluation. See
:ref:`language-constant-types` for the distinction between the two constant types.

.. _language-python-lambdas:

Captured Python Lambdas
~~~~~~~~~~~~~~~~~~~~~~~

A Python lambda defined outside default mode cannot currently be captured and
called as a default-mode helper. The compiler cannot reliably recover its
abstract syntax tree (AST) from the Python function object. The same limitation
applies to lambdas returned from ``static_eval`` or a ``@cl.static_def`` helper.
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
   from cuda.tile._exception import InternalError

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

Use a named ``def`` helper for captured Python functions. Lambdas defined
directly inside default-mode code are supported because their AST is already
available to the compiler. Lambdas created and called entirely within static
evaluation follow ordinary Python semantics.
