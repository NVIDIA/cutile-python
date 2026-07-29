.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.tile.compilation

Compilation and Export
======================

When a kernel function marked with :py:class:`@ct.kernel <cuda.tile.kernel>` is launched
using :py:func:`ct.launch() <cuda.tile.launch>`, it is specialized and compiled just in time (JIT)
for the concrete launch arguments. It is also possible to compile a kernel ahead of time (AOT)
and export it as a CUDA binary (cubin) file, or as a TileIR bytecode file.

While just-in-time compilation provides the convenience of automatic kernel specialization,
ahead-of-time compilation requires the user to precisely describe the arguments for which the kernel
is being compiled, including their types and additional constraints (assumptions) imposed on their
values.

The main API entry point for ahead-of-time compilation
is :py:func:`cuda.tile.compilation.export_kernel`:

.. autofunction:: cuda.tile.compilation.export_kernel


Kernel Signatures
-----------------
There are two ways to construct a :py:class:`KernelSignature` object for use with
:py:func:`export_kernel`. The recommended way is to do it explicitly, by instantiating
a :py:class:`KernelSignature` object and providing a list of manually constructed
:py:class:`ParameterConstraint` objects.

Alternatively, one may use :py:meth:`KernelSignature.from_kernel_args` to obtain a signature
that would be used if the kernel was compiled just-in-time for the given example arguments.
While convenient, this approach may create undesired assumptions on kernel parameters. For example,
if the base address of an example array argument happens to be divisible by 16, an assumption may
be made that it will always be so. Launching the exported kernel with an array that doesn't
satisfy this assumption would then result in undefined behavior. It is therefore recommended to limit
the use of this approach to testing or prototyping.

.. autoclass:: cuda.tile.compilation.KernelSignature
    :members:


The :py:class:`ParameterConstraint` type alias is used as a type hint for a kernel parameter
constraint:

.. autoclass:: cuda.tile.compilation.ParameterConstraint

.. autoclass:: cuda.tile.compilation.ScalarConstraint

.. autoclass:: cuda.tile.compilation.ArrayConstraint

.. autoclass:: cuda.tile.compilation.ListConstraint

.. autoclass:: cuda.tile.compilation.TupleConstraint

.. autoclass:: cuda.tile.compilation.ConstantConstraint

.. autoclass:: cuda.tile.compilation.ConstantValue


.. _compilation-callconv:

Calling Conventions
-------------------
A calling convention defines three aspects of the binary interface provided by an exported kernel:

*   The binary format and the order of kernel arguments, e.g. as passed to the ``cuLaunchKernel()``
    CUDA Driver API function.
*   The set of supported parameter constraints.
*   The name mangling algorithm used to automatically derive a symbol name from the kernel's
    function name and a kernel signature.

Two calling conventions are available: ``cutile_python_v1`` and ``cutile_python_v2``.
Both pass binary kernel arguments in the same order as the kernel parameters are declared
in the Python kernel function, except that parameters annotated with
:py:class:`ct.Constant <cuda.tile.Constant>` are omitted.
``cutile_python_v2`` extends ``cutile_python_v1`` with support for tuple parameters
(:py:class:`TupleConstraint`) and static shapes (the ``shape_constant`` field of
:py:class:`ArrayConstraint`). New code should use ``cutile_python_v2``.

The following table lists the supported parameter constraints and their binary formats.
Constraints marked |v2only| require ``cutile_python_v2``.

.. |v2only| replace:: :sup:`v2`

.. list-table:: Parameter Constraints and their Binary Formats.
    :header-rows: 1

    * - Constraint Class
      - Binary Format of Arguments

    * - :py:class:`ScalarConstraint`
      - Passed as a single argument of the corresponding type. For example, if the constraint's
        `dtype` is :py:data:`ct.int32 <cuda.tile.int32>`, the corresponding C type is ``int32_t``;
        :py:data:`ct.float64 <cuda.tile.float64>` corresponds to C's ``double`` and so on.

    * - :py:class:`ArrayConstraint`
      - Passed as `1 + 2n` arguments, where `n` is the number of dimensions (`ndim`) of the array.
        The first argument is the device pointer to the base of the array's data. It is followed
        by `n` arguments representing the shape of the array. Finally, the last `n` arguments
        represent the strides of the array.
        The type of shape and stride arguments is determined by the ``index_dtype`` field of the
        constraint:

        The ``shape_constant`` field |v2only| requires ``cutile_python_v2``.

        .. list-table::
            :header-rows: 1

            * - Constraint's ``index_dtype``
              - C Type of Shape & Strides

            * - :py:data:`ct.int32 <cuda.tile.int32>`
              - ``int32_t``

            * - :py:data:`ct.uint32 <cuda.tile.uint32>`
              - ``uint32_t``

            * - :py:data:`ct.int64 <cuda.tile.int64>`
              - ``int64_t``

    * - :py:class:`ListConstraint` with an :py:class:`ArrayConstraint` element
      - Passed as two arguments: a device pointer to the base of the list data and an ``int32_t``
        denoting the length of the list. The base pointer must point to an 8-byte aligned
        contiguous buffer in the global GPU memory, consisting of
        `(1 + 2n) * L` 64-bit words, where `L` is the length of the list
        and `n` is the `ndim` of the element array constraint.
        Each element array of the list is represented by `(1 + 2n)` words in this buffer. The first
        word stores a device pointer to the base of array; the next `n` signed integers
        store the shape of the array; the final `n` signed integers store the strides of the array.
        Even though 64-bit integers are always used for the shape and the strides,
        they are truncated to the ``index_dtype`` of the element constraint.

    * - :py:class:`TupleConstraint` |v2only|
      - The tuple's elements are passed as consecutive arguments, as if each element were
        an individual top-level parameter. The binary format of each element follows the same
        rules as its corresponding constraint type (:py:class:`ScalarConstraint` or
        :py:class:`ArrayConstraint`).

        When the tuple parameter is annotated with
        :py:class:`ct.Constant[tuple] <cuda.tile.Constant>`, the entire tuple is omitted from
        the launch arguments (same as :py:class:`ConstantConstraint`). In this case, all
        elements of the :py:class:`TupleConstraint` must be :py:class:`ConstantConstraint`.

    * - :py:class:`ConstantConstraint`
      - Omitted from the launch arguments.


Calling conventions are represented by the :py:class:`CallingConvention` class:


.. autoclass:: cuda.tile.compilation.CallingConvention
    :members:
