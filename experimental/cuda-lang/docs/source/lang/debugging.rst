.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang

Debugging and Validation
========================

``cuda.lang`` provides compiler logs and source line information for debugging
SIMT kernels. Reference comparisons check numerical results; Compute Sanitizer
checks memory accesses and synchronization for the exercised inputs and launch
dimensions.

Compilation and PTX Inspection
------------------------------

``CUDA_LANG_LOGS`` selects compiler output written to stderr. To capture PTX,
run the :doc:`SIMT quickstart </getting-started/lang>`, saved as
``vector_add_lang.py``, with ``CUDA_LANG_LOGS=PTX``:

.. code-block:: bash

   CUDA_LANG_LOGS=PTX python vector_add_lang.py 2>compile.log

Logs are emitted when a specialization is compiled. In a fresh Python process,
the first launch triggers compilation; later launches can reuse the cached
specialization. ``compile.log`` contains the PTX dump and its header.

Combine logging keys with commas:

.. code-block:: bash

   CUDA_LANG_LOGS=HIR,IR,PTX,TIMINGS python vector_add_lang.py 2>compile.log

.. list-table:: Compiler logging keys
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Output
   * - ``HIR``
     - High-level IR derived from Python.
   * - ``IR``
     - IR before and after transformation passes; also enables IR diagnostics
       on errors.
   * - ``FLATIR``
     - IR after control-flow flattening.
   * - ``MLIR``, ``NVVM``
     - Intermediate compiler representations before PTX generation.
   * - ``PTX``
     - Generated PTX.
   * - ``TIMINGS``
     - Compilation phase timings.

The PTX identifies the target, entry point, and generated instructions, whose
semantics are defined in the
`PTX ISA reference <https://docs.nvidia.com/cuda/parallel-thread-execution/>`_.
Optimization can simplify or eliminate operations, so the generated code
depends on the specialization.

.. _lang-specialization-visibility:

Observing JIT Specializations
-----------------------------

A ``cl.static_eval(print(...))`` call reports values during compilation.
The :ref:`constant-parameter example <language-parameter-annotations>` prints
each compiled ``offset`` value and shows reuse of an existing specialization.
``CUDA_LANG_LOGS=PTX,TIMINGS`` provides generated code and timings for each
compilation. Cached launches produce neither a new compile-time print nor a
new compilation log.

Both ``cuda.lang`` and ``cuda.tile`` kernel specializations are cached in the
Python process. ``cuda.tile`` also has an on-disk bytecode-to-cubin cache that
can be inspected with the ``cutile-cache log`` command-line interface.
See :doc:`/tile/debugging`.

Correctness Checks
------------------

The quickstart compares kernel output with a NumPy reference. To extend it to
boundary sizes and strided inputs, keep its imports and ``vector_add``
definition and replace the input-and-launch section with:

.. code-block:: python

   stream = cp.cuda.get_current_stream()
   threads_per_block = 128
   rng = cp.random.default_rng()
   for n in (0, 1, 31, 32, 33, 127, 128, 129, 1000):
       for stride in (1, 2):
           a_storage = rng.random(n * stride, dtype=cp.float32)
           a = a_storage[::stride]
           b = rng.random(n, dtype=cp.float32)
           out = cp.zeros_like(a)
           if n:
               blocks = cl.cdiv(n, threads_per_block)
               cl.launch(stream, (blocks,), (threads_per_block,),
                         vector_add, (a, b, out))
           stream.synchronize()
           expected = cp.asnumpy(a_storage)[::stride] + cp.asnumpy(b)
           np.testing.assert_allclose(cp.asnumpy(out), expected,
                                      rtol=1e-6, atol=1e-6)
   print("Boundary and stride checks passed.")

Synchronization after a launch associates asynchronous CUDA errors with that
launch. Test coverage depends on the exercised dtypes, layouts, block sizes,
and data values. For reductions and MMA, accumulation order and precision
affect the tolerance of a reference comparison. Repeated runs can expose
intermittent failures.

Compute Sanitizer
-----------------

Compute Sanitizer is included in the CUDA Toolkit. Add the toolkit's ``bin``
directory to ``PATH`` to run ``compute-sanitizer``. For source line information
in reports, change the quickstart kernel's decorator to
``@cl.kernel(debug_info="line")``.

Run the relevant tool with one of the following commands:

.. code-block:: bash

   compute-sanitizer --tool memcheck --error-exitcode 1 python vector_add_lang.py
   compute-sanitizer --tool racecheck --error-exitcode 1 python vector_add_lang.py
   compute-sanitizer --tool initcheck --error-exitcode 1 python vector_add_lang.py
   compute-sanitizer --tool synccheck --error-exitcode 1 python vector_add_lang.py

``memcheck`` detects invalid and misaligned memory accesses. ``racecheck`` checks
shared-memory access hazards; ``initcheck`` checks uninitialized global-memory
reads. ``synccheck`` checks supported synchronization primitives for invalid
participation. ``--error-exitcode 1`` makes tool-detected errors fail an otherwise
successful application run. See the `Compute Sanitizer manual
<https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>`_
for each tool's coverage and version requirements.

The :doc:`synchronization protocols <sync>` and individual API contracts
describe participation and completion requirements, including those of TMA
and Gen5 pipelines.

Error Diagnostics
-----------------

* **Compilation errors** include argument-type mismatches, unmet ``const``
  requirements, unsupported execution spaces, and unavailable target features.
  Diagnostics identify Python source locations; ``HIR`` and ``IR`` logs expose
  the intermediate expressions. ``CompilerExecutionError`` includes the backend
  diagnostic and compiler flags.
* **Illegal accesses** can result from out-of-bounds indices, pointer arithmetic,
  strides, vector alignment, or shared-memory allocation sizes. ``memcheck``
  reports invalid accesses. Because launches are asynchronous, an error can
  surface at a later CUDA call; synchronization after a launch narrows that
  interval. A device fault can leave the CUDA context unusable; a new process
  creates a fresh context.
* **Wrong results** can reflect work assignment, initialization, aliasing,
  floating-point semantics, or data races. Reference comparisons identify
  differing elements. ``racecheck`` detects shared-memory access hazards, and
  ``initcheck`` detects uninitialized global-memory reads.
* **Synchronization hangs** can result from mismatched barrier participation,
  IDs, counts, or warp masks. Mbarrier waits also depend on arrivals, transaction
  bytes, and phase parity. Circular dependencies can prevent a producer from
  reaching the signal that releases a consumer. :doc:`tma` and
  :doc:`warp-specialization` describe these dependencies and buffer reuse.

On Linux, ``timeout`` limits the duration of a process, including compilation
and sanitizer overhead. For example:

.. code-block:: bash

   timeout --kill-after=5s 60s python vector_add_lang.py

The compiler timeout documented in :doc:`/tile/debugging` controls ``tileiras``;
an external process timeout also applies to SIMT compiler and application runs.

Validation Scope
----------------

Compilation, hardware execution, and sanitizer checks establish different
properties of an example:

* **Compile-tested** means code generation succeeded for a particular target
  and argument signature with a particular compiler version.
* **Validated on hardware** means execution and reference comparisons passed
  for the exercised GPU, driver, launch dimensions, and input shapes/dtypes/strides.
* **Checked with Compute Sanitizer** means the selected tools examined those
  executions. Coverage depends on the tool version and exercised cases.

For example, compiling a Gen5 kernel for ``sm_100a`` establishes compilation
support for that target. Hardware validation also requires execution on a
compatible GPU. A skipped hardware test leaves execution unvalidated. An HTML
build checks documentation rendering; running examples or their tests on
compatible hardware establishes execution results.
