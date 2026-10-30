.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

First cuda.lang Kernel
======================

After :doc:`installation`, save this example as ``vector_add_lang.py`` and run
``python vector_add_lang.py``. It adds two 1,000-element vectors and checks the
result against CuPy.

.. testcode:: quickstart-lang

   import cuda.lang as cl
   import cupy as cp

   @cl.kernel
   def vector_add(a, b, out):
       i = cl.block_index(0) * cl.thread_count(0) + cl.thread_index(0)
       if i < out.shape[0]:
           out[i] = a[i] + b[i]

   n = 1000
   a = cp.arange(n, dtype=cp.float32)
   b = cp.full_like(a, 2.0)
   out = cp.empty_like(a)
   stream = cp.cuda.get_current_stream()

   threads_per_block = 128
   blocks = cl.cdiv(n, threads_per_block)
   cl.launch(stream, (blocks,), (threads_per_block,), vector_add, (a, b, out))

   stream.synchronize()
   cp.testing.assert_allclose(out, a + b)
   print("Vector addition passed.")

.. testoutput:: quickstart-lang

   Vector addition passed.

Each thread computes an element index from its block and thread indices.
The host launches eight blocks of 128 threads. The final block contains more
threads than remaining elements, so the ``if`` guards the array accesses.

:py:obj:`cl.kernel <cuda.lang.kernel>` marks the GPU entry point. :py:obj:`cl.launch <cuda.lang.launch>` takes a stream, a grid
size in blocks, a block size in threads, the kernel, and its arguments.
Launches are asynchronous with respect to Python; synchronization here waits
for completion before validating the result.

Additional Samples
------------------

These examples use the :ref:`additional sample dependencies
<getting-started-sample-dependencies>`. The commands below use paths relative to the
root of a checkout of the `cuTile Python repository
<https://github.com/NVIDIA/cutile-python>`_.

The ``experimental/cuda-lang/test/examples/`` directory contains executable
examples written as pytest tests, including vectorized SAXPY, matrix
multiplication with shared memory, transpose, and softmax. Run the SAXPY and
shared-memory matrix multiplication tests with:

.. code-block:: bash

   python -m pytest experimental/cuda-lang/test/examples/test_vector_saxpy.py
   python -m pytest experimental/cuda-lang/test/examples/test_matmul_sharedmem.py

Advanced tensor-core kernel examples are in ``experimental/cuda-lang/tutorial/blackwell``.
For example, the FP16 GEMM tutorials demonstrate TMA transfers, asynchronous
barriers, and ``tcgen05`` Tensor Core instructions. On an SM100 GPU, such as an
NVIDIA B200, run the first tutorial with:

.. code-block:: bash

   python experimental/cuda-lang/tutorial/blackwell/fp16_gemm_0.py

The :doc:`tile` quickstart presents the same computation with tile operations.
The :doc:`/lang/index` and :doc:`/language` cover execution and language semantics.
