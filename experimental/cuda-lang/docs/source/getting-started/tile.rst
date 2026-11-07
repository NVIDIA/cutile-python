.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

First cuda.tile Kernel
======================

After :doc:`installation`, save this example as ``vector_add_tile.py`` and run
``python vector_add_tile.py``.

.. testcode:: quickstart-tile

   import cuda.tile as ct
   import cupy as cp
   import numpy as np

   TILE_SIZE = 128

   @ct.kernel
   def vector_add(a, b, out):
       block = ct.bid(0)
       x = ct.load(a, index=(block,), shape=(TILE_SIZE,),
                   padding_mode=ct.PaddingMode.ZERO)
       y = ct.load(b, index=(block,), shape=(TILE_SIZE,),
                   padding_mode=ct.PaddingMode.ZERO)
       ct.store(out, index=(block,), tile=x + y)

   n = 1000
   rng = cp.random.default_rng()
   a = rng.random(n, dtype=cp.float32)
   b = rng.random(n, dtype=cp.float32)
   out = cp.zeros_like(a)
   stream = cp.cuda.get_current_stream()

   grid = ((n + TILE_SIZE - 1) // TILE_SIZE,)
   ct.launch(stream, grid, vector_add, (a, b, out))

   stream.synchronize()
   expected = cp.asnumpy(a) + cp.asnumpy(b)
   np.testing.assert_allclose(cp.asnumpy(out), expected)
   print("Vector addition passed.")

.. testoutput:: quickstart-tile

   Vector addition passed.

Each logical block loads two 128-element tiles, adds them, and stores the
result. :py:obj:`ct.bid(0) <cuda.tile.bid>` supplies the block index used to select each tile.
The tile shape is a compile-time constant.

The grid contains eight logical tile blocks. The final tile extends beyond the
1,000-element array: loads pad out-of-bounds elements with zero, and the store
writes only in-bounds elements. No explicit thread indices or barriers are
needed.

:py:obj:`ct.launch <cuda.tile.launch>` takes a stream, a grid size in tile blocks, the kernel, and
its arguments. The compiler chooses the thread mapping within each block,
so the launch has no explicit thread-block size. As in the SIMT example,
the host waits for completion before validating the result.

Additional Samples
------------------

These examples use the :ref:`additional sample dependencies
<getting-started-sample-dependencies>`. The paths below are relative to the
root of a checkout of the `cuTile Python repository
<https://github.com/NVIDIA/cutile-python>`_.

The ``samples/`` directory contains tile kernels, including vector addition,
matrix multiplication, FFT, and attention. Run individual samples with Python:

.. code-block:: bash

   python samples/quickstart/VectorAdd_quickstart.py
   python samples/MatMul.py

Run the sample test suite with:

.. code-block:: bash

   python -m pytest samples

The :doc:`/tile/index` and :doc:`/language` cover execution and language semantics.
