.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. _quickstart:

Quickstart
===============================================================================

This page will guide you through getting setup and running with cuTile Python, including running a first example.

.. _quickstart-prereqs:

Prerequisites
-------------------------------------------------------------------------------

cuTile Python requires the following:

    - Linux x86_64, Linux aarch64 or Windows x86_64
    - A GPU with compute capability 10.x, 11.x or 12.x
    - NVIDIA Driver r580 or later
    - Python version 3.10, 3.11, 3.12 or 3.13


Installing cuTile Python
-------------------------------------------------------------------------------

cuTile Python depends on CUDA TileIR compiler ``tileiras``, which futher depends on ``ptxas`` and ``libnvvm``
from the CUDA Toolkit.

If your system does not have system-wide CUDA Toolkit (13.1+), you can install cuTile Python along with ``[tileiras]``,
which installs ``nvidia-cuda-tileiras``, ``nvidia-cuda-nvcc`` and
``nvidia-nvvm`` into your Python virtual environment.

.. code-block::  bash

    pip install cuda-tile[tileiras]

Note: the package versions for ``nvidia-cuda-tileiras``, ``nvidia-cuda-nvcc`` and
``nvidia-nvvm`` must match up to the same major.minor version.


Alternatively if you already have system-wide CUDA Toolkit (13.1+) installed, you can install cuTile Python as a
standalone package. cuTile automatically searches for ``tileiras`` from the location of CUDA Toolkit.

.. code-block::  bash

    pip install cuda-tile


Other Packages
-------------------------------------------------------------------------------

Some of the cuTile Python samples also use other Python packages. 

The quickstart sample on this page uses cupy, which can be installed with:

.. code-block::  bash 
    
    pip install cupy-cuda13x


The cuTile Python samples in the ``samples/`` directory also use pytest, torch, and numpy packages.

For PyTorch installation instructions, see `<https://pytorch.org/get-started/locally/>`__.

Pytest and Numpy can be installed with:

.. code-block::  bash

    pip install pytest numpy


Example Code
-------------------------------------------------------------------------------

The following example shows vector addition, a typical first kernel for CUDA, but uses cuTile for tile-based programming. This makes use of a 1-dimensional tile to add two 1-dimensional vectors. 

This example shows a structure common to cuTile kernels:

* Load one or more tiles from GPU memory
* Perform computation(s) on the tile(s), resulting in new tile(s)
* Write the resulting tile(s) out to GPU memory

In this case, the kernel loads tiles from two vectors, ``a`` and ``b``. These loads create tiles called ``a_tile`` and ``b_tile``. These tiles are added together to form a third tile, called ``result``. In the last step, the kernel stores the ``result`` tile to the output vector ``c``.
More samples can be found in the cuTile Python `repository <https://github.com/nvidia/cutile-python>`_.

.. literalinclude:: ../../samples/quickstart/VectorAdd_quickstart.py
   :language: python
   :dedent:


Run this from a command line as shown below. If everything has been setup correctly, the test will print that the example passed.

.. code-block:: bash

    $ python3 samples/quickstart/VectorAdd_quickstart.py
    ✓ vector_add_example passed!

To run more of the cuTile Python examples, you can directly run the samples by invoking them in the same way as the quickstart example:

.. code-block:: bash   

    $ python3 samples/FFT.py
    # output not shown

You can also use pytest to run all the samples:

.. code-block:: bash

    $  pytest samples
    ========================= test session starts =========================
    platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0
    rootdir: /home/ascudiero/sw/cutile-python
    configfile: pytest.ini
    collected 6 items                                                               

    samples/test_samples.py ......                                  [100%]

    ========================= 6 passed in 30.74s ==========================

Developer Tools
-------------------------------------------------------------------------------

`NVIDIA Nsight Compute <https://developer.nvidia.com/nsight-compute>`__ can profile cuTile Python kernels in the same way as SIMT CUDA kernels. With NVIDIA Nsight Compute installed, the quickstart vector addition kernel introduced here can be profiled using the following command to create a profile:

.. code-block:: bash

    ncu -o VecAddProfile --set detailed python3 VectorAdd_quickstart.py

This profile can then be loaded in a graphical instance of Nsight Compute and the kernel ``vector_add`` selected to see statistics about the kernel.

.. note:: 
    Capturing detailed statistics for cuTile Python kernels requires running on NVIDIA Driver equals or later than r580.126.09 (linux) or r582.16 (windows).

