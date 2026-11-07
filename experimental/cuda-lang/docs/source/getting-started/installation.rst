.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

Installation
============

The ``cuda-tile`` package includes both ``cuda.lang`` and ``cuda.tile``. It will
be renamed to ``cuda-languages`` when ``cuda.lang`` is stable.

Prerequisites
-------------

``cuda.tile`` supports Linux x86_64, Linux aarch64, Windows x86_64 and Windows on Arm.
``cuda.lang`` supports Linux x86_64 and Linux aarch64. Windows support for
``cuda.lang`` is coming soon.

Supported Python versions are 3.11 through 3.15, including free-threaded builds.
Both namespaces require NVIDIA Driver r580 or later.

``cuda.lang`` requires an NVIDIA GPU with compute capability 7.5 or higher.
Individual operations can require newer architectures; see their target
requirements in :doc:`/operations`.

.. _getting-started-gpu-compatibility:

``cuda.tile`` requires a Tile IR toolchain that supports the GPU architecture.
Choose ``tileiras`` and its matching compiler components using this table:

.. list-table:: cuda.tile GPU/compiler compatibility
   :header-rows: 1
   :widths: 20 45 35

   * - GPU family
     - Compute capability
     - First supporting Tile IR version
   * - Ampere/Ada
     - 8.0, 8.6, 8.7, 8.8, 8.9
     - 13.2
   * - Hopper
     - 9.0
     - 13.3
   * - Blackwell
     - 10.0, 10.3, 11.0, 12.0, 12.1
     - 13.1

For the full target list, including preview support, see NVIDIA's `Tile IR
Supported Architectures table
<https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#supported-architectures>`_.
Individual features and datatypes can have additional requirements; see the
`Tile IR feature availability and hardware support matrix
<https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#feature-availability-emulation>`_.

Installing the Package
----------------------

``cuda.tile`` depends on the CUDA TileIR compiler, ``tileiras``, which in turn
depends on ``ptxas`` and ``libnvvm`` from the CUDA Toolkit. These tools can be
installed as Python packages in a virtual environment or as part of a
system-wide CUDA Toolkit installation.

Without a system-wide CUDA Toolkit, install the package and its compiler
dependencies into your Python environment:

.. code-block:: bash

   python -m pip install --upgrade "cuda-tile[tileiras]"

The extra installs ``nvidia-cuda-tileiras``, ``nvidia-cuda-nvcc``, and
``nvidia-nvvm``. Their major.minor versions must match.

Choosing a Compiler Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cuTile supports multiple versions of ``tileiras``. To select a particular
major.minor version, check the :ref:`GPU/compiler compatibility table
<getting-started-gpu-compatibility>` and install the matching compiler components
together. For example, to use version 13.3:

.. code-block:: bash

   python -m pip install "cuda-toolkit[tileiras,nvvm,nvcc]>=13.3,<13.4"

Using a System-Wide CUDA Toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a system-wide CUDA Toolkit provides a ``tileiras`` version supporting your
GPU, install ``cuda-tile`` without the extra:

.. code-block:: bash

   python -m pip install cuda-tile

cuTile automatically searches the CUDA Toolkit installation for ``tileiras``.

``cuda.lang`` includes a compiler distinct from ``tileiras`` and does not
require a CUDA Toolkit installation to compile SIMT kernels.

.. _getting-started-sample-dependencies:

Example Dependencies and Verification
-----------------------------------------

Both quickstarts use CuPy for GPU arrays, streams, and result validation.
With CUDA 13.x installed, install CuPy with:

.. code-block:: bash

   python -m pip install cupy-cuda13x

For other configurations, including installation without a system-wide CUDA
Toolkit, see the `CuPy installation instructions
<https://docs.cupy.dev/en/stable/install.html>`_.

The additional examples in :doc:`lang` and :doc:`tile` use packages such as
PyTorch, NumPy, and pytest. Install the latter two with:

.. code-block:: bash

   python -m pip install pytest numpy

For a CUDA-enabled PyTorch build compatible with your environment, follow the
`PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_.

Verify that CuPy can use the GPU:

.. code-block:: bash

   python -c "import cupy as cp; assert cp.cuda.is_available(); print(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'])"

Check that both namespaces can be imported:

.. code-block:: bash

   python -c "import cuda.tile"
   python -c "import cuda.lang"

Developer Tools
---------------

`NVIDIA Nsight Compute <https://developer.nvidia.com/nsight-compute>`_ can
profile cuTile kernels in the same way as SIMT CUDA kernels. After installing
Nsight Compute and saving the :doc:`tile` example as ``vector_add_tile.py``,
capture a profile with:

.. code-block:: bash

   ncu -o VectorAddProfile --set detailed python vector_add_tile.py

Open the resulting profile in the Nsight Compute graphical interface and
select the ``vector_add`` kernel to inspect its statistics.

.. note::

   Capturing detailed statistics for cuTile kernels requires NVIDIA Driver
   r580.126.09 or later on Linux, or r582.16 or later on Windows.

Continue with :doc:`lang` or :doc:`tile`.
