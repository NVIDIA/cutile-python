.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.tile

Debugging
=========

Exception Types
---------------

.. autoclass:: TileSyntaxError()
.. autoclass:: TileTypeError()
.. autoclass:: TileValueError()
.. autoclass:: TileUnsupportedFeatureError()
.. autoclass:: TileCompilerExecutionError()
.. autoclass:: TileCompilerTimeoutError()


Compiler Timeout
----------------

.. autofunction:: cuda.tile.compiler_timeout


Environment Variables
---------------------

The following environment variables control diagnostics and compilation
artifacts.

``CUDA_TILE_ENABLE_CRASH_DUMP=1`` enables crash-dump archives containing TileIR
bytecode. These archives provide compiler inputs for bug reports involving
:class:`TileCompilerExecutionError` or :class:`TileCompilerTimeoutError`.

``CUDA_TILE_COMPILER_TIMEOUT_SEC`` limits the time the TileIR compiler
``tileiras`` can take.

``CUDA_TILE_LOGS=CUTILEIR`` prints cuTile Python IR to stderr during compilation.

``CUDA_TILE_LOGS=TILEIR`` prints TileIR text to stderr. Requires
CUDA Toolkit 13.4 or later.

``CUDA_TILE_DUMP_TILEIR`` specifies a directory for TileIR text files.
Requires CUDA Toolkit 13.4 or later.

``CUDA_TILE_TEMP_DIR`` specifies the directory for temporary files.

``CUDA_TILE_CACHE_DIR`` specifies the directory for the bytecode-to-cubin disk
cache. Compiled cubins are cached here to avoid recompilation of unchanged
kernels. Values ``0``, ``off``, ``none``, or an empty string disable caching.
Defaults to ``~/.cache/cutile-python``.

``CUDA_TILE_CACHE_SIZE`` specifies the maximum disk cache size in bytes.
Oldest entries are evicted when the cache exceeds this limit. Defaults to
2 GB (2147483648).


Inspecting the Compilation Cache
--------------------------------

The ``cutile-cache log`` command displays cached compilations in
most-recently-accessed order. Each entry shows the kernel's mangled name,
compiler version, compilation date and duration, CUBIN size, and optimization
remarks::

    cutile-cache log

Compilation remarks are captured in YAML format when using ``tileiras`` 13.4
or newer. Entries produced by earlier cuTile Python or compiler versions remain
readable but may show metadata as unavailable.
