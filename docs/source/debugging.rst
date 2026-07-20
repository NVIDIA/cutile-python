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

The following environment variables are useful when
the above exceptions are encountered during kernel
development.

Set ``CUDA_TILE_ENABLE_CRASH_DUMP=1`` to enable dumping
an archive including the TileIR bytecode
for submitting a bug report on :class:`TileCompilerExecutionError`
or :class:`TileCompilerTimeoutError`.

Set ``CUDA_TILE_COMPILER_TIMEOUT_SEC`` to limit the
time the TileIR compiler `tileiras` can take.

Set ``CUDA_TILE_LOGS=CUTILEIR`` to print cuTile Python
IR during compilation to stderr. This is useful when
debugging :class:`TileTypeError`.

Set ``CUDA_TILE_TEMP_DIR`` to configure the directory
for storing temporary files.

Set ``CUDA_TILE_CACHE_DIR`` to configure the directory
for the bytecode-to-cubin disk cache. Compiled cubins
are cached here to avoid recompilation of unchanged
kernels. Set to ``0``, ``off``, ``none``, or an empty
string to disable caching. Defaults to
``~/.cache/cutile-python``.

Set ``CUDA_TILE_CACHE_SIZE`` to configure the maximum
disk cache size in bytes. Oldest entries are evicted
when the cache exceeds this limit. Defaults to
2 GB (2147483648).


Inspecting the Compilation Cache
--------------------------------

Use the ``cutile-cache log`` command to page through cached compilations in
most-recently-accessed order. Each entry shows the kernel's mangled name,
compiler version, compilation date and duration, CUBIN size, and optimization
remarks::

    cutile-cache log

Compilation remarks are captured in YAML format when using ``tileiras`` 13.4
or newer. Entries produced by earlier cuTile Python or compiler versions remain
readable but may show metadata as unavailable.
