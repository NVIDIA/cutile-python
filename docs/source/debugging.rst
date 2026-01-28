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
