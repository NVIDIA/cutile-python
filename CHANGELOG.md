<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

Release Notes
=============

1.4.0 (2026-05-26)
------------------

This release highlights compatability with CTK 13.3 which introduces
Hopper(sm_90) GPU support, block-scaled MMA, new float dtypes, pack/unpack
operations, atomic store operations, load and store with advanced indexing,
tiled view with gapped or overlapped tile access, and support for large arrays
and scalars. It also adds support for Python 3.14 including free-threading,
star "\*" expression, and frozen dataclasses, as well as integration with JAX.

### CTK 13.3 features
- Support Hopper (sm_90 family) GPUs.
- Add `float8_e8m0fnu` and `float4_e2m1fn` restricted float dtype.
- Add {py:func}`ct.mma_scaled() <cuda.tile.mma_scaled>` operation for
  block-scaled matrix multiply-accumulate.
- Add {py:func}`ct.pack_to_bytes() <cuda.tile.pack_to_bytes>` operation that
  flattens a tile and reinterprets its raw bytes as a 1D uint8 tile;
  {py:func}`ct.unpack_from_bytes() <cuda.tile.unpack_from_bytes>`
  is the inverse of {py:func}`ct.pack_to_bytes() <cuda.tile.pack_to_bytes>`.
- Add {py:func}`ct.load_advanced_indexing() <cuda.tile.load_advanced_indexing>`
  and {py:func}`ct.store_advanced_indexing() <cuda.tile.store_advanced_indexing>`
  for gathering/scattering along one dimension while slicing on other
  dimensions.
- Add {py:meth}`atomic_store_add <cuda.tile.TiledView.atomic_store_add>` and
  more atomic methods on
  {py:class}`TiledView <cuda.tile.TiledView>` for performing element-wise
  atomic read-modify-write operations on a tiled view at a given tile index.
- Add support for {py:meth}`Array.tiled_view() <cuda.tile.Array.tiled_view>`
  with `traversal_steps` for creating a tiled space with overlapped or spaced tiles.
- {py:class}`ByTarget <cuda.tile.ByTarget>` now accepts a `default` value that
  applies to all architectures (e.g. `ByTarget(sm_100=8, default=2)`), allowing
  generated TileIR bytecode to be independent of the GPU architecture.
- Add {py:func}`ct.astile() <cuda.tile.astile>` for creating a tile from a
  scalar (yielding a 0-d tile) or a (possibly nested) tuple of scalars whose
  nesting determines the tile's shape.
- Support large arrays (>2B elements):
    - New {py:data}`ct.IndexedWithInt64 <cuda.tile.IndexedWithInt64>` annotation
      for array kernel parameters whose shape or stride values exceed the range
      of a 32-bit integer. Arrays without the annotation continue to use
      `int32` for shape and stride.
    - New {py:data}`ct.ScalarInt64 <cuda.tile.ScalarInt64>` annotation that
      forces a scalar integer kernel parameter to be inferred as `int64`
      instead of the default `int32`.
- Add `use_fast_acc` option to {py:func}`ct.mma() <cuda.tile.mma>`
  to enable fast accumulation mode for FP8 inputs (`float8_e4m3fn`,
  `float8_e5m2`) on Hopper GPUs.
- Add a new kernel hint `num_worker_warps` to {py:class}`ct.kernel <cuda.tile.kernel>`.
- {py:func}`ct.atomic_add() <cuda.tile.atomic_add>` now supports `bfloat16`
  operands on Hopper (sm_90) and newer architectures.
- Optional `rounding_mode` parameter for {py:func}`ct.exp() <cuda.tile.exp>`
  (supports `RoundingMode.FULL` and `RoundingMode.APPROX` for f32).


### Python features
- Add Python 3.14 and Python 3.14t (free-threading) support.
- Add support for frozen dataclasses.
- Add support for passing a tuple as a starred function argument, e.g. `foo(a, *b, c)`.
- Add support for variadic parameters in user-defined functions, e.g. `def foo(*args)`.

### Enhancements
- {py:func}`ct.extract() <cuda.tile.extract>` now raises a compile-time
  `TileTypeError` when a constant index is out of bounds for the tile grid.
  Dynamic indices are unaffected.
- {py:func}`ct.floordiv() <cuda.tile.floordiv>` and the `//` operator now
  support floating-point operands.

### Ecosystem
- Add support for JAX via {py:func}`ct.jax.cutile_call() <cuda.tile.jax.cutile_call>`.

1.3.0 (2026-04-20)
------------------
### Features
- Add API for ahead-of-time compilation and export via {py:func}`compilation.export_kernel() <cuda.tile.compilation.export_kernel>`.<br>
  See the {doc}`Compilation and Export </compilation>` section for more details.
- Add API for autotuning via {py:func}`tune.exhaustive_search() <cuda.tile.tune.exhaustive_search>` and the following helpers:
    - Add API {py:meth}`kernel.replace_hints() <cuda.tile.kernel.replace_hints>` to get a new kernel with updated hints.
    - Add API function {py:func}`compiler_timeout() <cuda.tile.compiler_timeout>` for temporarily setting the
      timeout on the tileiras compiler.<br>
  See the {ref}`Autotuning <autotuning>` section for more details.
- Add API {py:meth}`Array.tiled_view() <cuda.tile.Array.tiled_view>` to create a tiled view of an array
  with a fixed tile shape and padding mode.

### Enhancements
- Add support for specifying `memory_order` and `memory_scope` on `cuda.tile.load` and
  `cuda.tile.store` operations.
- Improve `print()` to handle tuple and nested fstring.


### Bug Fixes
- Fix a bug where restricted float dtype with simple reduce and scan did not
  raise proper `TileTypeError`.

### ABI Changes
- Change kernel ABI convention to omit parameters annotated with `cuda.tile.Constant`.

1.2.0 (2026-03-05)
------------------
### CTK 13.2 features
- Support Ampere and Ada (sm80 family) GPUs.
- Support `pip install cuda-tile[tileiras]` to use `tileiras` from Python environment
  without system-wide CTK installation.
- Add `ct.atan2(y, x)` operation for computing the arctangent of y/x.
- Add optional `rounding_mode` parameter for `ct.tanh()`, supporting `RoundingMode.FULL` and
  `RoundingMode.APPROX`.
- Compiling FP8 operations for sm80 family GPUs will raise `TileUnsupportedFeatureError`.
- Setting `opt_level=0` on `ct.kernel` is no longer required for `ct.printf()` and `ct.print()`.


### Features
- Add `ct.static_iter` keyword that enables compile-time `for` loops.
- Add `ct.static_assert` keyword that can be used to assert that a condition is true at compile time.
- Add `ct.static_eval` keyword that enables compile-time evaluation using the host Python interpreter.
- Add `ct.scan()` for custom scan.
- Add `ct.isnan()`.
- Add `print()` and `ct.print()` that supports python-style print and f-strings.
- Add optional `mask` parameter to `ct.gather()` and `ct.scatter()` for custom boolean masking.
- Operator `+` can now be used to concatenate tuples.
- Support unpacking nested tuples (e.g., `a, (b, c) = t`) and using square brackets
  for unpacking (e.g., `[a, b] = 1, 2`).
- Add bytecode-to-cubin disk cache to avoid recompilation of unchanged kernels.
  Controlled by `CUDA_TILE_CACHE_DIR` and `CUDA_TILE_CACHE_SIZE`.

### Bug Fixes
- Fix a bug where `nan != nan` returns False.
- Fix "potentially undefined variable `$retval`" error when a helper function
  returns after a `while` loop that contains no early return.
- Fix the missing column indicator in error messages when the underlined text is only one
  character wide.
- Add a missing check for unpacking a tuple with too many values. For example, `a, b = 1, 2, 3`
  now raises an error instead of silently discarding the extra value.
- Fix a bug where the promoted dtype of uint16 and uint64 was incorrectly set to uint32.


### Enhancements
- Erase the distinction between scalars and zero-dimensional tiles.
  They are now completely interchangeable.
- `~x` for const boolean `x` will raise a TypeError to prevent inconsistent
  results compared to `~x` on a boolean Tile.
- Add `TileUnsupportedFeatureError` to the public API.


1.1.0 (2026-01-30)
------------------
### Features
- Add support for nested functions and lambdas.
- Add support for custom reduction via `ct.reduce()`.
- Add `Array.slice(axis, start, stop)` to create a view of an array sliced along a single axis. 
  The result shares memory with the original array (no data copy).

### Bug Fixes
- Fix reductions with multiple axes specified in non-increasing order.
- Fix a bug when pattern matching (FusedMultiplyAdd) attempts to remove a value that is used by the new operation.

### Enhancements
- Allow assignments with type annotations. Type annotations are ignored.
- Support constructors of built-in numeric types (bool, int, float), e.g., `float('inf')`.
- Lift the ban on recursive helper function calls. Instead, add a limit on recursion depth.
  Add a new exception class `TileRecursionError`, thrown at compile time when the recursion limit
  is reached during function call inlining.
- Improve error messages for type mismatches in control flow statements.
- Relax type checking rules for variables that are assigned a different type
  depending on the branch taken: it is now only an error if the variable is used
  afterwards.
- Stricter rules for potentially-undefined variable detection: if a variable
  is first assigned inside a `for` loop, and then used after the loop,
  it is now an error because the loop may take zero iterations, resulting
  in a use of an undefined variable.
- Include a full cuTile traceback in error messages. Improve formatting of code locations;
  include function names, remove unnecessary characters to reduce line lengths.
- Delay the loading of CUDA driver until kernel launch.
- Expose the `TileError` base class in the public API.
- Add `ct.abs()` for completeness.


1.0.1 (2025-12-18)
------------------
### Bug Fixes
- Fix a bug in hash function that resulted in potential performance regression
    for kernels with many specializations.
- Fix a bug where an if statement within a loop can trigger an internal compiler error.
- Fix SliceType `__eq__` comparison logic.

### Enhancements
- Improve error message for `ct.cat()`.
- Support `is not None` comparison.


1.0.0 (2025-12-02)
------------------
Initial release.
