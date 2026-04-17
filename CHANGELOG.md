<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

Release Notes
=============
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
