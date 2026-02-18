<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

Release Notes
=============
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
