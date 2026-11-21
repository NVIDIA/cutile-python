- `opt_level=None` (including inside `ByTarget`) is now rejected with a `ValueError`
  when the kernel is defined, instead of failing with an `AssertionError` at compile time.
