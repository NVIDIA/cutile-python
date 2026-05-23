- Extend ``ct.arange()`` with optional ``start`` and ``step`` arguments:
  ``ct.arange(size, start=0, step=1, dtype=...)``. ``size`` must be a constant
  integer, while ``start`` and ``step`` may be dynamic numbers. For example, 
  ``ct.arange(8, start=7, step=-1, dtype=ct.int32)`` creates a reversed range.
