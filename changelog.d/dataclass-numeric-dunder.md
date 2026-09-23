- Dataclasses now support default comparisons via `==`, `!=`, `<`, `<=`, `>` and `>=`.
- Dataclasses now support user-defined overloads of binary arithmetic operators (`__add__()`,
  `__radd__()`, `__sub__()`, `__rsub__()` etc.), comparison operators (`__eq__()`, `__ne__()`,
  `__lt__()`, etc.), and `len()` (via `__len__()`).
