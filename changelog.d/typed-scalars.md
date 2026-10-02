- Added `ct.wrapping_add()`, `ct.wrapping_sub()` and `ct.wrapping_mul()` for explicit
  wrap-around integer arithmetic.
- **Breaking change:** integer overflow in arithmetic on strictly typed scalar constants is now
  a compile-time error. Previously, an expression like `ct.uint8(255) + ct.uint8(3)` silently
  wrapped around to 2. To get wrap-around behavior, use `ct.wrapping_add()`, `ct.wrapping_sub()`
  or `ct.wrapping_mul()` instead.
- Added support for strictly typed constants in `ct.static_eval()`, `ct.static_assert()` and
  other statically evaluated expressions. Previously, a strictly typed scalar constant decayed to
  a Python `int` inside such expressions. Now it is represented as a `ct.Tile` instance.
