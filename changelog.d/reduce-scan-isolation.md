- Custom ``ct.reduce()`` and ``ct.scan()`` callbacks now keep captured scalar compile-time
  constants inside their own bodies, and every value inside a callback must be a scalar tile.
  Capturing runtime values or non-scalar constants is rejected with a compile-time error.
  Note: capturing runtime values in these callbacks previously compiled and is no longer
  supported.
