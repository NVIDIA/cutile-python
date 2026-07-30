- Added support for specializing array stride dimensions to their launch-time values, making them
  compile-time constants inside the kernel. Annotate the parameter with
  `Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(...))]`, listing the dimensions to
  specialize. The annotation is the sole source of static strides for that array: once any dimension
  is listed, the dispatcher stops inferring ``stride == 1`` for *every* dimension of it, so list the
  contiguous dimension explicitly to keep it a compile-time constant.
- **Non-breaking change**: an array's inferred ``stride == 1`` dimension is no longer part of its
  type (``Array[f32,(?,?):(?,1)]`` is now ``Array[f32,(?,?):(?,?)]``), unless annotated. This allows
  more permissive type unification in control flow. For example:

  ```python
  if cond:
      x = A  # A has inferred stride == 1 at dim 0
  else:
      x = B  # B has inferred stride == 1 at dim 1
  ```

  is now allowed, where previously it raised a ``TypeError``.
