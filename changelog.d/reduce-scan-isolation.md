- Custom ``ct.reduce()`` and ``ct.scan()`` callbacks now keep captured scalar compile-time
  constants inside their isolated bodies. Unsupported runtime and shaped captures are diagnosed
  during cuTile Python compilation, before TileIR is invoked.
