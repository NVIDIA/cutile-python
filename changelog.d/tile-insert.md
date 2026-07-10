- Added ``ct.insert()`` (and the ``Tile.insert()`` method) to store a subtile
  into a bigger tile. It is the inverse of ``ct.extract()`` and returns a new
  tile equal to the destination except for the subtile at the given grid index,
  which is replaced by the source.
