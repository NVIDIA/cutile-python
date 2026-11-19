- Fixed loop splitting computing a wrong split point for ``if i > c`` and ``if i <= c``
  when ``c`` is the maximum value of the loop index type. The split point ``c + 1``
  wrapped around, so every iteration took the wrong branch and the kernel returned
  incorrect results.
