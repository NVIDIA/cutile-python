Fixed bug in ceiling division calculations. It used to be computed using `(x + y - 1) // y` and
`(x - 1) // y + 1`, which give incorrect values for when both x and y are negative numbers.