import cuda.tile as ct
from cuda.tile.jax import OutputPlaceholder, InputOutput, cutile_call

import jax
import jax.numpy as jnp
import numpy as np

jnp.set_printoptions(precision=1)

# begin-snippet
{{body}}
# end-snippet
