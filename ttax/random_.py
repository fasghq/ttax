import numpy as np
import jax
import jax.numpy as jnp
from base_class import TT


def tensor(rng, shape, tt_rank=2, dtype=jnp.float32):
  """Generate a random TT-tensor of the given shape and TT-rank.
  """
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)

  num_dims = shape.size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1)
    tt_rank = np.insert(tt_rank, 0, 1)
    tt_rank = np.append(tt_rank, 1)

  tt_rank = tt_rank.astype(int)

  tt_cores = []
  rng_arr = jax.random.split(rng, num_dims)
  for i in range(num_dims):
    curr_core_shape = (tt_rank[i], shape[i], tt_rank[i + 1])
    tt_cores.append(jax.random.normal(rng_arr[i], curr_core_shape, dtype=dtype))

  return TT(tt_cores)
