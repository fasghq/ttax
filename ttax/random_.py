import numpy as np
import jax
import jax.numpy as jnp

from ttax.base_class import TT
from ttax.base_class import TTMatrix


def tensor(rng, shape, tt_rank=2, batch_shape=None, dtype=jnp.float32):
  """Generate a random TT-tensor of the given shape and TT-rank.
  """
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  batch_shape = list(batch_shape) if batch_shape else []

  num_dims = shape.size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1)
    tt_rank = np.insert(tt_rank, 0, 1)
    tt_rank = np.append(tt_rank, 1)

  tt_rank = tt_rank.astype(int)

  tt_cores = []
  rng_arr = jax.random.split(rng, num_dims)
  for i in range(num_dims):
    curr_core_shape = [tt_rank[i], shape[i], tt_rank[i + 1]]
    curr_core_shape = batch_shape + curr_core_shape
    tt_cores.append(jax.random.normal(rng_arr[i], curr_core_shape, dtype=dtype))

  return TT(tt_cores)


def matrix(rng, shape, tt_rank=2, batch_shape=None, dtype=jnp.float32):
  """Generate a random TT-matrix of the given shape and TT-rank.
  """
  shape = [np.array(shape[0]), np.array(shape[1])]
  tt_rank = np.array(tt_rank)
  batch_shape = list(batch_shape) if batch_shape else []

  num_dims = shape[0].size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1)
    tt_rank = np.insert(tt_rank, 0, 1)
    tt_rank = np.append(tt_rank, 1)

  tt_rank = tt_rank.astype(int)

  tt_cores = []
  rng_arr = jax.random.split(rng, num_dims)
  for i in range(num_dims):
    curr_core_shape = [tt_rank[i], shape[0][i], shape[1][i], tt_rank[i + 1]]
    curr_core_shape = batch_shape + curr_core_shape
    tt_cores.append(jax.random.normal(rng_arr[i], curr_core_shape, dtype=dtype))

  return TTMatrix(tt_cores)
