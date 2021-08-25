import numpy as np
import jax
import jax.numpy as jnp

from ttax.base_class import TT
from ttax.base_class import TTMatrix


def tensor(rng, shape, tt_rank=2, batch_shape=None, dtype=jnp.float32):
  """Generate a random `TT-Tensor` of the given shape and `TT-rank`.
  
  :param rng: JAX PRNG key
  :type rng: random state is described by two unsigned 32-bit integers
  :param shape: desired tensor shape
  :type shape: array
  :param tt_rank: desired `TT-ranks` of `TT-Tensor`
  :type tt_rank: single number for equal `TT-ranks` or array specifying all `TT-ranks`
  :param batch_shape: desired batch shape of `TT-Tensor`
  :type batch_shape: array
  :param dtype: type of elements in `TT-Tensor`
  :type dtype: `dtype`
  :return: generated `TT-Tensor`
  :rtype: TT
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
  """Generate a random `TT-Matrix` of the given shape and `TT-rank`.
    
  :param rng: JAX PRNG key
  :type rng: random state is described by two unsigned 32-bit integers
  :param shape: desired tensor shape. Also supports omitting one of the dimensions
        matrix(..., shape=((2, 2, 2), None))
      and
        matrix(..., shape=(None, (2, 2, 2)))
      will create an 8-element column/row vector.
  :type shape: tuple
  :param tt_rank: desired `TT-ranks` of `TT-Matrix`
  :type tt_rank: single number for equal `TT-ranks` or array specifying all `TT-ranks`
  :param batch_shape: desired batch shape of `TT-Matrix`
  :type batch_shape: tuple
  :param dtype: type of elements in `TT-Matrix`
  :type dtype: `dtype`
  :return: generated `TT-Matrix`
  :rtype: TTMatrix
  :raises [ValueError]: if shape is (None, None)
  """
  if shape == (None, None):
    raise ValueError("At least one of shape elements must not be None")
  if None in shape:
    shape = [
      np.array(shape[0]) if shape[0] is not None else np.ones(len(shape[1]), dtype=int),
      np.array(shape[1]) if shape[1] is not None else np.ones(len(shape[0]), dtype=int),
    ]
  else:
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
