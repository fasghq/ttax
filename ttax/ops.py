import functools
import jax
import jax.numpy as jnp

from ttax.base_class import TTBase
from ttax.base_class import TT
from ttax.base_class import TTMatrix
from ttax.compile import TTEinsum, to_function, I_OR_IJ


def tt_vmap(func):
  """Decorator which makes a function support batch TT-inputs."""
  @functools.wraps(func)
  def vectorized_func(*args, **kwargs):
    tt_arg = args[0]  # TODO: what if only kwargs are present?
    if tt_arg.num_batch_dims == 0:
      return func(*args, **kwargs)
    else:
      # Vmap everything num_batch_dims times.
      vmapped = func
      for _ in range(tt_arg.num_batch_dims):
        vmapped = jax.vmap(vmapped)
      return vmapped(*args, **kwargs)
  return vectorized_func


def full_tt_tensor(tt: TT) -> jnp.array:
  """Converts TT into a regular tensor.
  """
  num_dims = tt.ndim

  dtype = tt.tt_cores[0].dtype  # TODO: make tt.dtype.
  res = jnp.ones((1, 1), dtype=dtype)
  for i in range(num_dims):
    curr_core = tt.tt_cores[i]
    right_rank = curr_core.shape[-1]
    res = jnp.einsum('pa,aib->pib', res, curr_core)
    res = res.reshape(-1, right_rank)

  return jnp.reshape(res, tt.shape)


def full_tt_matrix(tt: TTMatrix) -> jnp.array:
  """Converts TT-matrix into a regular matrix.
  """
  num_dims = tt.ndim

  dtype = tt.tt_cores[0].dtype  # TODO: make tt.dtype.
  res = jnp.ones((1, 1), dtype=dtype)
  for i in range(num_dims):
    curr_core = tt.tt_cores[i]
    right_rank = curr_core.shape[-1]
    res = jnp.einsum('pa,aijb->pijb', res, curr_core)
    res = res.reshape(-1, right_rank)

  raw_shape = tt.raw_tensor_shape
  intermediate_shape = []
  for i in range(num_dims):
    intermediate_shape.append(raw_shape[0][i])
    intermediate_shape.append(raw_shape[1][i])
  res = jnp.reshape(res, list(tt.batch_shape) + intermediate_shape)
  transpose = []
  for i in range(0, 2 * num_dims, 2):
    transpose.append(i)
  for i in range(1, 2 * num_dims, 2):
    transpose.append(i)
  res = jnp.transpose(res, transpose)
  return jnp.reshape(res, tt.shape)


@tt_vmap
def full(tt: TTBase) -> jnp.array:
  """Converts TT or TTMatrix into a regular tensor/matrix.
  """
  if isinstance(tt, TT):
    return full_tt_tensor(tt)
  elif isinstance(tt, TTMatrix):
    return full_tt_matrix(tt)


def multiply(a, b):
  tt_einsum = TTEinsum(
      inputs=[['a', I_OR_IJ, 'b'], ['c', I_OR_IJ, 'd']],
      output=['ac', I_OR_IJ, 'bd'],
      how_to_apply='independent'
  )
  func = to_function(tt_einsum)
  return func(a, b)


def flat_inner(a, b):
  tt_einsum = TTEinsum(
      inputs=[['a', I_OR_IJ, 'b'], ['c', I_OR_IJ, 'd'], ['a', 'c']],
      output=['b', 'd'],
      how_to_apply='cumulative'
  )
  func = to_function(tt_einsum)
  res = jnp.squeeze(func(a, b)[-1])
  return res


def matmul(a, b):
  # TODO: support TT x dense matmul.
  tt_einsum = TTEinsum(
      inputs=[['a', 'ij', 'b'], ['c', 'jk', 'd']],
      output=['ac', 'ik', 'bd'],
      how_to_apply='independent'
  )
  func = to_function(tt_einsum)
  return func(a, b)
