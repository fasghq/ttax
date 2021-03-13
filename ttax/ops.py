import functools
import jax
import jax.numpy as jnp
import numpy as np

from ttax.base_class import TTBase
from ttax.base_class import TT
from ttax.base_class import TTMatrix
from ttax.compile import TTEinsum, to_function, I_OR_IJ


def tt_vmap(num_batch_args=None):
  """Decorator which makes a function support batch TT-inputs.
  Arg:
    num_batch_args - integer or None.
      If None, than function will be mapped over all arguments.
      If integer, specifies the count of first arguments to map over, e.g.
      num_batch_args=n means that function will be mapped over 
      first n arguments.
  Returns:
    Decorator
  Comments:
    The function is vmapped num_batch_dims times, as it supports 
    multidimensional batches. The number of batch dimension to be 
    mapped over is shown by num_batch_dims property and should be 
    the same for all args of the function, by which it will be mapped over. 
    Otherwise such axis should be specified by num_batch_args.
  """
  def tt_vmap_fixed_batching_pattern(func):
    @functools.wraps(func)
    def vectorized_func(*args, **kwargs):
      tt_arg = args[0]  # TODO: what if only kwargs are present?
      if tt_arg.num_batch_dims == 0:
        return func(*args, **kwargs)
      else:
        if num_batch_args is not None:
          num_non_batch_args = len(args) + len(kwargs) - num_batch_args
          in_axis = [0] * num_batch_args + [None] * num_non_batch_args
          num_args = num_batch_args
        else:
          num_args = len(args) + len(kwargs)
          in_axis = [0] * num_args
        if num_args > 1 and (isinstance(args[1], TTMatrix) or
                                 isinstance(args[1], TT)):
          if args[0].is_tt_matrix != args[1].is_tt_matrix:
            raise ValueError('Types of the arguments are different.')
          if not are_batches_broadcastable(args[0], args[1]):
            raise ValueError('The batch sizes are different and not 1, '
                             'broadcasting is not available.')
          broadcast_shape = np.maximum(list(args[0].batch_shape), 
                                       list(args[1].batch_shape))
          new_args = list(args)
          if args[0].is_tt_matrix:
            for i, tt in enumerate(args[:2]):
              new_cores = []
              for core in tt.tt_cores:
                core = jnp.broadcast_to(core, list(broadcast_shape) +
                                              list(core.shape[-4:]))
                new_cores.append(core)
              new_args[i] = TTMatrix(new_cores)
          else:
            for i, tt in enumerate(args[:2]):
              new_cores = []
              for core in tt.tt_cores:
                core = jnp.broadcast_to(core, list(broadcast_shape) + 
                                              list(core.shape[-3:]))
                new_cores.append(core)
              new_args[i] = TT(new_cores)
        else:
          new_args = args
        vmapped = func
        for _ in range(tt_arg.num_batch_dims):
          vmapped = jax.vmap(vmapped, in_axis)
        return vmapped(*new_args, **kwargs)
    return vectorized_func
  return tt_vmap_fixed_batching_pattern


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


@tt_vmap()
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


def are_batches_broadcastable(tt_a, tt_b):
  """Returns the result of compatibility check of 2 tensors' batches: 
  True if batches are compatible and False otherwise.
  The batch sizes should be equal otherwise at least one of them 
  should equal to 1 for broadcasting to be available.
  Args:
    tt_a: TT or TT-Matrix
    tt_b: TT or TT-Matrix
  Returns:
    bool
  """
  if tt_a.num_batch_dims != tt_b.num_batch_dims:
    return False
  for a, b in zip(tt_a.batch_shape, tt_b.batch_shape):
    if a == 1 or b == 1 or a == b:
      pass
    else:
      return False
  return True
