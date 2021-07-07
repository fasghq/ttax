import functools
import jax
import jax.numpy as jnp
import numpy as np

from ttax.base_class import TTBase
from ttax.base_class import TT
from ttax.base_class import TTMatrix
from ttax.compile import WrappedTT
from ttax.compile import TTEinsum, to_function, I_OR_IJ
from ttax.compile import unwrap_tt
from ttax.utils import is_tt_tensor
from ttax.utils import is_tt_matrix
from ttax.utils import is_tt_object


def tt_vmap(num_batch_args=None):
  """Decorator which makes a function support batch TT-inputs.
  
  :type num_batch_args: int or None
  :param num_batch_args: The amount of arguments that are batches of `TT-objects`.
  
    - If None, than function will be mapped over all arguments.
    - If integer, specifies the count of first arguments to map over, e.g.
      `num_batch_args=n` means that function will be mapped over 
      first `n` arguments.
  :return: Decorator
  
  Comments:
    The function is vmapped `num_batch_dims` times, as it supports 
    multidimensional batches. The number of batch dimension to be 
    mapped over is shown by `num_batch_dims` property and should be 
    the same for all args of the function, by which it will be mapped over. 
    Otherwise such axis should be specified by `num_batch_args`.
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
  """Converts `TT-Tensor` or `TT-Matrix` into a dense format.
  """
  if isinstance(tt, TT):
    return full_tt_tensor(tt)
  elif isinstance(tt, TTMatrix):
    return full_tt_matrix(tt)


def tt_tt_multiply(a, b):
  tt_einsum = TTEinsum(
      inputs=[['a', I_OR_IJ, 'b'], ['c', I_OR_IJ, 'd']],
      output=['ac', I_OR_IJ, 'bd'],
      how_to_apply='independent'
  )
  func = to_function(tt_einsum)
  return func(a, b)


def flat_inner(a, b):
  """Calculate inner product of given `TT-Tensors` or `TT-Matrices` wrapped with `WrappedTT`.
  
  :param a: first argument
  :type a: `WrappedTT`
  :param b: second argument
  :type b: `WrappedTT`
  :rerurn: the result of inner product
  :rtype: `WrappedTT`
  """
  tt_einsum = TTEinsum(
      inputs=[['a', I_OR_IJ, 'b'], ['c', I_OR_IJ, 'd'], ['a', 'c']],
      output=['b', 'd'],
      how_to_apply='cumulative'
  )
  func = to_function(tt_einsum)
  res = jnp.squeeze(func(a, b)[-1])
  return res


def matmul(a, b):
  """Calculate matrix multiplication of given `TT-Matrices` wrapped with `WrappedTT`.
  
  :param a: first argument
  :type a: `WrappedTT`
  :param b: second argument
  :type b: `WrappedTT`
  :rerurn: the result of inner product
  :rtype: `WrappedTT`
  """
  # TODO: support TT x dense matmul.
  tt_einsum = TTEinsum(
      inputs=[['a', 'ij', 'b'], ['c', 'jk', 'd']],
      output=['ac', 'ik', 'bd'],
      how_to_apply='independent'
  )
  func = to_function(tt_einsum)
  return func(a, b)


@tt_vmap()
def add(tt_a, tt_b):
  """Returns a `TT-object` corresponding to elementwise sum `tt_a + tt_b`.
  The shapes of `tt_a` and `tt_b` should coincide.
  Supports broadcasting, e.g. you can add a tensor train with
  batch size 7 and a tensor train with batch size 1:
  
  ``tt_batch.add(tt_single.batch_loc[np.newaxis])``
  
  where ``tt_single.batch_loc[np.newaxis]`` 
  creates a singleton batch dimension.

  :type tt_a: `TT-Tensor` or `TT-Matrix`
  :param tt_a: first argument
  :type tt_b: `TT-Tensor` or `TT-Matrix`
  :param tt_b: second argument
  :rtype: `TT-Tensor` or `TT-Matrix`
  :return: `tt_a + tt_b`
  :raises [ValueError]: if the arguments shapes do not coincide
  """
  tt_a = unwrap_tt(tt_a)
  tt_b = unwrap_tt(tt_b)

  if not are_shapes_equal(tt_a, tt_b):
    raise ValueError('Types of the arguments or their tensor '
                     'shapes are different, addition is not '
                     'available.')
  if not are_batches_broadcastable(tt_a, tt_b):
    raise ValueError('The batch sizes are different and not 1, '
                     'broadcasting is not available.')

  if tt_a.is_tt_matrix:
    tt_cores = _add_matrix_cores(tt_a, tt_b)
    return TTMatrix(tt_cores)
  else:
    tt_cores = _add_tensor_cores(tt_a, tt_b)
    return TT(tt_cores)


def _add_tensor_cores(tt_a, tt_b):
  """Internal function to be called from add for two TT-tensors.
  Does the actual assembling of the TT-cores to add two TT-tensors.
  """
  num_dims = tt_a.ndim
  shape = tt_a.shape
  a_ranks = tt_a.tt_ranks
  b_ranks = tt_b.tt_ranks
  tt_cores = []
  for core_idx in range(num_dims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    if core_idx == 0:
      curr_core = jnp.concatenate((a_core, b_core), axis=2)
    elif core_idx == num_dims - 1:
      curr_core = jnp.concatenate((a_core, b_core), axis=0)
    else:
      upper_zeros = jnp.zeros((a_ranks[core_idx], shape[core_idx],
                              b_ranks[core_idx + 1]))
      lower_zeros = jnp.zeros((b_ranks[core_idx], shape[core_idx],
                              a_ranks[core_idx + 1]))
      upper = jnp.concatenate((a_core, upper_zeros), axis=2)
      lower = jnp.concatenate((lower_zeros, b_core), axis=2)
      curr_core = jnp.concatenate((upper, lower), axis=0)
    tt_cores.append(curr_core)
  return tt_cores


def _add_matrix_cores(tt_a, tt_b):
  """Internal function to be called from add for two TT-matrices.
  Does the actual assembling of the TT-cores to add two TT-matrices.
  """
  num_dims = tt_a.ndim
  shape = tt_a.raw_tensor_shape
  a_ranks = tt_a.tt_ranks
  b_ranks = tt_b.tt_ranks
  tt_cores = []
  for core_idx in range(num_dims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    if core_idx == 0:
      curr_core = jnp.concatenate((a_core, b_core), axis=3)
    elif core_idx == num_dims - 1:
      curr_core = jnp.concatenate((a_core, b_core), axis=0)
    else:
      upper_zeros = jnp.zeros((a_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], b_ranks[core_idx + 1]))
      lower_zeros = jnp.zeros((b_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], a_ranks[core_idx + 1]))
      upper = jnp.concatenate((a_core, upper_zeros), axis=3)
      lower = jnp.concatenate((lower_zeros, b_core), axis=3)
      curr_core = jnp.concatenate((upper, lower), axis=0)
    tt_cores.append(curr_core)
  return tt_cores


def are_shapes_equal(tt_a, tt_b):
  """Returns the result of equality check of 2 tensors' shapes: 
  `True` if shapes are equal and `False` otherwise.
  The arguments should be both `TT-Tensors` or both `TT-Matrices`.
  The arguments should have the same tensor shape
  but their `TT-ranks` differ.

  :type tt_a: `TT-Tensor` or `TT-Matrix`
  :param tt_a: first argument to check
  :type tt_b: `TT-Tensor` or `TT-Matrix`
  :param tt_b: second argument to check
  :return: `tensor_check` - the result of shape check
  :rtype: bool
  """
  tensor_check = True
  if tt_a.is_tt_matrix != tt_b.is_tt_matrix:
    tensor_check = False
  if jnp.any(jnp.array(tt_a.raw_tensor_shape) != 
            jnp.array(tt_b.raw_tensor_shape)):
    tensor_check = False
  return tensor_check


def are_batches_broadcastable(tt_a, tt_b):
  """Returns the result of compatibility check of 2 tensors' batches: 
  `True` if batches are compatible and `False` otherwise.
  The batch sizes should be equal otherwise at least one of them 
  should equal to 1 for broadcasting to be available.
  
  :type tt_a: `TT-Tensor` or `TT-Matrix`
  :param tt_a: first argument to check
  :type tt_b: `TT-Tensor` or `TT-Matrix`
  :param tt_b: second argument to check
  :return: the result of broadcasting check
  :rtype: bool
  """
  if tt_a.num_batch_dims != tt_b.num_batch_dims:
    return False
  for a, b in zip(tt_a.batch_shape, tt_b.batch_shape):
    if a == 1 or b == 1 or a == b:
      pass
    else:
      return False
  return True


def multiply(a, b):
  """Calculate elementwise product of 2 `TT-Tensors` \ `TT-Matrices` or their product by scalar. Arguments could be wrapped by `WrappedTT` or not.
  
  :param a: first argument
  :type a: Union[float, TT-object]
  :param b: second argument
  :type b: Union[float, TT-object]
  :return: the result of elementwise product
  :rtype: `TT-object`
  """
  if not is_tt_object(a) or not is_tt_object(b):
    return multiply_by_scalar(a, b)
  else:
    return tt_tt_multiply(a, b)


def multiply_by_scalar(a, b):
  """Returns the result of multiplication so called `TT-object` 
  (`TTTensOrMat` or `WrappedTT`) by scalar. Takes 2 arguments 
  as input, one of which is `TT-object` and other is a scalar. 
  Does not depends on arguments order. 
  
  :return: the result of multiplication by scalar
  :rtype: `TTTensOrMat`
  """
  if is_tt_object(a):
    return _mul_by_scalar(a, b)
  else:
    return _mul_by_scalar(b, a)


@tt_vmap(1)
def _mul_by_scalar(tt, c):
  tt = unwrap_tt(tt)
  cores = list(tt.tt_cores)
  cores[0] = c * cores[0]
  if tt.is_tt_matrix:
    return TTMatrix(cores)
  else:
    return TT(cores)
