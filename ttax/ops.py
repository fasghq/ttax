import functools
import jax
import jax.numpy as jnp

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
          in_axis = [0] * num_batch_args + [None] *  num_non_batch_args
        else:
          in_axis = 0
        # Vmap everything num_batch_dims times.
        vmapped = func
        for _ in range(tt_arg.num_batch_dims):
            vmapped = jax.vmap(vmapped, in_axis)
        return vmapped(*args, **kwargs)
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


def add(tt_a, tt_b):
  """Returns a TensorTrain corresponding to elementwise sum tt_a + tt_b.
  The shapes of tt_a and tt_b should coincide.
  Supports broadcasting:
    add(TensorTrainBatch, TensorTrain)
  adds TensorTrain to each element in the batch of TTs in TensorTrainBatch.
  Args:
    tt_a: TT or TT-Matrix
    tt_b: TT or TT-Matrix
  Returns
    TT or TT-Matrix
  Raises
    ValueError if the arguments shapes do not coincide.
  """
  if not are_shapes_equal(tt_a, tt_b):
    raise ValueError('Types of the arguments or their tensor '
                     'shapes are different, addition is not '
                     ' available.')
  if not are_batches_broadcastable(tt_a, tt_b):
    raise ValueError('The batch sizes are different and not 1, '
                     'broadcasting is not available.')
  # batches are not supported yet

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
  shape = tt_a.raw_tensor_shape
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
      upper_zeros = jnp.zeros((a_ranks[core_idx], shape[core_idx],
                              shape[1][core_idx], b_ranks[core_idx + 1]))
      lower_zeros = jnp.zeros((b_ranks[core_idx], shape[core_idx],
                              shape[1][core_idx], a_ranks[core_idx + 1]))
      upper = jnp.concatenate((a_core, upper_zeros), axis=3)
      lower = jnp.concatenate((lower_zeros, b_core), axis=3)
      curr_core = jnp.concatenate((upper, lower), axis=0)
    tt_cores.append(curr_core)
  return tt_cores


def are_shapes_equal(tt_a, tt_b):
  """Returns the result of equality check of 2 tensors' shapes: 
  True if shapes are equal and False otherwise.
  The arguments should be both TT-tensors or both TT-matrices.
  The arguments should have the same tensor shape.
  e.g. for TT core (*, *, ..., *, r_{i}, n_{i}, r_{i+1}):
  axis *, *, ..., * correspond to the batch shape,
  axis r_{i}, n_{i}, r_{i+1} correspond to the tensor shape.
  Args:
    tt_a: TT or TT-Matrix
    tt_b: TT or TT-Matrix
  Returns:
    tensor_check: bool
  """
  tensor_check = True
  if tt_a.is_tt_matrix != tt_b.is_tt_matrix:
    tensor_check = False
  if tt_a.raw_tensor_shape != tt_b.raw_tensor_shape:
    tensor_check = False
  return tensor_check


def are_batches_broadcastable(tt_a, tt_b):
  """Returns the result of compatibility check of 2 tensors' batches: 
  True if batches are compatible and False otherwise.
  The batch sizes should be equal otherwise at least one of them 
  should equal to 1 for broadcasting to be available.
  e.g. for TT core (*, *, ..., *, r_{i}, n_{i}, r_{i+1}):
  axis *, *, ..., * correspond to the batch shape,
  axis r_{i}, n_{i}, r_{i+1} correspond to the tensor shape.
  Args:
    tt_a: TT or TT-Matrix
    tt_b: TT or TT-Matrix
  Returns:
    batch_check: bool
  """
  batch_check = True
  if tt_a.num_batch_dims != tt_b.num_batch_dims:
    batch_check = False
  else:
    for a, b in zip(tt_a.batch_shape, tt_b.batch_shape):
      if a == 1 or b == 1 or a == b:
        pass
      else:
        batch_check = False
  return batch_check
