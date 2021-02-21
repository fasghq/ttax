from typing import List
import numpy as np
import jax.numpy as jnp
import flax


class TTBase:

  def __mul__(self, other):
    # We can't import ops in the beginning since it creates cyclic dependencies.
    from ttax import ops
    return ops.multiply(self, other)

  def __matmul__(self, other):
    # We can't import ops in the beginning since it creates cyclic dependencies.
    from ttax import ops
    return ops.matmul(self, other)
  
  def __add__(self, other):
    # We can't import ops in the beginning since it creates cyclic dependencies.
    from ttax import ops
    return ops.add(self, other)

  @property
  def axis_dim(self):
    return self.num_batch_dims + 1

  @property
  def batch_shape(self):
    return self.tt_cores[0].shape[:self.num_batch_dims]

  @property
  def tt_ranks(self):
    ranks = [c.shape[self.num_batch_dims] for c in self.tt_cores]
    ranks.append(self.tt_cores[-1].shape[-1])
    return ranks
  
  @property
  def ndim(self):
    return len(self.tt_cores)

  @property
  def batch_loc(self):
    return BatchIndexing(self)


@flax.struct.dataclass
class TT(TTBase):
  tt_cores: List[jnp.array]

  @property
  def shape(self):
    no_batch_shape = [c.shape[self.axis_dim] for c in self.tt_cores]
    return tuple(list(self.batch_shape) + no_batch_shape)

  @property
  def num_batch_dims(self):
    return len(self.tt_cores[0].shape) - 3

  @property
  def is_tt_matrix(self):
    return False

  @property
  def raw_tensor_shape(self):
    return [c.shape[self.axis_dim] for c in self.tt_cores]
  
  def __getitem__(self, slice_spec):
    """Basic indexing, returns a TT containing the specified region.
    Examples:
      >>> a = ttax.random.tensor(rng, [2, 3, 4])
      >>> a[1, :, :]
      is a 2D TensorTrain 3 x 4.
      >>> a[1:2, :, :]
      is a 3D TensorTrain 1 x 3 x 4
    """
    if len(slice_spec) != self.ndim:
      raise ValueError('Expected %d indices, got %d' % (self.ndim,
                                                        len(slice_spec)))
    new_tt_cores = []
    remainder = None
    for i in range(self.ndim):
      curr_core = self.tt_cores[i]
      sliced_core = curr_core[..., :, slice_spec[i], :]
      if len(curr_core.shape) != len(sliced_core.shape):
        # This index is specified exactly and we want to collapse this axis.
        if remainder is None:
          remainder = sliced_core
        else:
          remainder = jnp.matmul(remainder, sliced_core)
      else:
        if remainder is not None:
          # Add reminder from the previous collapsed cores to the current core.
          sliced_core = jnp.einsum('...ab,...bid->...aid', 
                                   remainder, sliced_core)
          remainder = None
        new_tt_cores.append(sliced_core)

    if remainder is not None:
      # The reminder obtained from collapsing the last cores.
      new_tt_cores[-1] = jnp.einsum('...aib,...bd->...aid', 
                                    new_tt_cores[-1], remainder)
      remainder = None
    # TODO: infer the output ranks and shape.
    return TT(new_tt_cores)


@flax.struct.dataclass
class TTMatrix(TTBase):
  tt_cores: List[jnp.array]

  @property
  def raw_tensor_shape(self):
    left_shape = [c.shape[self.axis_dim] for c in self.tt_cores]
    right_shape = [c.shape[self.axis_dim + 1] for c in self.tt_cores]
    return left_shape, right_shape

  @property
  def shape(self):
    left_shape, right_shape = self.raw_tensor_shape
    no_batch_shape = [np.prod(left_shape), np.prod(right_shape)]
    return tuple(list(self.batch_shape) + no_batch_shape)

  @property
  def num_batch_dims(self):
    return len(self.tt_cores[0].shape) - 4

  @property
  def is_tt_matrix(self):
    return True
  
  def __getitem__(self, slice_spec):

    n = self.ndim
    if len(slice_spec) != 2 * n:
      raise ValueError('Expected %d indices, got %d' % (2 * n, len(slice_spec)))
    for i in range(n):
      if isinstance(slice_spec[i], slice) != isinstance(slice_spec[n+i], slice):
        raise ValueError('Elements i_%d and j_%d should be the same type.' % (i,
                                                                            i))
    new_tt_cores = []
    remainder = None
    for i in range(self.ndim):
      curr_core = self.tt_cores[i]
      sliced_core = curr_core[..., :, slice_spec[i], slice_spec[n+i], :]
      if len(curr_core.shape) != len(sliced_core.shape):
        # These indices are specified exactly and we want to collapse this axis.
        if remainder is None:
          remainder = sliced_core
        else:
          remainder = jnp.matmul(remainder, sliced_core)
      else:
        if remainder is not None:
          # Add reminder from the previous collapsed cores to the current core.
          sliced_core = jnp.einsum('...ab,...bijd->...aijd', 
                                   remainder, sliced_core)
          remainder = None
        new_tt_cores.append(sliced_core)

    if remainder is not None:
      # The reminder obtained from collapsing the last cores.
      new_tt_cores[-1] = jnp.einsum('...aijb,...bd->...aijd', 
                                    new_tt_cores[-1], remainder)
      remainder = None

    return TTMatrix(new_tt_cores)


class BatchIndexing:
  def __init__(self, tt):
    self.tt = tt

  def __getitem__(self, indices: list):
    non_none_indices = [idx for idx in indices if idx is not None]
    if len(non_none_indices) > self.tt.tt_num_batch_dims:
      raise ValueError('Expected %d indices, got %d' % (self.tt.num_batch_dims,
                                                        len(non_none_indices)))
    new_cores = []
    for core_idx in range(self.tt.ndim):
      curr_core = self.tt.tt_cores[core_idx]
      new_cores.append(curr_core.__getitem__(indices))

    if self.tt.is_tt_matrix:
      return TTMatrix(new_cores)
    else:
      return TT(new_cores)
