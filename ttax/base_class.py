from typing import List, Union
import numpy as np
import jax.numpy as jnp
import flax


class TTBase:
  """Represents the base for both `TT-Tensor` and `TT-Matrix` (`TT-object`).
  Includes some basic routines and properties.
  """
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

  def __rmul__(self, other):
    # We can't import ops in the beginning since it creates cyclic dependencies.
    from ttax import ops
    return ops.multiply(self, other)

  @property
  def axis_dim(self):
    """Get the position of mode axis in `TT-core`.
    It could differ according to the batch shape.
    
    :return: index
    :rtype: int
    """
    return self.num_batch_dims + 1

  @property
  def batch_shape(self):
    """Get the list representing the shape of the batch of `TT-object`. 
    
    :return: batch shape
    :rtype: list
    """
    return self.tt_cores[0].shape[:self.num_batch_dims]

  @property
  def tt_ranks(self):
    """Get `TT-ranks` of the `TT-object` in amount of ``ndim + 1``.
    The first `TT-rank` and the last one equals to `1`.
    
    :return: `TT-ranks`
    :rtype: list
    """
    ranks = [c.shape[self.num_batch_dims] for c in self.tt_cores]
    ranks.append(self.tt_cores[-1].shape[-1])
    return ranks
  
  @property
  def ndim(self):
    """Get the number of dimensions of the `TT-object`.
    
    :return: dimensions number
    :rtype: int
    """
    return len(self.tt_cores)

  @property
  def dtype(self):
    """Represents the `dtype` of elements in `TT-object`.
    
    :return: `dtype` of elements
    :rtype: dtype
    """
    return self.tt_cores[0].dtype

  @property
  def batch_loc(self):
    """Represents the batch indexing for `TT-object`.
    Wraps `TT-object` by special `BatchIndexing` class
    with overloaded ``__getitem__`` method.
    
    Example:
      ``tt.batch_loc[1, :, :]``
    """
    return BatchIndexing(self)



@flax.struct.dataclass
class TT(TTBase):
  """Represents a `TT-Tensor` object as a list of `TT-cores`.
  
  `TT-Tensor` cores take form (r_l, n, r_r), where
  
  - r_l, r_r are `TT-ranks`
  - n makes `TT-Tensor` shape
  """
  tt_cores: List[jnp.array]

  @property
  def shape(self):
    """Get the tuple representing the shape of `TT-Tensor`. 
    In batch case includes the shape of the batch.
    
    :return: `TT-Tensor` shape with batch shape
    :rtype: tuple
    """
    no_batch_shape = [c.shape[self.axis_dim] for c in self.tt_cores]
    return tuple(list(self.batch_shape) + no_batch_shape)

  @property
  def num_batch_dims(self):
    """Get the number of batch dimensions for batch of `TT-Tensors`.
    
    :return: number of batch dimensions
    :rtype: int
    """
    return len(self.tt_cores[0].shape) - 3

  @property
  def is_tt_matrix(self):
    """Determine whether the object is a `TT-Matrix`.

    :return: `True` if `TT-Matrix`, `False` if `TT-Tensor`
    :rtype: bool
    """
    return False

  @property
  def raw_tensor_shape(self):
    """Get the tuple representing the shape of `TT-Tensor`. 
    In batch case does not include the shape of the batch.
    
    :return: `TT-Tensor` shape
    :rtype: tuple
    """
    return [c.shape[self.axis_dim] for c in self.tt_cores]
  
  def __str__(self):
    """Creates a string describing TT-Tensor.
    :return: TT-Tensor description
    :rtype: string
    """
    if tt.num_batch_dims == 0:
      s = "TT-Tensor of shape {0} and TT-ranks {1}"
      s = s.format(self.shape, self.tt_ranks)
    else:
      s = "Batch of {0} TT-Tensors of shape {1} and TT-ranks {2}"
      s = s.format(self.batch_shape, self.raw_tensor_shape, self.tt_ranks)
    s += " with {0} elements.".format(self.dtype)
    return s
  
  def __getitem__(self, slice_spec):
    """Basic indexing, returns a TT containing the specified element / slice.
    
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
    return TT(new_tt_cores)


@flax.struct.dataclass
class TTMatrix(TTBase):
  """Represents a `TT-Matrix` object as a list of `TT-cores`.
  
  `TT-Matrix` cores take form (r_l, n_l, n_r, r_r), where
  
  - r_l, r_r are `TT-ranks` just as for `TT-Tensor`
  - n_l, n_r make left and right shapes of `TT-Matrix` as rows and cols
  """
  tt_cores: List[jnp.array]

  @property
  def raw_tensor_shape(self):
    """Get the lists representing left and right shapes of `TT-Matrix`. 
    In batch case does not include the shape of the batch.
    
    For example if `TT-Matrix` cores are (1, 2, 3, 5) (5, 6, 7, 1)
    returns (2, 6), (3, 7).
    
    :return: `TT-Matrix` shapes
    :rtype: list, list
    """
    left_shape = [c.shape[self.axis_dim] for c in self.tt_cores]
    right_shape = [c.shape[self.axis_dim + 1] for c in self.tt_cores]
    return left_shape, right_shape

  @property
  def shape(self):
    """Get the tuple representing the shape of underlying dense tensor as matrix. 
    In batch case includes the shape of the batch.
    
    For example if `TT-Matrix` cores are (1, 2, 3, 5) (5, 6, 7, 1)
    it's shape is (12, 21).
    
    :return: `TT-Matrix` shape in dense form with batch shape
    :rtype: tuple
    """
    left_shape, right_shape = self.raw_tensor_shape
    no_batch_shape = [np.prod(left_shape), np.prod(right_shape)]
    return tuple(list(self.batch_shape) + no_batch_shape)

  @property
  def num_batch_dims(self):
    """Get the number of batch dimensions for batch of `TT-Matrices.`
    
    :return: number of batch dimensions
    :rtype: int
    """
    return len(self.tt_cores[0].shape) - 4

  @property
  def is_tt_matrix(self):
    """Determine whether the object is a `TT-Matrix`.

    :return: `True` if `TT-Matrix`, `False` if `TT-Tensor`
    :rtype: bool
    """
    return True
  
  def __str__(self):
    """Creates a string describing TT-Matrix.
    :return: TT-Matrix description
    :rtype: string
    """
    if tt.num_batch_dims == 0:
      s = "TT-Matrix of shape {0} and TT-ranks {1}"
      s = s.format(self.raw_tensor_shape, self.tt_ranks)
    else:
      s = "Batch of {0} TT-Matrices of shape {1} and TT-ranks {2}"
      s = s.format(self.batch_shape, self.raw_tensor_shape, self.tt_ranks)
    s += " with {0} elements.".format(self.dtype)
    return s
  
  def __getitem__(self, slice_spec):
    """Basic indexing, returns a TTMatrix containing the specified element / slice."""
    d = self.ndim
    if len(slice_spec) != 2 * d:
      raise ValueError('Expected %d indices, got %d' % (2 * d, len(slice_spec)))
    for i in range(d):
      if isinstance(slice_spec[i], slice) != isinstance(slice_spec[d+i], slice):
        raise ValueError('Elements i_%d and j_%d should be the same type, '
                         'instead: %s and %s.' % (i, i, slice_spec[i], 
                                                  slice_spec[d+i]))
    new_tt_cores = []
    remainder = None
    for i in range(self.ndim):
      curr_core = self.tt_cores[i]
      sliced_core = curr_core[..., :, slice_spec[i], slice_spec[d+i], :]
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
    self._tt = tt

  def __getitem__(self, indices: list):
    non_none_indices = [idx for idx in indices if idx is not None]
    if len(non_none_indices) > self._tt.num_batch_dims:
      raise ValueError('Expected %d indices, got %d' % (self._tt.num_batch_dims,
                                                        len(non_none_indices)))
    new_cores = []
    for core_idx in range(self._tt.ndim):
      curr_core = self._tt.tt_cores[core_idx]
      new_cores.append(curr_core.__getitem__(indices))

    if self._tt.is_tt_matrix:
      return TTMatrix(new_cores)
    else:
      return TT(new_cores)

TTTensOrMat = Union[TT, TTMatrix]
