import numpy as np
import jax
import jax.numpy as jnp

from ttax.base_class import TT
from ttax.base_class import TTMatrix
from ttax.ops import tt_vmap


@tt_vmap(1)
def round(tt, max_tt_rank=None, epsilon=None):
  """Tensor Train rounding procedure, returns a `TT-object` with smaller `TT-ranks`.

  :param tt: argument which ranks would be reduced
  :type tt: `TT-Tensor` or `TT-Matrix`
  :type max_tt_rank: int or list of ints
  :param max_tt_rank: 
  
    - If a number, than defines the maximal `TT-rank` of the result. 
    - If a list of numbers, than `max_tt_rank` length should be d+1 
      (where d is the number of dimensions) and `max_tt_rank[i]`
      defines the maximal (i+1)-th `TT-rank` of the result.
      
      The following two versions are equivalent
      
      - ``max_tt_rank = r``
      
      - ``max_tt_rank = [1] + [r] * (d-1) + [1]``  
       
  :type epsilon: float or None
  :param epsilon:
  
    - If the `TT-ranks` are not restricted (`max_tt_rank=None`), then
      the result would be guarantied to be `epsilon`-close to `tt`
      in terms of relative Frobenius error:
      
        `||res - tt||_F / ||tt||_F <= epsilon`
        
    - If the `TT-ranks` are restricted, providing a loose `epsilon` may
      reduce the `TT-ranks` of the result. E.g.
      
        ``round(tt, max_tt_rank=100, epsilon=0.9)``
        
      will probably return you a `TT-Tensor` with `TT-ranks` close to 1, not 100.
      Note that providing a nontrivial (= not equal to `None`) epsilon will make
      the `TT-ranks` of the result change depending on the data, 
      which will prevent you from using ``jax.jit``
      for speeding up the computations.
  :return: `TT-object` with reduced `TT-ranks`
  :rtype: `TT-Tensor` or `TT-Matrix`
  :raises:
    ValueError if `max_tt_rank` is less than 0, if `max_tt_rank` is not a number and
    not a vector of length d + 1 where d is the number of dimensions of
    the input tensor, if `epsilon` is less than 0.
  """

  if max_tt_rank is None:
    max_tt_rank = np.iinfo(np.int32).max
  num_dims = tt.ndim
  max_tt_rank = np.array(max_tt_rank).astype(np.int32)
  if np.any(max_tt_rank < 1):
    raise ValueError('Maximum TT-rank should be greater or equal to 1.')
  if epsilon is not None:
    raise NotImplementedError('Epsilon is not supported yet.')
  if max_tt_rank.size == 1:
    max_tt_rank = (max_tt_rank * np.ones(num_dims + 1)).astype(jnp.int32)
  elif max_tt_rank.size != num_dims + 1:
    raise ValueError('max_tt_rank should be a number or a vector of size (d+1) '
                     'where d is the number of dimensions of the tensor.')
  if tt.is_tt_matrix:
    raw_shape = tt.raw_tensor_shape
  else:
    raw_shape = tt.shape

  tt_cores = orthogonalize(tt).tt_cores
  # Copy cores references so we can change the cores.
  tt_cores = list(tt_cores)

  ranks = [1] * (num_dims + 1)

  # Right to left SVD compression.
  for core_idx in range(num_dims - 1, 0, -1):
    curr_core = tt_cores[core_idx]
    if tt.is_tt_matrix:
      curr_mode_left = raw_shape[0][core_idx]
      curr_mode_right = raw_shape[1][core_idx]
      curr_mode = curr_mode_left * curr_mode_right
    else:
      curr_mode = raw_shape[core_idx]
    
    columns = curr_mode * ranks[core_idx + 1]
    curr_core = jnp.reshape(curr_core, [-1, columns])
    rows = curr_core.shape[0]
    if max_tt_rank[core_idx] == 1:
      ranks[core_idx] = 1
    else:
      ranks[core_idx] = min(max_tt_rank[core_idx], rows, columns)
    u, s, v = jnp.linalg.svd(curr_core, full_matrices=False)
    u = u[:, 0:ranks[core_idx]]
    s = s[0:ranks[core_idx]]
    v = v[0:ranks[core_idx], :]
    if tt.is_tt_matrix:
      core_shape = (ranks[core_idx], curr_mode_left, curr_mode_right,
                    ranks[core_idx + 1])
    else:
      core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
    tt_cores[core_idx] = jnp.reshape(v, core_shape)
    prev_core_shape = (-1, rows)
    tt_cores[core_idx - 1] = jnp.reshape(tt_cores[core_idx - 1], prev_core_shape)
    tt_cores[core_idx - 1] = jnp.matmul(tt_cores[core_idx - 1], u)
    tt_cores[core_idx - 1] = jnp.matmul(tt_cores[core_idx - 1], jnp.diag(s))

  if tt.is_tt_matrix:
    core_shape = (ranks[0], raw_shape[0][0], raw_shape[1][0], ranks[1])
  else:
    core_shape = (ranks[0], raw_shape[0], ranks[1])
  tt_cores[0] = jnp.reshape(tt_cores[0], core_shape)

  if tt.is_tt_matrix:
    return TTMatrix(tt_cores)
  else:
    return TT(tt_cores)


def orthogonalize(tt, left_to_right=True):
  """Orthogonalize `TT-cores` of a `TT-object`.
  
  :type tt: `TT-Tensor` or `TT-Matrix`
  :param tt: `TT-object` which `TT-cores` would be orthogonalized
  :param left_to_right: the direction of orthogonalization, `True` for left to right and `False` for right to left
  :type left_to_right: bool
  :return: `TT-object` with orthogonalized `TT-cores`
  :rtype: `TT-Tensor` or `TT-Matrix`
  """
  if left_to_right:
    return _orthogonalize_tt_cores_left_to_right(tt)
  else:
    return _orthogonalize_tt_cores_right_to_left(tt)


def _orthogonalize_tt_cores_left_to_right(tt):
  """Orthogonalize `TT-cores` of a `TT-object`.

  :type tt: `TT-tensor` or `TT-matrix`
  :param tt: `TT-object` which `TT-cores` would be orthogonalized
  :rtype: `TT-tensor` or `TT-matrix`
  :return: `TT-object` with orthogonalized `TT-cores`
  """

  # Left to right orthogonalization.
  num_dims = tt.ndim
  if tt.is_tt_matrix:
    raw_shape = tt.raw_tensor_shape
  else:
    raw_shape = tt.shape

  tt_ranks = tt.tt_ranks
  next_rank = tt_ranks[0]
  # Copy cores references so we can change the cores.
  tt_cores = list(tt.tt_cores)
  for core_idx in range(num_dims - 1):
    curr_core = tt_cores[core_idx]
    # TT-ranks could have changed on the previous iteration, so `tt_ranks` can
    # be outdated for the current TT-rank, but should be valid for the next
    # TT-rank.
    curr_rank = next_rank
    next_rank = tt_ranks[core_idx + 1]
    if tt.is_tt_matrix:
      curr_mode_left = raw_shape[0][core_idx]
      curr_mode_right = raw_shape[1][core_idx]
      curr_mode = curr_mode_left * curr_mode_right
    else:
      curr_mode = raw_shape[core_idx]
    
    qr_shape = (curr_rank * curr_mode, next_rank)
    curr_core = jnp.reshape(curr_core, qr_shape)
    curr_core, triang = jnp.linalg.qr(curr_core)
    
    triang_shape = triang.shape

    # The TT-rank could have changed: if qr_shape is e.g. 4 x 10, than q would
    # be of size 4 x 4 and r would be 4 x 10, which means that the next rank
    # should be changed to 4.
    next_rank = triang_shape[0]
    if tt.is_tt_matrix:
      new_core_shape = (curr_rank, curr_mode_left, curr_mode_right, next_rank)
    else:
      new_core_shape = (curr_rank, curr_mode, next_rank)
    tt_cores[core_idx] = jnp.reshape(curr_core, new_core_shape)

    next_core = jnp.reshape(tt_cores[core_idx + 1], (triang_shape[1], -1))
    tt_cores[core_idx + 1] = jnp.matmul(triang, next_core)

  if tt.is_tt_matrix:
    last_core_shape = (next_rank, raw_shape[0][-1], raw_shape[1][-1], 1)
  else:
    last_core_shape = (next_rank, raw_shape[-1], 1)
  tt_cores[-1] = jnp.reshape(tt_cores[-1], last_core_shape)

  if tt.is_tt_matrix:
    return TTMatrix(tt_cores)
  else:
    return TT(tt_cores)


def _orthogonalize_tt_cores_right_to_left(tt):
  """Orthogonalize `TT-cores` of a `TT-object`.
  
  :type tt: `TT-tensor` or `TT-matrix`
  :param tt: `TT-object` which `TT-cores` would be orthogonalized
  :rtype: `TT-tensor` or `TT-matrix`
  :return: `TT-object` with orthogonalized `TT-cores`
  """
  
  # Right to left orthogonalization.
  num_dims = tt.ndim
  if tt.is_tt_matrix:
    raw_shape = tt.raw_tensor_shape
  else:
    raw_shape = tt.shape

  tt_ranks = tt.tt_ranks
  prev_rank = tt_ranks[num_dims]
  # Copy cores references so we can change the cores.
  tt_cores = list(tt.tt_cores)
  for core_idx in range(num_dims - 1, 0, -1):
    curr_core = tt_cores[core_idx]
    # TT-ranks could have changed on the previous iteration, so `tt_ranks` can
    # be outdated for the current TT-rank, but should be valid for the next
    # TT-rank.
    curr_rank = prev_rank
    prev_rank = tt_ranks[core_idx]
    if tt.is_tt_matrix:
      curr_mode_left = raw_shape[0][core_idx]
      curr_mode_right = raw_shape[1][core_idx]
      curr_mode = curr_mode_left * curr_mode_right
    else:
      curr_mode = raw_shape[core_idx]
    
    qr_shape = (prev_rank, curr_mode * curr_rank)
    curr_core = jnp.reshape(curr_core, qr_shape)
    curr_core, triang = jnp.linalg.qr(curr_core.T)
    curr_core = curr_core.T
    triang = triang.T
    triang_shape = triang.shape

    # The TT-rank could have changed: if qr_shape is e.g. 4 x 10, than q would
    # be of size 4 x 4 and r would be 4 x 10, which means that the next rank
    # should be changed to 4.
    prev_rank = triang_shape[1]
    if tt.is_tt_matrix:
      new_core_shape = (prev_rank, curr_mode_left, curr_mode_right, curr_rank)
    else:
      new_core_shape = (prev_rank, curr_mode, curr_rank)
    tt_cores[core_idx] = jnp.reshape(curr_core, new_core_shape)

    prev_core = jnp.reshape(tt_cores[core_idx - 1], (-1, triang_shape[0]))
    tt_cores[core_idx - 1] = jnp.matmul(prev_core, triang)

  if tt.is_tt_matrix:
    first_core_shape = (1, raw_shape[0][0], raw_shape[1][0], prev_rank)
  else:
    first_core_shape = (1, raw_shape[0], prev_rank)
  tt_cores[0] = jnp.reshape(tt_cores[0], first_core_shape)
  
  if tt.is_tt_matrix:
    return TTMatrix(tt_cores)
  else:
    return TT(tt_cores)
  
def to_tt_tensor(tens, max_tt_rank=10, epsilon=None):
  """Converts a given tensor to a `TT-tensor` of the same shape.
  
  :type tt: `TT-Tensor` or `TT-Matrix`
  :param tt: `TT-object` which `TT-cores` would be orthogonalized
  :param left_to_right: the direction of orthogonalization, `True` for left to right and `False` for right to left
  :type left_to_right: bool
  :return: `TT-object` with orthogonalized `TT-cores`
  :rtype: `TT-Tensor` or `TT-Matrix`
  
  :param tens: Tensor to convert to `TT-tensor`
  :type max_tt_rank: a number or a list of numbers
  :param max_tt_rank: 
      
    - If a number, than defines the maximal `TT-rank` of the result.
    
    - If a list of numbers, than `max_tt_rank` length should be d+1
      (where d is the rank of `tens`) and `max_tt_rank[i]` defines
      the maximal (i+1)-th `TT-rank` of the result.
      The following two versions are equivalent
      
        - ``max_tt_rank = r``
        
      and
      
        - ``max_tt_rank = r * np.ones(d-1)``
        
  :type epsilon: a floating point number or None
  :param epsilon:
  
    - If the `TT-ranks` are not restricted (`max_tt_rank=np.inf`), then
      the result would be guarantied to be `epsilon` close to `tens`
      in terms of relative Frobenius error:
      
        ``||res - tens||_F / ||tens||_F <= epsilon``
        
    - If the `TT-ranks` are restricted, providing a loose `epsilon` may
      reduce the TT-ranks of the result.
      E.g.
      
        ``to_tt_tensor(tens, max_tt_rank=100, epsilon=0.9)``
        
      will probably return you a `TT-tensor` with `TT-ranks` close to 1, not 100.
      
  :rtype: `TT`
  :return: `TT-object` containing a `TT-tensor`
  Raises:
    ValueError if the rank (number of dimensions) of the input tensor is
    not defined, if `max_tt_rank` is less than 0, if `max_tt_rank` is not a number
    and not a vector of length d + 1 where d is the number of dimensions (rank)
    of the input tensor, if `epsilon` is less than 0.
  """
  shape = tens.shape
  d = len(tens.shape)
  max_tt_rank = np.array(max_tt_rank).astype(np.int32)
  if np.any(max_tt_rank < 1):
    raise ValueError('Maximum TT-rank should be greater or equal to 1.')
  if epsilon is not None and epsilon < 0:
    raise ValueError('Epsilon should be non-negative.')
  if max_tt_rank.size == 1:
    max_tt_rank = (max_tt_rank * np.ones(d+1)).astype(np.int32)
  elif max_tt_rank.size != d + 1:
    raise ValueError('max_tt_rank should be a number or a vector of size '
                     '(d+1) where d is the number of dimensions (rank) of '
                     'the tensor.')
  ranks = [1] * (d + 1)
  tt_cores = []
  for core_idx in range(d - 1):
    curr_mode = shape[core_idx]
    rows = ranks[core_idx] * curr_mode
    tens = jnp.reshape(tens, [rows, -1])
    columns = tens.shape[1]
    u, s, v = jnp.linalg.svd(tens, full_matrices=False)
    if max_tt_rank[core_idx + 1] == 1:
      ranks[core_idx + 1] = 1
    else:
      ranks[core_idx + 1] = min(max_tt_rank[core_idx + 1], rows, columns)
    u = u[:, 0:ranks[core_idx + 1]]
    s = s[0:ranks[core_idx + 1]]
    v = v[0:ranks[core_idx + 1], :]
    core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
    tt_cores.append(jnp.reshape(u, core_shape))
    tens = jnp.matmul(jnp.diag(s), v)
  last_mode = shape[-1]
  core_shape = (ranks[d - 1], last_mode, ranks[d])
  tt_cores.append(jnp.reshape(tens, core_shape))
  return TT(tt_cores)


def to_tt_matrix(mat, shape, max_tt_rank=10, epsilon=None):
  """Converts a given matrix or vector to a `TT-matrix`.
  The matrix dimensions should factorize into d numbers.
  If e.g. the dimensions are prime numbers, it's usually better to
  pad the matrix with zeros until the dimensions factorize into
  (ideally) 3-8 numbers.

  :param mat: two dimensional tensor (a matrix).
  :type shape: two dimensional array (np.array or list of lists)
  :param shape: Represents the tensor shape of the matrix.
      
    - E.g. for a (a1 * a2 * a3) x (b1 * b2 * b3) matrix `shape` should be
      ((a1, a2, a3), (b1, b2, b3))
      
    - `shape[0]`` and `shape[1]`` should have the same length.
    
    -  For vectors you may use ((a1, a2, a3), (1, 1, 1)) or, equivalently,
       ((a1, a2, a3), None)
  
  :type max_tt_rank: a number or a list of numbers
  :param max_tt_rank:
  
    - If a number, than defines the maximal `TT-rank` of the result.
      
    - If a list of numbers, than `max_tt_rank` length should be d+1
      (where d is the length of `shape[0]`) and `max_tt_rank[i]` defines
      the maximal (i+1)-th `TT-rank` of the result.
      The following two versions are equivalent
      
        - ``max_tt_rank = r``
      
      and
        
        - ``max_tt_rank = r * np.ones(d-1)``
        
  :type epsilon: a floating point number or None
  :param epsilon:
  
    - If the `TT-ranks` are not restricted (`max_tt_rank=np.inf`), then
      the result would be guarantied to be `epsilon` close to `mat`
      in terms of relative Frobenius error:
      
        ``||res - mat||_F / ||mat||_F <= epsilon``
        
    - If the `TT-ranks` are restricted, providing a loose `epsilon` may reduce
      the `TT-ranks` of the result.
      E.g.
      
        ``to_tt_matrix(mat, shape, max_tt_rank=100, epsilon=0.9)``
      
      will probably return you a `TT-matrix` with `TT-ranks` close to 1, not 100.
     
  :rtype: `TTMatrix`
  :return: `TT-object` containing a `TT-matrix`
  Raises:
    ValueError if `max_tt_rank` is less than 0, if `max_tt_rank` is not a number and
    not a vector of length d + 1 where d is the number of dimensions (rank) of
    the input tensor, if `epsilon` is less than 0.
  """
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1])).astype(int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0])).astype(int)
    
  shape = np.array(shape)
  tens = jnp.reshape(mat, shape.flatten())
  d = len(shape[0])
  # transpose_idx = 0, d, 1, d+1 ...
  transpose_idx = np.arange(2 * d).reshape(2, d).T.flatten()
  transpose_idx = transpose_idx.astype(int)
  tens = jnp.transpose(tens, transpose_idx)
  new_shape = jnp.prod(shape, axis=0)
  tens = jnp.reshape(tens, new_shape)
  tt_tens = to_tt_tensor(tens, max_tt_rank, epsilon)
  tt_cores = []
  for core_idx in range(d):
    curr_core = tt_tens.tt_cores[core_idx]
    curr_rank = tt_tens.tt_ranks[core_idx]
    next_rank = tt_tens.tt_ranks[core_idx + 1]
    curr_core_new_shape = (curr_rank, shape[0, core_idx],
                             shape[1, core_idx], next_rank)
    curr_core = jnp.reshape(curr_core, curr_core_new_shape)
    tt_cores.append(curr_core)
  return TTMatrix(tt_cores)
