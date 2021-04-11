import functools
from typing import List
import jax
import jax.numpy as jnp

from ttax.base_class import TT
from ttax.base_class import TTMatrix
from ttax.base_class import TTTensOrMat
from ttax import compile
from ttax.ops import tt_vmap
from ttax.decompositions import orthogonalize


@tt_vmap()
def tangent_to_deltas(tangent_element: TTTensOrMat) -> List[jnp.ndarray]:
  """Convert an element of the tangent space to deltas representation.
  Tangent space elements (outputs of ttax.project) look like:
    dP1 V2 ... Vd + U1 dP2 V3 ... Vd + ... + U1 ... Ud-1 dPd.
  This function takes as input an element of the tangent space and converts
  it to the list of deltas [dP1, ..., dPd].
  Args:
      tangent_element: `TT` or `TTMatrix` that is a result of ttax.project.
  Returns:
      A list of delta-cores.
  """
  # TODO: project on
  ndim = tangent_element.ndim
  tt_ranks = tangent_element.tt_ranks
  deltas = [None] * ndim
  for i in range(1, ndim - 1):
    if int(tt_ranks[i] / 2) != float(tt_ranks[i]) / 2:
      raise ValueError('tangent_element argument is supposed to be a '
                       'projection, but its ranks are not even.')
  for i in range(1, ndim - 1):
    r1, r2 = tt_ranks[i], tt_ranks[i + 1]
    curr_core = tangent_element.tt_cores[i]
    deltas[i] = curr_core[int(r1 / 2):, ..., :int(r2 / 2)]
  deltas[0] = tangent_element.tt_cores[0][..., :int(tt_ranks[1] / 2)]
  deltas[ndim - 1] = tangent_element.tt_cores[ndim - 1][int(tt_ranks[-2] / 2):]
  return deltas


def _deltas_tt_vmap(_deltas_to_tangent):
  """Like tt_vmap, but tailored to be used with deltas_to_tangent."""
  @functools.wraps(_deltas_to_tangent)
  def vectorized_deltas_to_tangent(deltas, tt):
    if tt.is_tt_matrix:
      num_batch_dims = len(deltas[0].shape) - 4
    else:
      num_batch_dims = len(deltas[0].shape) - 3
    if num_batch_dims == 0:
      return _deltas_to_tangent(deltas, tt)
    else:
      # Vmap everything num_batch_dims times.
      vmapped = _deltas_to_tangent
      for _ in range(num_batch_dims):
        vmapped = jax.vmap(vmapped, in_axes=(0, None))
      return vmapped(deltas, tt)
  return vectorized_deltas_to_tangent


@_deltas_tt_vmap
def deltas_to_tangent(deltas: List[jnp.ndarray],
                      tt: TTTensOrMat) -> TTTensOrMat:
  """Converts deltas representation of tangent space vector to TT object.
  Takes as input a list of [dP1, ..., dPd] and returns
    dP1 V2 ... Vd + U1 dP2 V3 ... Vd + ... + U1 ... Ud-1 dPd.
  This function is hard to use correctly because deltas should obey the
  so called gauge conditions. If they don't, the function will silently return
  incorrect result. That is why this function is not imported in __init__.
  Args:
      deltas: a list of deltas (essentially TT-cores) obeying the gauge
        conditions.
      tt: TT (or TTMatrix) object on which the tangent space tensor represented
        by delta is projected.
  Returns:
      TT (or TTMatrix) object constructed from deltas, that is from the tangent
        space at point `tt`.
  """
  cores = []
  dtype = tt.dtype
  left = orthogonalize(tt)
  right = orthogonalize(left, left_to_right=False)
  left_rank_dim = 0
  right_rank_dim = 3 if tt.is_tt_matrix else 2
  for i in range(tt.ndim):
    left_tt_core = left.tt_cores[i]
    right_tt_core = right.tt_cores[i]

    if i == 0:
      tangent_core = jnp.concatenate((deltas[i], left_tt_core),
                                     axis=right_rank_dim)
    elif i == tt.ndim - 1:
      tangent_core = jnp.concatenate((right_tt_core, deltas[i]),
                                     axis=0)
    else:
      rank_1 = right.tt_ranks[i]
      rank_2 = left.tt_ranks[i + 1]
      if tt.is_tt_matrix:
        mode_size_n = tt.raw_tensor_shape[0][i]
        mode_size_m = tt.raw_tensor_shape[1][i]
        shape = [rank_1, mode_size_n, mode_size_m, rank_2]
      else:
        mode_size_n = tt.shape[i]
        shape = [rank_1, mode_size_n, rank_2]
      zeros = jnp.zeros(shape, dtype=dtype)
      upper = jnp.concatenate((right_tt_core, zeros), axis=right_rank_dim)
      lower = jnp.concatenate((deltas[i], left_tt_core), axis=right_rank_dim)
      tangent_core = jnp.concatenate((upper, lower), axis=left_rank_dim)
    cores.append(tangent_core)
  if tt.is_tt_matrix:
    return TTMatrix(cores)
  else:
    return TT(cores)


@tt_vmap()  # TODO: don't need this once fully supported by einsum.
def project(what, where):
  # TODO: Use I_OR_IJ
  projection_rhs_einsum = compile.TTEinsum(
      inputs=[['a', 'i', 'b'], ['c', 'i', 'd'], ['b', 'd']],  output=['a', 'c'],
      order='right-to-left',
      how_to_apply='cumulative'
  )
  projection_rhs = compile.to_function(projection_rhs_einsum)

  projection_lhs_einsum = compile.TTEinsum(
      inputs=[['a', 'i', 'b'], ['c', 'i', 'd'], ['a', 'c']],  output=['b', 'd'],
      how_to_apply='cumulative'
  )
  projection_lhs = compile.to_function(projection_lhs_einsum)

  # project_1_einsum = compile.TTEinsum(
  #     inputs=[['a', 'b'], ['b', 'i', 'c']],  output=['a', 'i', 'c'],
  #     how_to_apply='independent'
  # )
  # project_1 = compile.to_function(project_1_einsum)

  def project_1(a_list, b_list):
    return [jnp.einsum('ab,bic->aic', a, b) for a, b in zip(a_list, b_list)]

  # project_2_einsum = compile.TTEinsum(
  #     inputs=[['a', 'i', 'b'], ['b', 'c']],  output=['a', 'i', 'c'],
  #     how_to_apply='independent'
  # )
  # project_2 = compile.to_function(project_2_einsum)

  dtype = jnp.float32

  def project_2(a_list, b_list):
    return [jnp.einsum('aib,bc->aic', a, b) for a, b in zip(a_list, b_list)]

  left = orthogonalize(where)
  right = orthogonalize(left, left_to_right=False)
  one = jnp.ones((1, 1), dtype=dtype)
  rhs = projection_rhs(what, right) + [one]
  lhs = [one] + projection_lhs(left, what)
  # TODO: we need something like raw_independent_project that would support a
  #  list of tensors instead of actual TT-cores.
  # TODO: fusion with cumulative + independent?
  proj_a = project_1(lhs[:-1], what.tt_cores)
  proj_b = project_2(left.tt_cores[:-1], lhs[1:-1])
  proj_deltas = [a - b for a, b in zip(proj_a, proj_b)]
  proj_deltas = project_2(proj_deltas, rhs[1:-1])
  proj_deltas.append(proj_a[-1])
  # TODO: pass left and right to deltas_to_tangent?
  return deltas_to_tangent(proj_deltas, where)