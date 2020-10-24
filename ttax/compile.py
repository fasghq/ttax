"""Utils for compiling functiones defined as einsum strings.

Here we use the notion of *tt_einsum*, which is similar to einsum strings, but
with more structure.

Basic element of tt_einsum is a *tt_einsum_core*: list with three elements,
which defines an einsum string for a single TT-core. First element is indices
for the left TT-rank, second element is the indices for the main dimensions of
the resulting TT-core, and the last element is the indices for the right
TT-rank.

TT_einsum consists of list of input and output cores defined like with
tt_einsum_cores.
"""

import opt_einsum as oe
import numpy as np
import jax.numpy as jnp
from base_class import TT




def tt_to_vanilla_einsum(explained_einsum):
  """Converting from tt_einsum to a regular einsum."""
  args = []
  for arg in explained_einsum['args']:
    args.append(''.join(arg))
  return ','.join(args) + '->' + ''.join(explained_einsum['res'])


def compile(func):
  """Decorator converting tt einsum definition into a function.
  
  Example:
  @compile
  def multiply(a, b):
    return {'args': [['a', 'i', 'b'], ['c', 'i', 'd']], 'res': ['ac', 'i', 'bd']}
  """
  def new_func(a, b):
    explained_einsum = func(a, b)
    einsum = tt_to_vanilla_einsum(explained_einsum)
    res_cores = [oe.contract(einsum, ca, cb, backend='jax') for ca, cb in zip(a.tt_cores, b.tt_cores)]
    new_res_cores = []
    for core in res_cores:
      shape = core.shape
      new_shape = []
      num_left_rank_dims = len(explained_einsum['res'][0])
      num_tensor_dims = len(explained_einsum['res'][1])
      split_points = (num_left_rank_dims, num_left_rank_dims + num_tensor_dims)
      new_shape = np.split(shape, split_points)
      new_shape = [np.prod(s) for s in new_shape]
      new_res_cores.append(core.reshape(new_shape))
    res = TT(new_res_cores)
    return res
  return new_func
