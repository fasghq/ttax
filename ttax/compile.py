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


class WrappedTT:
  def __init__(self, tt: TT, inputs=None, tt_einsum=None):
    self.tt = tt
    self.inputs = inputs
    self.tt_einsum = tt_einsum

def tt_to_vanilla_einsum(tt_einsum):
  """Converting from tt_einsum to a regular einsum."""
  args = []
  for arg in tt_einsum['args']:
    args.append(''.join(arg))
  return ','.join(args) + '->' + ''.join(tt_einsum['res'])


def compile(func):
  """Decorator converting tt einsum definition into a function.
  
  Example:
  @compile
  def multiply(a, b):
    return {
             'type': 'independent',
             'args': [['a', 'i', 'b'], ['c', 'i', 'd']],
             'res': ['ac', 'i', 'bd']
           }
  """
  tt_einsum = func(None, None)  # TODO: infer desired number of arguments.
  if tt_einsum['type'] == 'independent':
    return compile_independent(tt_einsum)
  elif tt_einsum['type'] == 'running':
    return compile_running(tt_einsum)
  else:
    raise ValueError('Unsupported tt_einsum type "%s"' % tt_einsum['type'])


def compile_independent(tt_einsum):
  einsum = tt_to_vanilla_einsum(tt_einsum)
  def new_func(*args):
    # TODO: do in parallel w.r.t. cores.
    # TODO: use optimal einsum.
    res_cores = []
    for i in range(len(args[0].tt_cores)):
      curr_input_cores = [tt.tt_cores[i] for tt in args]
      core = oe.contract(einsum, *curr_input_cores, backend='jax')
      shape = core.shape
      new_shape = []
      num_left_rank_dims = len(tt_einsum['res'][0])
      num_tensor_dims = len(tt_einsum['res'][1])
      split_points = (num_left_rank_dims, num_left_rank_dims + num_tensor_dims)
      new_shape = np.split(shape, split_points)
      new_shape = [np.prod(s) for s in new_shape]
      res_cores.append(core.reshape(new_shape))
    res = TT(res_cores)
    return res
  return new_func


def compile_running(tt_einsum):
  einsum = tt_to_vanilla_einsum(tt_einsum)
  def new_func(*args):
    res = tt_einsum['init'](args[0].tt_cores[0].dtype)
    res_list = []
    for core_idx in range(len(args[0].tt_cores)):
      curr_tensors = [a.tt_cores[core_idx] for a in args]
      curr_tensors.append(res)
      # TODO: use optimal einsum.
      res = oe.contract(einsum, *curr_tensors, backend='jax')
      res_list.append(res)
    return res_list
  return new_func


def fuse(func):
  def _func(*args):
    wrapped_args = [WrappedTT(arg) for arg in args]
    res = func(*wrapped_args)
    if isinstance(res, WrappedTT):
      res = res.tt
    return res
  return _func
