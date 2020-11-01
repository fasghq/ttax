import jax.numpy as jnp
from base_class import TT
from compile import compile


def full(tt: TT) -> jnp.array:
  """Converts TT into a regular tensor.
  """
  num_dims = len(tt.tt_cores)

  dtype = tt.tt_cores[0].dtype  # TODO: make tt.dtype.
  res = jnp.ones((1, 1), dtype=dtype)
  for i in range(num_dims):
    curr_core = tt.tt_cores[i]
    right_rank = curr_core.shape[-1]
    res = jnp.einsum('pa,aib->pib', res, curr_core)
    res = res.reshape(-1, right_rank)

  shape = [c.shape[1] for c in tt.tt_cores]
  return jnp.reshape(res, shape)


@compile
def multiply(a, b):
  return {
      'type': 'independent',
      'args': [['a', 'i', 'b'], ['c', 'i', 'd']],
      'res': ['ac', 'i', 'bd']
  }


def flat_inner(a, b):
  @compile
  def main_loop(a, b):
    return {
        'type': 'running',
        'args': [['a', 'i', 'b'], ['c', 'i', 'd'], ['a', 'c']],
        'res': ['b', 'd']
    }
  res = jnp.squeeze(main_loop(a, b)[-1])
  return res
