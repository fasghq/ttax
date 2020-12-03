from typing import List
import jax.numpy as jnp
import flax


@flax.struct.dataclass
class TT:
  tt_cores: List[jnp.array]
  
  def __mul__(self, other):
    # We can't import ops in the beginning since it creates cyclic dependencies.
    from ttax import ops
    return ops.multiply(self, other)