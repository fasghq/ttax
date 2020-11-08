from typing import List
import jax.numpy as jnp
import flax
from ttax import ops


@flax.struct.dataclass
class TT:
  tt_cores: List[jnp.array]
  
  def __mul__(self, other):
    return ops.multiply(self, other)