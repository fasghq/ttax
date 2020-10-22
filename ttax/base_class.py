from typing import List
import jax.numpy as jnp
import flax


@flax.struct.dataclass
class TT():
  factors: List[jnp.array]