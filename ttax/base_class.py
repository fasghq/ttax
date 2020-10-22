import jax.numpy as jnp
from typing import List

@struct.dataclass
class TT():
    factors: List[jnp.array]