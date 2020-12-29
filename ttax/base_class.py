from typing import List
import jax.numpy as jnp
import flax


class TTBase:

  def __mul__(self, other):
    # We can't import ops in the beginning since it creates cyclic dependencies.
    from ttax import ops
    return ops.multiply(self, other)

  @property
  def axis_dim(self):
    return self.num_batch_dims + 1

  @property
  def batch_shape(self):
    return self.tt_cores[0].shape[:self.num_batch_dims]


@flax.struct.dataclass
class TT(TTBase):
  tt_cores: List[jnp.array]

  @property
  def shape(self):
    no_batch_shape = [c.shape[self.axis_dim] for c in self.tt_cores]
    return tuple(list(self.batch_shape) + no_batch_shape)

  @property
  def num_batch_dims(self):
    return len(self.tt_cores[0].shape) - 3


@flax.struct.dataclass
class TTMatrix(TTBase):
  tt_cores: List[jnp.array]

  @property
  def num_batch_dims(self):
    return len(self.tt_cores[0].shape) - 4