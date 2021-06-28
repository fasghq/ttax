from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax
import jax.numpy as jnp
import jax.test_util as jtu
from jax.config import config

from ttax.base_class import TT
from ttax.base_class import TTMatrix
from ttax import random_
from ttax import ops
from ttax import autodiff

config.parse_flags_with_absl()


class AutodiffTest(jtu.JaxTestCase):

  def testNormGrad(self):
    dtype = jnp.float32
    rng = jax.random.PRNGKey(42)
    tensor = random_.tensor(rng, (2, 1, 3, 4), tt_rank=[1, 2, 4, 3, 1],
                            dtype=dtype)
    def f(x):
      return 0.5 * ops.flat_inner(x, x)

    grad = autodiff.grad(f)
    actual = grad(tensor)
    desired = tensor
    self.assertAllClose(ops.full(actual), ops.full(desired), rtol=1e-4)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
