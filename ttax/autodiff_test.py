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

  def testDoubleProjection(self):
    """Compare P grad f(x) against P grad (<x, stop_grad(P grad f(x))>)."""
    dtype = jnp.float32
    rng = jax.random.PRNGKey(42)
    vector = random_.matrix(rng, ((2, 1, 3, 4), (1, 1, 1, 1)),
                            tt_rank=[1, 2, 4, 3, 1], dtype=dtype)
    matrix = random_.matrix(rng, ((2, 1, 3, 4), (2, 1, 3, 4)),
                            tt_rank=[1, 2, 4, 3, 1], dtype=dtype)

    def f(x):
      mat_vec = ops.matmul(matrix, x)
      return ops.flat_inner(x, mat_vec)

    grad_f = autodiff.grad(f)
    grad = grad_f(vector)

    def f2(x):
      return ops.flat_inner(x, grad)

    double_grad = autodiff.grad(f2)(vector)
    self.assertAllClose(ops.full(double_grad), ops.full(grad), rtol=1e-4)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
