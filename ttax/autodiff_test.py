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

    project = autodiff.project(matrix @ vector, vector)
    double_project = autodiff.project(project, vector)
    self.assertAllClose(ops.full(project), ops.full(double_project), rtol=1e-4)

  def testHessianVectorProduct(self):
      rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 3)
      dtype = jnp.float32
      shape = (5, 5, 5)
      A = random_.matrix(rng1, (shape, shape), dtype=dtype)
      AT = ops.transpose(A)
      A_plus_AT = A + AT
      x = random_.tensor(rng2, shape, dtype=dtype)
      vec = random_.tensor(rng3, shape, dtype=dtype)
      proj_vec = autodiff.project(vec, x)

      func = lambda x: ops.flat_inner(x, ops.matrix_tensor_matmul(A, x))
      desired = autodiff.project(ops.matrix_tensor_matmul(A_plus_AT, proj_vec), x)
      desired = ops.full(desired)
      actual = ops.full(autodiff.hessian_vector_product(func)(x, vec))
      self.assertAllClose(desired, actual, rtol=1e-4)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
