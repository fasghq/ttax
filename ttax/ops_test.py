from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax
import jax.numpy as jnp
import jax.test_util as jtu

from jax.config import config

from base_class import TT
# import random
import ops

config.parse_flags_with_absl()


class TTTensorTest(jtu.JaxTestCase):

  def testFullTensor2d(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(10, rank)
      b = np.random.rand(rank, 9)
      tt_cores = (a.reshape(1, 10, rank), b.reshape(rank, 9, 1))
      desired = np.dot(a, b)
      tf_tens = TT(tt_cores)
      actual = ops.full(tf_tens)
      self.assertAllClose(desired, actual)

  def testMultiply(self):
    # Multiply two TT-tensors.
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    tt_a = random_.tensor(rng1, (1, 2, 3, 4), tt_rank=2, dtype=dtype)
    tt_b = random_.tensor(rng2, (1, 2, 3, 4), tt_rank=[1, 1, 4, 3, 1], dtype=dtype)
    res_actual = ops.full(ops.multiply(tt_a, tt_b))
    res_desired = ops.full(tt_a) * ops.full(tt_b)
    self.assertAllClose(res_actual, res_desired)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())