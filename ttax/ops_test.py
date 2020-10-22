from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax.numpy as jnp
import jax.test_util as jtu

from jax.config import config

from base_class import TT
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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())