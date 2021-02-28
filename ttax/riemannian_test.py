from absl.testing import absltest

import numpy as np
import jax
import jax.numpy as jnp
import jax.test_util as jtu
from jax.config import config

from ttax import random_
from ttax import riemannian

config.parse_flags_with_absl()


class TTTensorTest(jtu.JaxTestCase):

  def testToAndFromDeltas(self):
    np.random.seed(1)
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    tt = random_.tensor(rng1, (10, 10, 10), tt_rank=10, dtype=dtype)
    # TODO: it's incorrect in general to use random deltas (gauge conditions)!
    deltas = [np.random.randn(*c.shape).astype(dtype) for c in tt.tt_cores]
    tangent_element = riemannian.deltas_to_tangent(deltas, tt)
    back_translated_deltas = riemannian.tangent_to_deltas(tangent_element)
    for desired_d, actual_d in zip(deltas, back_translated_deltas):
      self.assertAllClose(desired_d, actual_d)

  def testToAndFromDeltasBatch(self):
    np.random.seed(1)
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    tt = random_.tensor(rng1, (10, 10, 10), tt_rank=10, dtype=dtype)
    # TODO: it's incorrect in general to use random deltas (gauge conditions)!
    deltas = [np.random.randn(7, *c.shape).astype(dtype) for c in tt.tt_cores]
    tangent_element = riemannian.deltas_to_tangent(deltas, tt)
    back_translated_deltas = riemannian.tangent_to_deltas(tangent_element)
    for desired_d, actual_d in zip(deltas, back_translated_deltas):
      self.assertAllClose(desired_d, actual_d)

  def testProject(self):
    np.random.seed(1)
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    what = random_.tensor(rng1, (10, 10, 10), tt_rank=10, dtype=dtype)
    where = random_.tensor(rng2, (10, 10, 10), tt_rank=10, dtype=dtype)
    projected = riemannian.project(what, where)
    double_projected = riemannian.project(what, where)
    self.assertAllClose(ops.full(projected), ops.full(double_projected))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())