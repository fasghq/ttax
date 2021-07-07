from absl.testing import absltest

import numpy as np
import jax
import jax.numpy as jnp
import jax.test_util as jtu
from jax.config import config

from ttax import random_
from ttax import ops
from ttax import riemannian

config.parse_flags_with_absl()


class TTTensorTest(jtu.JaxTestCase):

  def testToAndFromDeltas(self):
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    what = random_.tensor(rng1, (2, 3, 4), tt_rank=4, dtype=dtype)
    where = random_.tensor(rng2, (2, 3, 4), tt_rank=3, dtype=dtype)
    projected = riemannian.project(what, where)

    deltas = riemannian.tangent_to_deltas(projected)
    reconstructed_projected = riemannian.deltas_to_tangent(deltas, where)
    # Tangent space element norm can be computed from deltas norm.
    projected_normsq_desired = ops.flat_inner(projected, projected)
    projected_normsq_actual = sum([jnp.sum(c * c) for c in deltas])
    self.assertAllClose(ops.full(projected), ops.full(reconstructed_projected))
    self.assertAllClose(projected_normsq_desired, projected_normsq_actual)

  def testToAndFromDeltasBatch(self):
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    what = random_.tensor(rng1, (2, 3, 4), tt_rank=4, dtype=dtype,
                          batch_shape=(3,))
    where = random_.tensor(rng2, (2, 3, 4), tt_rank=3, dtype=dtype,
                           batch_shape=(3,))
    projected = riemannian.project(what, where)

    deltas = riemannian.tangent_to_deltas(projected)
    reconstructed_projected = riemannian.deltas_to_tangent(deltas, where)
    self.assertAllClose(ops.full(projected), ops.full(reconstructed_projected))
    # Tangent space element norm can be computed from deltas norm.
    projected_normsq_desired = ops.flat_inner(projected, projected)
    for i in range(3):
      projected_normsq_actual = sum([jnp.sum(c[i] * c[i]) for c in deltas])
      self.assertAllClose(projected_normsq_desired[i], projected_normsq_actual)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
