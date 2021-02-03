from absl.testing import absltest

import numpy as np
import jax
import jax.numpy as jnp
import jax.test_util as jtu
from jax.config import config

from ttax import random_
from ttax import decompositions
from ttax import ops

config.parse_flags_with_absl()


class DecompositionsTest(jtu.JaxTestCase):

  def testOrthogonalizeLeftToRight(self):
    dtype = jnp.float32
    rng = jax.random.PRNGKey(0)
    shape = (2, 4, 3, 3)
    tt_ranks = (1, 5, 2, 17, 1)
    updated_tt_ranks = (1, 2, 2, 6, 1)
    tens = random_.tensor(rng, shape, tt_rank=tt_ranks, dtype=dtype)
    orthogonal = decompositions.orthogonalize_tt_cores(tens)

    self.assertAllClose(ops.full(tens), ops.full(orthogonal), atol=1e-5,
                        rtol=1e-5)
    self.assertArraysEqual(updated_tt_ranks, orthogonal.tt_ranks)
    # Check that the TT-cores are orthogonal.
    for core_idx in range(4 - 1):
      core = orthogonal.tt_cores[core_idx]
      core = jnp.reshape(core, (updated_tt_ranks[core_idx] * shape[core_idx],
                                updated_tt_ranks[core_idx + 1]))
      should_be_eye = core.T @ core
      self.assertAllClose(np.eye(updated_tt_ranks[core_idx + 1]), should_be_eye)

  def testOrthogonalizeRightToLeft(self):
    dtype = jnp.float32
    rng = jax.random.PRNGKey(0)
    shape = (2, 4, 3, 3)
    tt_ranks = (1, 5, 2, 17, 1)
    updated_tt_ranks = (1, 5, 2, 3, 1)
    tens = random_.tensor(rng, shape, tt_rank=tt_ranks, dtype=dtype)
    orthogonal = decompositions.orthogonalize_tt_cores(tens, left_to_right=False)

    self.assertAllClose(ops.full(tens), ops.full(orthogonal), atol=1e-5,
                        rtol=1e-5)
    self.assertArraysEqual(updated_tt_ranks, orthogonal.tt_ranks)
    # Check that the TT-cores are orthogonal.
    for core_idx in range(1, 4):
      core = orthogonal.tt_cores[core_idx]
      core = jnp.reshape(core, (updated_tt_ranks[core_idx], shape[core_idx] *
                                updated_tt_ranks[core_idx + 1]))
      should_be_eye = core @ core.T
      self.assertAllClose(np.eye(updated_tt_ranks[core_idx]), should_be_eye)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())