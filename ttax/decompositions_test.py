from absl.testing import absltest

import numpy as np
import jax
import jax.numpy as jnp
import jax.test_util as jtu
from jax.config import config

from ttax import random_
from ttax import decompositions
from ttax import ops
from ttax.base_class import TT

config.parse_flags_with_absl()


class DecompositionsTest(jtu.JaxTestCase):

  def testOrthogonalizeLeftToRight(self):
    dtype = jnp.float32
    rng = jax.random.PRNGKey(0)
    shape = (2, 4, 3, 3)
    tt_ranks = (1, 5, 2, 17, 1)
    updated_tt_ranks = (1, 2, 2, 6, 1)
    tens = random_.tensor(rng, shape, tt_rank=tt_ranks, dtype=dtype)
    orthogonal = decompositions.orthogonalize(tens)

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
    orthogonal = decompositions.orthogonalize(tens, left_to_right=False)

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

  def testRound2d(self):
    dtype = jnp.float32
    rank = 5
    np.random.seed(0)
    x = np.random.randn(10, 20).astype(dtype)
    u, s, v = np.linalg.svd(x, full_matrices=False)
    core_1 = u @ np.diag(s)
    core_1 = core_1.reshape(1, 10, 10)
    core_2 = v
    core_2 = core_2.reshape(10, 20, 1)
    tt = TT((core_1, core_2))
    truncated_x = u[:, :rank] @ np.diag(s[:rank]) @ v[:rank, :]
    rounded = decompositions.round(tt, 5)
    self.assertAllClose(truncated_x, ops.full(rounded), rtol=1e-5, atol=1e-5)
    
  def testTTTensor(self):
    dtype = jnp.float32
    shape = (2, 1, 4, 3)
    np.random.seed(0)
    tens = np.random.randn(*shape).astype(dtype)
    tt_tens = decompositions.to_tt_tensor(tens, max_tt_rank=3)
    self.assertAllClose(ops.full(tt_tens), tens, atol=1e-5, rtol=1e-5)
    
  def testTTVector(self):
    dtype = jnp.float32
    vec_shape = (2, 1, 4, 3)
    rows = np.prod(vec_shape)
    np.random.seed(0)
    vec = np.random.randn(rows, 1).astype(dtype)
    tt_vec = decompositions.to_tt_matrix(vec, (vec_shape, None))
    self.assertAllClose(ops.full(tt_vec), vec, atol=1e-5, rtol=1e-5)
    
  def testTTMatrix(self):
    dtype = jnp.float32
    inp_shape = (2, 5, 2, 3)
    out_shape = (3, 3, 2, 3)
    np.random.seed(0)
    mat = np.random.randn(np.prod(out_shape), np.prod(inp_shape)).astype(dtype)
    tt_mat = decompositions.to_tt_matrix(mat, (out_shape, inp_shape),
                                         max_tt_rank=90)
    self.assertAllClose(ops.full(tt_mat), mat, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
