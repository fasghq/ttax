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

config.parse_flags_with_absl()


class TTTensorTest(jtu.JaxTestCase):

  def testFullTensor2d(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(10, rank)
      b = np.random.rand(rank, 9)
      tt_cores = (a.reshape(1, 10, rank), b.reshape(rank, 9, 1))
      desired = np.dot(a, b)
      tt_tens = TT(tt_cores)
      actual = ops.full(tt_tens)
      self.assertAllClose(desired, actual)

  def testFullTensor2dBatch(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(3, 10, rank)
      b = np.random.rand(3, rank, 9)
      tt_cores = (a.reshape(3, 1, 10, rank), b.reshape(3, rank, 9, 1))
      desired = np.einsum('bij,bjk->bik', a, b)
      tt_tens = TT(tt_cores)
      actual = ops.full(tt_tens)
      self.assertAllClose(desired, actual)

  def testMultiply(self):
    # Multiply two TT-tensors.
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    tt_a = random_.tensor(rng1, (1, 2, 3, 4), tt_rank=2, dtype=dtype)
    tt_b = random_.tensor(rng2, (1, 2, 3, 4), tt_rank=[1, 1, 4, 3, 1],
                          dtype=dtype)

    res_actual1 = ops.full(ops.multiply(tt_a, tt_b))
    res_actual2 = ops.full(tt_a * tt_b)
    res_desired = ops.full(tt_a) * ops.full(tt_b)
    self.assertAllClose(res_actual1, res_desired)
    self.assertAllClose(res_actual2, res_desired)

  def testMultiplyBatch(self):
    # Multiply two batches of TT-tensors.
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    tt_a = random_.tensor(rng1, (1, 2, 3, 4), tt_rank=2, batch_shape=(3,),
                          dtype=dtype)
    tt_b = random_.tensor(rng2, (1, 2, 3, 4), tt_rank=[1, 1, 4, 3, 1],
                          batch_shape=(3,), dtype=dtype)

    res_actual1 = ops.full(ops.multiply(tt_a, tt_b))
    res_actual2 = ops.full(tt_a * tt_b)
    res_desired = ops.full(tt_a) * ops.full(tt_b)
    self.assertAllClose(res_actual1, res_desired)
    self.assertAllClose(res_actual2, res_desired)

  def testFlatInner(self):
    # Multiply two TT-tensors.
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    tt_a = random_.tensor(rng1, (1, 2, 3, 4), tt_rank=2, dtype=dtype)
    tt_b = random_.tensor(rng2, (1, 2, 3, 4), tt_rank=[1, 1, 4, 3, 1], dtype=dtype)
    res_actual = ops.flat_inner(tt_a, tt_b)
    res_desired = jnp.sum(ops.full(tt_a) * ops.full(tt_b))
    self.assertAllClose(res_actual, res_desired)


class TTMatrixTest(jtu.JaxTestCase):

  def testFull2d(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(9, rank)
      b = np.random.rand(rank, 10)
      tt_cores = (a.reshape(1, 3, 3, rank), b.reshape(rank, 2, 5, 1))
      desired = np.einsum('aijb,bpqc->ipjq', *tt_cores)
      desired = desired.reshape(6, 15)
      tt_tens = TTMatrix(tt_cores)
      actual = ops.full(tt_tens)
      self.assertAllClose(desired, actual)

  def testFull2dBatch(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(7, 9, rank)
      b = np.random.rand(7, rank, 10)
      tt_cores = (a.reshape(7, 1, 3, 3, rank), b.reshape(7, rank, 2, 5, 1))
      desired = np.einsum('taijb,tbpqc->tipjq', *tt_cores)
      desired = desired.reshape(7, 6, 15)
      tt_tens = TTMatrix(tt_cores)
      actual = ops.full(tt_tens)
      self.assertAllClose(desired, actual)

  def testMatmul(self):
    # Multiply two TT-matrices.
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    left_shape = (2, 3, 4)
    sum_shape = (4, 3, 5)
    right_shape = (4, 4, 4)
    tt_a = random_.matrix(rng1, (left_shape, sum_shape), tt_rank=3, dtype=dtype)
    tt_b = random_.matrix(rng2, (sum_shape, right_shape), tt_rank=[1, 4, 3, 1],
                          dtype=dtype)

    res_actual = ops.full(ops.matmul(tt_a, tt_b))
    res_desired = ops.full(tt_a) @ ops.full(tt_b)
    # TODO: why such low precision?
    self.assertAllClose(res_actual, res_desired, rtol=1e-3)

  def testMultiply(self):
    # Elementwise multiply two TT-matrices.
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    left_shape = (2, 3, 4)
    right_shape = (4, 4, 4)
    tt_a = random_.matrix(rng1, (left_shape, right_shape), tt_rank=3,
                          dtype=dtype)
    tt_b = random_.matrix(rng2, (left_shape, right_shape), tt_rank=[1, 4, 3, 1],
                          dtype=dtype)

    res_actual1 = ops.full(ops.multiply(tt_a, tt_b))
    res_actual2 = ops.full(tt_a * tt_b)
    res_desired = ops.full(tt_a) * ops.full(tt_b)
    self.assertAllClose(res_actual1, res_desired, rtol=1e-4)
    self.assertAllClose(res_actual2, res_desired, rtol=1e-4)

  def testMultiplyBatch(self):
    # Elementwise multiply two batches of TT-matrices.
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    left_shape = (2, 3, 4)
    right_shape = (4, 4, 4)
    tt_a = random_.matrix(rng1, (left_shape, right_shape), tt_rank=3,
                          batch_shape=(3,), dtype=dtype)
    tt_b = random_.matrix(rng2, (left_shape, right_shape), tt_rank=[1, 4, 3, 1],
                          batch_shape=(3,), dtype=dtype)

    res_actual1 = ops.full(ops.multiply(tt_a, tt_b))
    res_actual2 = ops.full(tt_a * tt_b)
    res_desired = ops.full(tt_a) * ops.full(tt_b)
    # TODO: why such low precision?
    self.assertAllClose(res_actual1, res_desired, rtol=1e-3)
    self.assertAllClose(res_actual2, res_desired, rtol=1e-3)

  def testFlatInner(self):
    # Multiply two TT-matrices.
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    left_shape = (2, 3, 4)
    right_shape = (4, 4, 4)
    tt_a = random_.matrix(rng1, (left_shape, right_shape), tt_rank=3,
                          dtype=dtype)
    tt_b = random_.matrix(rng2, (left_shape, right_shape), tt_rank=[1, 4, 3, 1],
                          dtype=dtype)
    res_actual = ops.flat_inner(tt_a, tt_b)
    res_desired = jnp.sum(ops.full(tt_a) * ops.full(tt_b))
    self.assertAllClose(res_actual, res_desired)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())