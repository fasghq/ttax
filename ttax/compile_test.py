from absl.testing import absltest
from absl.testing import parameterized

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.test_util as jtu
from jax.config import config

from ttax import random_
from ttax import ops
from ttax import compile

config.parse_flags_with_absl()


def benchmark(func, *args):
  func_jit = jax.jit(func)
  # Warmup.
  for _ in range(3):
    func_jit(*args).block_until_ready()

  num_repeats = 100
  s = time.time()
  outs = []
  for _ in range(num_repeats):
    outs.append(func_jit(*args))
  for o in outs:
    o.block_until_ready()
  e = time.time()
  return (e - s) / num_repeats


class TTTensorTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      ('_a*b*c', 'a*b*c'),
      ('<a*b, c>', '<a*b, c>')
  )
  def testFuse(self, op_type):
    np.random.seed(1)
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 3)
    dtype = jnp.float32
    tt_a = random_.tensor(rng1, (1, 2, 3, 4), tt_rank=2, dtype=dtype)
    tt_b = random_.tensor(rng2, (1, 2, 3, 4), tt_rank=[1, 1, 4, 3, 1], dtype=dtype)
    tt_c = random_.tensor(rng3, (1, 2, 3, 4), tt_rank=3, dtype=dtype)
    if op_type == 'a*b*c':
      func = lambda a, b, c: ops.multiply(ops.multiply(a, b), c)
      fused_func = compile.fuse(func)
      res_actual = ops.full(func(tt_a, tt_b, tt_c))
      res_desired = ops.full(fused_func(tt_a, tt_b, tt_c))
    elif op_type == '<a*b, c>':
      func = lambda a, b, c: ops.flat_inner(ops.multiply(a, b), c)
      fused_func = compile.fuse(func)
      res_actual = func(tt_a, tt_b, tt_c)
      res_desired = fused_func(tt_a, tt_b, tt_c)

      self.assertAllClose(res_actual, res_desired)

  def testFuseIsFaster(self):
    np.random.seed(1)
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 3)
    dtype = jnp.float32
    tt_a = random_.tensor(rng1, (10, 10, 10, 10), tt_rank=30, dtype=dtype)
    tt_b = random_.tensor(rng2, (10, 10, 10, 10), tt_rank=30, dtype=dtype)
    tt_c = random_.tensor(rng3, (10, 10, 10, 10), tt_rank=1, dtype=dtype)
    func = lambda a, b, c: ops.flat_inner(ops.multiply(a, b), c)
    fused_func = compile.fuse(func)

    func_speed = benchmark(func, tt_a, tt_b, tt_c)
    fused_func_speed = benchmark(fused_func, tt_a, tt_b, tt_c)
    # Check that fused version is at least 10x faster than non-fused.
    self.assertLess(fused_func_speed, 0.1 * func_speed)


class TTMatrixTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      ('(a*b) @ c', '(a*b) @ c'),
  )
  def testFuse(self, op_type):
    np.random.seed(1)
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 3)
    dtype = jnp.float32
    left_shape = (2, 3, 4)
    sum_shape = (4, 3, 5)
    right_shape = (4, 4, 4)
    tt_a = random_.matrix(rng1, (left_shape, sum_shape), tt_rank=3, dtype=dtype)
    tt_b = random_.matrix(rng2, (left_shape, sum_shape), tt_rank=3, dtype=dtype)
    tt_c = random_.matrix(rng3, (sum_shape, right_shape), tt_rank=[1, 4, 3, 1],
                              dtype=dtype)
    if op_type == '(a*b) @ c':
      def func(a, b, c):
        return (a * b) @ c
      fused_func = compile.fuse(func)
      res_actual = ops.full(func(tt_a, tt_b, tt_c))
      res_desired = ops.full(fused_func(tt_a, tt_b, tt_c))
      self.assertAllClose(res_actual, res_desired, rtol=1e-4)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())