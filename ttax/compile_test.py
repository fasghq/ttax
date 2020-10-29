from absl.testing import absltest
from absl.testing import parameterized

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.test_util as jtu

from jax.config import config

import random_
import ops
import compile

config.parse_flags_with_absl()


def benchmark(func, *args):
  func_jit = jax.jit(func)
  # Warmup.
  for _ in range(3):
    func_jit(*args).block_until_ready()

  num_repeats = 100
  s = time.time()
  for _ in range(num_repeats):
    func_jit(*args)
  e = time.time()
  return (e - s) / num_repeats


class TTTensorTest(jtu.JaxTestCase):

  def testFuse(self):
    np.random.seed(1)
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 3)
    dtype = jnp.float32
    tt_a = random_.tensor(rng1, (1, 2, 3, 4), tt_rank=2, dtype=dtype)
    tt_b = random_.tensor(rng2, (1, 2, 3, 4), tt_rank=[1, 1, 4, 3, 1], dtype=dtype)
    tt_c = random_.tensor(rng3, (1, 2, 3, 4), tt_rank=3, dtype=dtype)
    for op in ['a*b*c', '<a*b, c>']:
      if op == 'a*b*c':
        func = lambda a, b, c: ops.multiply(ops.multiply(a, b), c)
        fused_func = compile.fuse(func)
        res_actual = ops.full(func(tt_a, tt_b, tt_c))
        res_desired = ops.full(fused_func(tt_a, tt_b, tt_c))
      elif op == '<a*b, c>':
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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())