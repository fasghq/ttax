from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax
import jax.numpy as jnp
import jax.test_util as jtu

from jax.config import config

import random_
import ops
import compile

config.parse_flags_with_absl()


@jax.jit
def abc(a, b, c):
  @compile.compile
  def main_loop(a, b):
    return {
        'type': 'running',
        'args': [['a', 'i', 'b'], ['c', 'i', 'd'], ['e', 'i', 'f'], ['a', 'c', 'e']],
        'res': ['b', 'd', 'f']
    }
  res = jnp.squeeze(main_loop(a, b, c)[-1])
  return res

class TTTensorTest(jtu.JaxTestCase):

  def testFuse(self):
    np.random.seed(1)
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 3)
    dtype = jnp.float32
    # tt_a = random_.tensor(rng1, (1, 2, 3, 4), tt_rank=2, dtype=dtype)
    # tt_b = random_.tensor(rng2, (1, 2, 3, 4), tt_rank=[1, 1, 4, 3, 1], dtype=dtype)
    # tt_c = random_.tensor(rng3, (1, 2, 3, 4), tt_rank=3, dtype=dtype)
    tt_a = random_.tensor(rng1, (10, 10, 10, 10), tt_rank=30, dtype=dtype)
    tt_b = random_.tensor(rng2, (10, 10, 10, 10), tt_rank=30, dtype=dtype)
    tt_c = random_.tensor(rng3, (10, 10, 10, 10), tt_rank=1, dtype=dtype)
    # for op in ['a*b*c', '<a*b, c>']:
    for op in ['<a*b, c>']:
      if op == 'a*b*c':
        func = lambda a, b, c: ops.multiply(ops.multiply(a, b), c)
        func_fused = compile.fuse(func)
        res_actual = ops.full(func(tt_a, tt_b, tt_c))
        res_desired = ops.full(func_fused(tt_a, tt_b, tt_c))
      elif op == '<a*b, c>':
        func = lambda a, b, c: ops.flat_inner(ops.multiply(a, b), c)
        func_fused = compile.fuse(func)
        res_actual = func(tt_a, tt_b, tt_c)
        res_desired = func_fused(tt_a, tt_b, tt_c)

        func_ = jax.jit(func)
        func_fused_ = jax.jit(func)
        import time
        for i in range(10):
          func_(tt_a, tt_b, tt_c)
        s = time.time()
        for i in range(100):
          func_(tt_a, tt_b, tt_c)
        e = time.time()
        print(e - s)

        for i in range(10):
          func_fused_(tt_a, tt_b, tt_c)
        s = time.time()
        for i in range(100):
          func_fused_(tt_a, tt_b, tt_c)
        e = time.time()
        print(e - s)

        for i in range(10):
          abc(tt_a, tt_b, tt_c)
        s = time.time()
        for i in range(100):
          abc(tt_a, tt_b, tt_c)
        e = time.time()
        print(e - s)

      self.assertAllClose(res_actual, res_desired)
      # TODO: also test that the fused version is faster.


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())