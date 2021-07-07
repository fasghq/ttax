# TTAX: Tensor-Train toolbox on Jax

Implementation of Tensor-Train toolbox in Jax containing several routines for working with tensors in TT format in Python.
## Installation
[JAX](https://github.com/google/jax/blob/master/README.md) and [Flax](https://github.com/google/flax#readme) are required for installation (see instructions there).
Install TTAX from PyPi:
```
pip install ttax
```
## Quick start
This is a quick starting guide to look at the basics of working with ttax library. Our library provides routines for Tensor-Train object – a compact (factorized) representation of a tensor.

In example below we create TT-tensor, multiply it by constant and convert it to full tensor format.
```python
import ttax
import numpy as np
import jax

rng = jax.random.PRNGKey(42)
t = ttax.random.tensor(rng, [10, 5, 2], tt_rank=3)
print(ttax.full(2 * t))
```
Detailed information read [here](https://ttax.readthedocs.io/en/latest/quickstart.html).

## Structure overview

The main classes representing TT-tensors and TT-matrices are [`TT`](https://ttax.readthedocs.io/en/latest/api.html?highlight=ttax.base_class.TT#ttax.base_class.TT) and [`TTMatrix`](https://ttax.readthedocs.io/en/latest/api.html?highlight=ttax.base_class.TT#ttax.base_class.TTMatrix).
The base for operations on the `einsum` method are [`TTEinsum`](https://ttax.readthedocs.io/en/latest/api.html?highlight=ttax.base_class.TT#ttax.ops.TTEinsum) and [`WrappedTT`](https://ttax.readthedocs.io/en/latest/api.html?highlight=ttax.base_class.TT#ttax.compile.WrappedTT).

## License
MIT License

