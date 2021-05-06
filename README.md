# TTAX: Tensor-Train toolbox on Jax
Implementation of Tensor-Train toolbox in Jax containing several routines for working with tensors in TT format in Python.
## Installation
[JAX](https://github.com/google/jax/blob/master/README.md) and [Flax](https://github.com/google/flax#readme) are required for installation (see instructions there).
Install TTAX from PyPi:
```
> pip install ttax
```
## Quick start
This is a quick starting guide to look at the basics of working with ttax library. Our library provides routines for Tensor-Train object – a compact (factorized) representation of a tensor.

In example below we create TT-tensor, multiply it by constant and convert it to full tensor format.
```python
!pip install ttax jax flax

import ttax
import numpy as np

rng = jax.random.PRNGKey(42)
t = ttax.random.tensor(rng, [10, 5, 2], tt_rank=3)
print(ttax.full(2 * t))
```
## Structure overview
For operations with TT tensors we provide these main classes:
```python
@flax.struct.dataclass
class TT(TTBase):
  tt_cores: List[jnp.array]
```
```python
@flax.struct.dataclass
class TTMatrix(TTBase):
  tt_cores: List[jnp.array]
```

These classes realised in `base_class.py` are 2 main classes which are used in most TT routines. Both of them inherit some basic methods from `TTBase` class but have the different ones depending on the type of object contained (TT-tensor/TT-matrix).  

We try to support multidymensional batches of TT-tensors/TT-matrices where it is possible (see `tt_vmap`).

```python
class WrappedTT:
  def __init__(self, tt: TT, tt_inputs=None, tt_einsum=None):
    self.tt = tt
    self.tt_inputs = tt_inputs
    self.tt_einsum = tt_einsum
```
```python
class TTEinsum:
  def __init__(self, inputs, output, how_to_apply):
    self.inputs = inputs
    self.output = output
    self.how_to_apply = how_to_apply
```

These classes realised in `compile.py` are the basis for tensor operations which could be reduced to the use of `einsum`. Such approach allows to optimize the sequence of operetions after their fusion (see `fuse`).

To learn more how to use routines from `module_name.py` module you can use corresponding `module_name_test.py` as example. 

## License
MIT License


