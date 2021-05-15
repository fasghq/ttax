Quick start
===========

Open this `page <https://colab.research.google.com/drive/1hxA-UzztmSqLpRhT70BmzdjN_wO17RND?usp=sharing>`_ in an interactive mode via Google Colaboratory.

This is a quick starting guide to look at the basics of working with ttax library. Our library provides routines for Tensor-Train object – a compact (factorized) representation of a tensor.

Let's import some libraries:

.. code-block:: 

    ! pip install ttax jax flax
    
    import jax
    import ttax
    import numpy as np
    
Converting to and from TT-format
--------------------------------

In code below we generate a random TT-tensor of size 10 x 5 x 2 with TT-rank = 3 and convert it to dense (full) format:

.. code-block:: python
    
    rng = jax.random.PRNGKey(42)
    dtype = jnp.float32
    a_tt = ttax.random_.tensor(rng, [10, 5, 2], tt_rank=3, dtype=dtype)
    a_dense = ttax.full(a_tt)
    
a_tt stores the factorized representation of the tensor, namely it stores the tensor as a product of 3 smaller tensors which are called TT-cores. You can access the TT-cores directly.
    
.. note::
    The larger the TT-rank, the more exactly the tensor will be converted, but the more memory and time everything will take.
    
Arithmetic operations
---------------------

TTAX provides different operations that can be applied to the tensors in the TT-format. 

Let's create several random TT-tensors of shape 3x4x5 and provide some arithmetic operations (sum and elementwise product) with them.

.. code-block:: python

    rng = jax.random.PRNGKey(41)
    a_tt = ttax.random_.tensor(rng, [3, 4, 5], tt_rank=30, dtype=dtype)
    b_tt = ttax.random_.tensor(rng, [3, 4, 5], tt_rank=30, dtype=dtype)
    sum_tt = a_tt + b_tt
    prod_tt = a_tt * b_tt
    twice_a_tt = 2 * a_tt
    
.. important::
    Most operations on TT-tensors increase the TT-rank. After applying a sequence of operations the TT-rank can increase by too much and we may want to reduce it. To do that there is a rounding operation, which finds the tensor that is of a smaller rank but is as close to the original one as possible.
    
.. code-block:: python

    rounded_prod_tt = ttax.round(prod_tt)
    
    a_max_tt_rank = np.max(a_tt.tt_ranks)
    b_max_tt_rank = np.max(b_tt.tt_ranks)
    exact_prod_max_tt_rank = np.max(prod_tt.tt_ranks)
    rounded_prod_max_tt_rank = np.max(rounded_prod_tt.tt_ranks)
    print('The TT-ranks of a and b are %d and %d. The TT-rank '
          'of their elementwise product is %d. The TT-rank of '
          'their product after rounding is %d.' % (a_max_tt_rank, 
          b_max_tt_rank, exact_prod_max_tt_rank, 
          rounded_prod_max_tt_rank))
    
Check that rounded TT-tensor of product converted to the full format is close to the product of full tensors a and b:

.. code-block:: python

    actual_prod = ttax.full(a_tt) * ttax.full(b_tt)
    prod_full = ttax.full(prod_tt)
    rounded_prod_full = ttax.full(rounded_prod_tt)
    np.testing.assert_allclose(actual_prod, rounded_prod_full, 1e-3)
    np.testing.assert_allclose(actual_prod, prod_full, 1e-3)
    np.testing.assert_allclose(prod_full, rounded_prod_full, 1e-3)
    
Working with TT-matrices
------------------------

Recall that for 2-dimensional tensors the TT-format coincides with the matrix low-rank format. However, sometimes matrices can have full matrix rank, but some tensor structure (for example a kronecker product of matrices). In this case there is a special object called Matrix TT-format. You can think of it as a sum of kronecker products (although it’s a bit more complicated than that).

Let’s say that you have a matrix of size 8 x 27. You can convert it into the matrix TT-format of tensor shape (2, 2, 2) x (3, 3, 3) (in which case the matrix will be represented with 3 TT-cores) or, for example, into the matrix TT-format of tensor shape (4, 2) x (3, 9) (in which case the matrix will be represented with 2 TT-cores).

.. code-block:: python

    rng = jax.random.PRNGKey(41)
    a_matrix_tt = ttax.random_.matrix(rng, ((2, 2, 2), (3, 3, 3)), tt_rank=4, dtype=dtype)
    twice_a_matrix_tt = 2.0 * a_matrix_tt
    prod_tt = a_matrix_tt * a_matrix_tt

.. code-block:: python

    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    left_shape = (2, 3, 4)
    sum_shape = (4, 3, 5)
    right_shape = (4, 4, 4)
    tt_a = ttax.random_.matrix(rng1, (left_shape, sum_shape), tt_rank=3, dtype=dtype)
    tt_b = ttax.random_.matrix(rng2, (sum_shape, right_shape), tt_rank=[1, 4, 3, 1],
                               dtype=dtype)
    res_actual = ttax.full(ttax.ops.matmul(tt_a, tt_b))
    res_desired = ttax.full(tt_a) @ ttax.full(tt_b)
    np.testing.assert_allclose(res_actual, res_desired, 1e-3)
    
Working with batches
--------------------

TTAX tries to support the work with multidimensional batches of tensors where it is possible, taking the input of multidimensional batches as if they were taking ordinary tensors. It means that if A and B are batches of TT-tensors/matices you can do A+B like you do for TT-tensors/matrices.

Let's see how it works. We create 2 batches of TT-tensors of the same batch size and then compare the result of sum in TT format with the one in full format.

.. code-block:: python

    rng1, rng2 = jax.random.split(jax.random.PRNGKey(0))
    dtype = jnp.float32
    tt_a = ttax.random_.tensor(rng1, (2, 1, 3, 4), tt_rank=2, batch_shape=(3,),
                      dtype=dtype)
    tt_b = ttax.random_.tensor(rng2, (2, 1, 3, 4), tt_rank=[1, 2, 4, 3, 1],
                      batch_shape=(3,), dtype=dtype)
    res_actual = ttax.ops.full(tt_a + tt_b)
    res_desired = ttax.ops.full(tt_a) + ttax.ops.full(tt_b)
    np.testing.assert_allclose(res_actual, res_desired, 1e-6)
    
.. note:: 
   
   You can use both tensor indexing and batch indexing.

You can use tensor indexing to get specified element / slice.

.. code-block:: python

    rng = jax.random.PRNGKey(41)
    tt = ttax.random_.tensor(rng, [2, 3, 4])
    print(tt[1, :, :].shape, "<- 2D Tensor-Train")
    print(tt[1:2, :, :].shape, "<- 3D Tensor-Train")
    
Similar idea for batch indexing but with a slightly different syntax.

.. code-block:: python
    
    rng = jax.random.PRNGKey(41)
    tt = ttax.random_.tensor(rng, [2, 3, 4], tt_rank=2, batch_shape=(3, 3, 3,))
    tt.batch_loc[1, :, :]
    
Speeding up your code
---------------------

Our library is written with the expectation of using the jax.jit for acceleration.

Some routines were based on einsum (see TTEinsum), to speed them up you can use fuse method (see compile.fuse).

Below is the example of how to use such speeding up and the difference it provides.

.. code-block:: python

    rng = jax.random.PRNGKey(42)
    tt_a = ttax.random_.tensor(rng, [3, 4, 5], tt_rank=30)
    tt_b = ttax.random_.tensor(rng, [3, 4, 5], tt_rank=30)
    tt_c = ttax.random_.tensor(rng, [3, 4, 5], tt_rank=1)
    
.. code-block:: python
    
    def f(a, b, c):
      return ttax.flat_inner(ttax.multiply(a, b), c)
      
.. code-block:: python

    fused_f = ttax.fuse(f)
    jit_f = jax.jit(f)
    jit_fused_f = jax.jit(fused_f)
    
.. code-block:: python

    %timeit f(tt_a, tt_b, tt_c)
    %timeit jit_f(tt_a, tt_b, tt_c)
    %timeit jit_fused_f(tt_a, tt_b, tt_c)
    

