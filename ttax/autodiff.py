import jax
import jax.numpy as jnp

from typing import Callable

from ttax import decompositions
from ttax import riemannian

from ttax.base_class import TTTensOrMat

TangentVector = TTTensOrMat


def _enforce_gauge_conditions(deltas, left):
  """Project deltas that define tangent space vec onto the gauge conditions."""
  proj_deltas = []
  for i in range(left.ndim):
    right_r = left.tt_ranks[i + 1]
    q = left.tt_cores[i].reshape((-1, right_r))
    if i < left.ndim - 1:
      proj_delta = deltas[i]
      proj_delta = proj_delta.reshape((-1, right_r))
      proj_delta -= q @ (q.T @ proj_delta)
      proj_delta = proj_delta.reshape(left.tt_cores[i].shape)
    else:
      proj_delta = deltas[i]
    proj_deltas.append(proj_delta)
  return proj_deltas


def grad(func: Callable[[TTTensOrMat], float]) -> Callable[[TTTensOrMat], TangentVector]:
  """Riemannian autodiff: decorator to compute gradient projected on tangent space.

  Returns a function which at a point X computes projection of the euclidian gradient
  df/dx onto the tangent space of TT-tensors at point x.
  Warning: grad may not work for some function, e.g. ones that include QR or
  SVD decomposition (ttax.project, ttax.round) or for functions that work
  with TT-cores directly (in contrast to working with TT-object only via ttax
  functions). In this cases this function can silently return wrong results!

  Example:
      # Scalar product with some predefined tensor squared 0.5 * <x, t>**2.
      # It's gradient is <x, t> t and it's Riemannian gradient is
      #     ttax.project(<x, t> * t, x)
      f = lambda x: 0.5 * ttax.flat_inner(x, t)**2
      # Equivalent to
      #   projected_grad = ttax.project(ttax.flat_inner(x, t) * t, x)
      projected_grad = ttax.grad(f)(x)

  Args:
      func: function that takes TensorTrain object as input and outputs a number.

  Returns:
      Function that computes Riemannian gradient of `func` at a given point.

  See also:
      ttax.hessian_vector_product
  """
  def _grad(x: TTTensOrMat) -> TangentVector:
    # TODO: support runtime checks
    left = decompositions.orthogonalize(x)
    right = decompositions.orthogonalize(left, left_to_right=False)
    deltas = [right.tt_cores[0]]
    deltas += [jnp.zeros_like(cc) for cc in right.tt_cores[1:]]

    def augmented_func(d):
      x_projection = riemannian.deltas_to_tangent(d, x)
      return func(x_projection)

    function_value, cores_grad = jax.value_and_grad(augmented_func)(deltas)

    deltas = _enforce_gauge_conditions(cores_grad, left)
    return riemannian.deltas_to_tangent(deltas, x)
  return _grad


def hessian_vector_product(func: Callable[[TTTensOrMat], float]) -> Callable[[TTTensOrMat, TangentVector], TangentVector]:
  """P_x [d^2f/dx^2] P_x vector, i.e. Riemannian hessian by vector product.

    Computes
      P_x [d^2f/dx^2] P_x vector
    where P_x is projection onto the tangent space of TT at point x and
    d^2f/dx^2 is the Hessian of the function.
    Note that the true Riemannian hessian also includes the manifold curvature
    term which is ignored here.
    Warning: this is experimental feature and it may not work for some function,
    e.g. ones that include QR or SVD decomposition (t3f.project, t3f.round) or
    for functions that work with TT-cores directly (in contrast to working with
    TT-object only via t3f functions). In this cases this function can silently
    return wrong results!

    Example:
        # Quadratic form with matrix A: <x, A x>.
        # It's gradient is (A + A.T) x, it's Hessian is (A + A.T)
        # It's Riemannian Hessian by vector product is
        #     proj_vec = t3f.project(vector, x)
        #     t3f.project(t3f.matmul(A + t3f.transpose(A), proj_vec), x)
        f = lambda x: t3f.bilinear_form(A, x, x)
        res = t3f.hessian_vector_product(f, x, vector)

    Args:
        func: function that takes TensorTrain object as input and outputs a number.

    Returns:
        `TensorTrain`, result of the Riemannian hessian by vector product.

    See also:
        `ttax.grad`
    """
  def _hess_by_vec(x: TTTensOrMat, vector: TangentVector) -> TangentVector:
    left = decompositions.orthogonalize(x)
    right = decompositions.orthogonalize(left, left_to_right=False)
    deltas = [right.tt_cores[0]]
    deltas += [jnp.zeros_like(cc) for cc in right.tt_cores[1:]]

    def augmented_outer_func(deltas_outer):

      def augmented_inner_func(deltas_inner):
        x_projection = riemannian.deltas_to_tangent(deltas_inner, x)
        return func(x_projection)

      function_value, cores_grad = jax.value_and_grad(augmented_inner_func,
                                                      deltas_outer)
      # TODO: support runtime checks

      vector_projected = riemannian.project(vector, x)
      vec_deltas = riemannian.tangent_space_to_deltas(vector_projected)
      products = [jnp.sum(a * b) for a, b in zip(cores_grad, vec_deltas)]
      return sum(products)

    _, second_cores_grad = jax.value_and_grad(augmented_outer_func, deltas)
    final_deltas = _enforce_gauge_conditions(second_cores_grad, left)
    # TODO: pass left and right?
    return riemannian.deltas_to_tangent(final_deltas, x)
  return _hess_by_vec
