import jax
import jax.numpy as jnp

from ttax.base_class import TTBase
from ttax.base_class import TT
from ttax.base_class import TTMatrix
from ttax.compile import WrappedTT

def is_tt_tensor(arg) -> bool:
  """Determine whether the object is a `TT-Tensor` or `WrappedTT` with underlying `TT-Tensor`.
  
  :return: `True` if `TT-Tensor` or `WrappedTT(TT-Tensor)`, `False` otherwise
  :rtype: bool
  """
  return isinstance(arg, TT) or (isinstance(arg, WrappedTT) and 
         not arg.tt.is_tt_matrix)

def is_tt_matrix(arg) -> bool:
  """Determine whether the object is a `TT-Matrix` or `WrappedTT` with underlying `TT-Matrix`.
  
  :return: `True` if `TT-Matrix` or `WrappedTT(TT-Matrix)`, `False` otherwise
  :rtype: bool
  """
  return isinstance(arg, TTMatrix) or (isinstance(arg, WrappedTT) and
         arg.is_tt_matrix)

def is_tt_object(arg) -> bool:
  """Determine whether the object is a `TT-Tensor`, `TT-Matrix` or `WrappedTT` with one of them.
  
  :return: `True` if `TT-object`, `False` otherwise
  :rtype: bool
  """
  return is_tt_tensor(arg) or is_tt_matrix(arg)
