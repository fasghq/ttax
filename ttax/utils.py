import jax
import jax.numpy as jnp

from ttax.base_class import TTBase
from ttax.base_class import TT
from ttax.base_class import TTMatrix
from ttax.compile import WrappedTT

def is_tt_tensor(arg) -> bool:
  return isinstance(arg, TT) or (isinstance(arg, WrappedTT) and 
         not arg.tt.is_tt_matrix)

def is_tt_matrix(arg) -> bool:
  return isinstance(arg, TTMatrix) or (isinstance(arg, WrappedTT) and
         arg.is_tt_matrix)

def is_tt_object(arg) -> bool:
  return is_tt_tensor(arg) or is_tt_matrix(arg)
