"""Utils for compiling functions defined as einsum strings.

Here we use the notion of *tt_einsum*, which is similar to einsum strings, but
with more structure.

Basic element of tt_einsum is a *tt_einsum_core*: list with three elements,
which defines an einsum string for a single TT-core. First element is indices
for the left TT-rank, second element is the indices for the main dimensions of
the resulting TT-core, and the last element is the indices for the right
TT-rank.

TT_einsum consists of list of input and output cores defined like with
tt_einsum_cores.
"""

from typing import Callable, List, Union, Dict
import tree
import opt_einsum as oe
import numpy as np
import jax.numpy as jnp
from string import ascii_lowercase, ascii_uppercase
import copy

from ttax import ops
from ttax.base_class import TT
from ttax.base_class import TTMatrix

# You can use this in TT-einsum expressions. It will be 'i' when working with
# TT-tensors and 'ij' when working with TT-matrices.
I_OR_IJ = 'I_OR_IJ'


class WrappedTT:
  """A class which wraps TT, which is needed for fusion to work.

  Base TT class can only have jnp.array objects so that you can pass it into
  jitted function. But, for fusing two functions together we need to track which
  operation created a TT object, so while fusing ops we wrap TT objects with
  this class, to track that.
  """

  def __init__(self, tt: TT, tt_inputs=None, tt_einsum=None):
    self.tt = tt
    self.tt_inputs = tt_inputs
    self.tt_einsum = tt_einsum

  def __mul__(self, other):
    return ops.multiply(self, other)

  def __matmul__(self, other):
    return ops.matmul(self, other)

  @property
  def tt_cores(self):
    return self.tt.tt_cores

  @property
  def batch_shape(self):
    return self.tt.batch_shape

  @property
  def shape(self):
    return self.tt.shape

  @property
  def axis_dim(self):
    return self.tt.axis_dim

  @property
  def num_batch_dims(self):
    return self.tt.num_batch_dims

  @property
  def is_tt_matrix(self):
    return self.tt.is_tt_matrix


class TTEinsum:

  def __init__(self, inputs, output, how_to_apply, batch_einsum_rule=None):
    self.inputs = inputs
    self.output = output
    if batch_einsum_rule is not None:
      self.inputs = batch_einsum_rule.split('->')[0].split(',')
      self.output = list(batch_einsum_rule.split('->')[1])
    else:
      self.input_b = None
      self.output_b = None
    self.how_to_apply = how_to_apply
    self.batch_einsum_rule = batch_einsum_rule

    if how_to_apply not in ['independent', 'cumulative']:
      raise ValueError('Unsupported "how_to_apply" type "%s"' % how_to_apply)

  def set_batch_einsum_rule(self, batch_einsum_rule):
    assert self.batch_einsum_rule is None
    tt_einsum = copy.deepcopy(self)
    tt_einsum.batch_einsum_rule = batch_einsum_rule
    return tt_einsum

  def to_vanilla_einsum(self):
    """Build regular einsum."""
    inputs = []
    for inp in self.inputs:
      inputs.append(''.join(inp))
    _, batch_out = self.batch_einsum_rule.split('->')
    return (','.join([b + a for (a, b) in zip(inputs, self.input_b)]) + 
           '->' + batch_out + ''.join(self.output))

  def resolve_batch_einsum_rule(self, num_batch_dims):
    if self.batch_einsum_rule is not None:
      return self
    if num_batch_dimes == 0:
      return self
    i = 0
    batch_letters = ""
    for _ in range(num_batch_dims):
      choose_letter = True
      while choose_letter:
        letter = ascii_uppercase[i]
        i += 1
        if letter not in self.to_vanilla_einsum():
          choose_letter = False
      batch_letters += letter
    tt_einsum = copy.deepcopy(self)
    tt_einsum.batch_einsum_rule = ','.join(batch_letters) + '->' + batch_letters
    return tt_einsum

  def apply_mapping(self, mapping: Dict[str, str]):
    """Rename letters according to the given mapping."""
    new_inputs = []
    for inp in self.inputs:
      new_inputs.append(apply_single_mapping(inp, mapping))
    new_output = apply_single_mapping(self.output, mapping)
    return TTEinsum(inputs=new_inputs, output=new_output,
                    how_to_apply=self.how_to_apply)

  def change_input(self, input_idx: int, new_inputs: List):
    """Change argument input_idx into new_inputs.

    E.g.
      tt_einsum = TTEinsum(inputs=[['a', 'i', 'b'], ['c', 'i', 'd']],
                           output=['ac', 'i', 'bd'],
                           how_to_apply='independent')
      tt_einsum.change_input(0, [['e', 'i', 'f'], ['g', 'i', 'h']])
      print(tt_einsum.to_vanilla_einsum())  # 'eif,gih,cid->acibd'.
    """
    prefix = self.inputs[:input_idx]
    postfix = self.inputs[input_idx + 1:]
    new_inputs = prefix + new_inputs + postfix
    return TTEinsum(new_inputs, self.output, self.how_to_apply)

  def to_distinct_letters(self, distinct_from):
    """Rename letters to make them distinct from letters used in distinct_from."""
    # assert that either no ... or everything is ... in batch rules
    distinct_from_einsum = distinct_from.to_vanilla_einsum()
    # TODO: add upper case
    vacant_letters = [l for l in ascii_lowercase if l not in distinct_from_einsum]
    einsum = self.to_vanilla_einsum()
    curr_unique_letters = [l for l in einsum if l in ascii_lowercase]
    mapping = {}
    for i, l in enumerate(curr_unique_letters):
      mapping[l] = vacant_letters[i]
    return self.apply_mapping(mapping)

  def resolve_i_or_ij(self, is_tt_matrix):
    """Return a version of TTEinsum with I_OR_IJ changed to either 'i' or 'ij'.
    """
    def resolve(el):
      if el == I_OR_IJ:
        return 'ij' if is_tt_matrix else 'i'
      return el
    new_inputs = tree.map_structure(resolve, self.inputs)
    new_output = tree.map_structure(resolve, self.output)
    return TTEinsum(new_inputs, new_output, self.how_to_apply, self.batch_einsum_rule)

def apply_single_mapping(strings, mapping):
  """Apply letter mapping to a list of strings."""
  new_strings = []
  for str in strings:
    curr_str = ''
    for l in str:
      curr_str += mapping.get(l, l)
    new_strings.append(curr_str)
  return new_strings
  
  def to_distinct_letters_batch(self, distinct_from):
    """Rename batch letters to make them distinct from letters used in distinct_from."""
    distinct_from_einsum = distinct_from.batch_einsum_rule
    vacant_letters = [l for l in ascii_uppercase if l not in distinct_from_einsum]
    einsum = self.batch_einsum_rule
    curr_unique_letters = [l for l in einsum if l in ascii_uppercase]
    new_einsum = ""
    for l in einsum:
      if l in curr_unique_letters or l not in ascii_uppercase:
        new_einsum += l
      else:
        new_einsum += vacant_letters[0]
        vacant_letters = vacant_letters[1:]
    return TTEinsum(self.inputs, self.output, self.how_to_apply, new_einsum)
    
def change_input_batch(self, input_idx: int, new_inputs: list):
    prefix = self.input_b[:input_idx]
    postfix = self.inputs_b[input_idx + 1:]
    new_inputs = prefix + new_inputs + postfix
    new_einsum = to_einsum_rule(new_inputs, self.output_b)
    return TTEinsum(self.inputs, self.output, self.how_to_apply, new_einsum)
    
def to_einsum_rule(list_in, list_out):
    """Build einsum rule from input and output lists."""
    inputs = []
    for inp in list_in:
      inputs.append(''.join(inp))
    return ','.join([a for a in inputs]) + '->' + ''.join(list_out)
   
def apply_mapping_batch(self, mapping: Dict[str, str]):
    """Rename letters according to the given mapping."""
    new_inputs = []
    for inp in self.input_b:
      new_inputs.append(apply_single_mapping(inp, mapping))
    new_output = apply_single_mapping(self.output_b, mapping)
    new_einsum = to_einsum_rule(new_inputs, new_output)
    return TTEinsum(self.inputs, self.output, self.how_to_apply, new_einsum)

def fuse_batch_einsum(tt_einsum: TTEinsum,
                      tensor_args: List[Union[TT, WrappedTT]]) -> TTEinsum:
  """Fuse
  """
  curr_tt_einsum = copy.deepcopy(tt_einsum)

  curr_tt_einsum_inp_idx = 0
  new_tensor_args = []
  for arg_idx, arg in enumerate(tensor_args):
    if isinstance(arg, WrappedTT) and arg.tt_einsum is not None:
      assert arg.tt_einsum.how_to_apply == 'independent'
      # Step 1.
      new_arg_tt_einsum = arg.tt_einsum.to_distinct_letters_batch(curr_tt_einsum)
      # Step 2.
      unchanged_inp = copy.deepcopy(curr_tt_einsum.input_b[curr_tt_einsum_inp_idx])
      curr_tt_einsum = curr_tt_einsum.change_input_batch(curr_tt_einsum_inp_idx,
                                                   new_arg_tt_einsum.input_b)
      # Step 3.
      mapping = {}
      for fr, to in zip(unchanged_inp, new_arg_tt_einsum.output_b):
        if len(fr) == len(to):
          for i in range(len(fr)):
            mapping[fr[i]] = to[i]
        else:
          raise ValueError()

      curr_tt_einsum = curr_tt_einsum.apply_mapping_batch(mapping)

      curr_tt_einsum_inp_idx += len(new_arg_tt_einsum.input_b)
      new_tensor_args += arg.tt_inputs
    else:
      curr_tt_einsum_inp_idx += 1
      new_tensor_args.append(arg)

  return curr_tt_einsum, new_tensor_args
      

def _fuse_tt_einsums(tt_einsum: TTEinsum,
                     tensor_args: List[Union[TT, WrappedTT]]) -> TTEinsum:
  """Fuse this TTEinsum with tensor args each of which can be generated by an einsum.

  Let us that you want to fuse the following op:
    flat_inner(a * b, c)
    = sum_{i_1, ..., i_d} a[i_1, ..., i_d] b[i_1, ..., i_d] c[i_1, ..., i_d]
  In this case, your tt_einsum will represent flat_inner
    tt_einsum represents 'aib,cid,ac->bd'
  and the first argument is actually a result of elementwise product, i.e.
    tensor_args[0].tt_einsum represents 'aib,cid->acibd'.

  Then this function will output TTEinsum representing a single fused op
    result represents 'lno,mnp,cnd,lmc->opd'

  It will do so by applying the following 3 steps to every tensor_arg:
  1) change letters of the tensor arg so that they don't intersect with the
      letters of the current einsum. E.g. in this case we will
      change 'aib,cid->acibd' to 'lno,mnp->lmnop': this is the same einsum,
      but it uses letters that are not intersecting with letters in
      the curr_einsum = 'aib,cid,ac->bd'.
  2) change current input of curr_einsum ('aib') into the einsum inputs that
      create this argument ('lno,mnp'). This changes curr_einsum
      from 'aib,cid,ac->bd' into 'lno,mnp,cid,ac->bd'.
  3) make letters in the curr_einsum consistent, e.g. change it
      from 'lno,mnp,cid,ac->bd' to 'lno,mnp,cnd,lmc->opd'. To do that notice
      that in the previous step we changed 'aib' into 'lno,mnp->lmnop', so
      'a' changes to 'lm', 'i' changes to 'n' and 'b' changes to 'op'.
  """
  curr_tt_einsum = copy.deepcopy(tt_einsum)

  if batch einsum rule anywhere, call .resolve_batch_einsum_rule everywhere.

  curr_tt_einsum_inp_idx = 0
  new_tensor_args = []
  for arg_idx, arg in enumerate(tensor_args):
    if isinstance(arg, WrappedTT) and arg.tt_einsum is not None:
      assert arg.tt_einsum.how_to_apply == 'independent'
      # Step 1.
      new_arg_tt_einsum = arg.tt_einsum.to_distinct_letters(curr_tt_einsum)
      # Step 2.
      unchanged_inp = copy.deepcopy(curr_tt_einsum.inputs[curr_tt_einsum_inp_idx])
      curr_tt_einsum = curr_tt_einsum.change_input(curr_tt_einsum_inp_idx,
                                                   new_arg_tt_einsum.inputs)
      # Step 3.
      mapping = {}
      for fr, to in zip(unchanged_inp, new_arg_tt_einsum.output):
        if len(fr) == 1:
          mapping[fr] = to
        elif len(fr) == len(to):
          for i in range(len(fr)):
            mapping[fr[i]] = to[i]
        else:
          raise ValueError()

      curr_tt_einsum = curr_tt_einsum.apply_mapping(mapping)

      curr_tt_einsum_inp_idx += len(new_arg_tt_einsum.inputs)
      new_tensor_args += arg.tt_inputs
    else:
      curr_tt_einsum_inp_idx += 1
      new_tensor_args.append(arg)

  return curr_tt_einsum, new_tensor_args


def to_function(tt_einsum: TTEinsum) -> Callable:
  """Compile TT-einsum into a function.

  Example:
  def multiply(a, b):
    tt_einsum = TTEinsum(
        inputs=[['a', 'i', 'b'], ['c', 'i', 'd']],
        output=['ac', 'i', 'bd'],
        how_to_apply='independent'
    )
    func = tt_einsum.to_function()
    return func(a, b)
  """
  if tt_einsum.how_to_apply == 'independent':
    return compile_independent(tt_einsum)
  elif tt_einsum.how_to_apply == 'cumulative':
    return compile_cumulative(tt_einsum)


def compile_independent(tt_einsum: TTEinsum) -> Callable:
  def new_func(*args, batch_einsum=None):
    are_tt_matrix_inputs = args[0].is_tt_matrix
    tt_einsum_ = tt_einsum.resolve_i_or_ij(are_tt_matrix_inputs)

    is_fusing = any([isinstance(tt, WrappedTT) for tt in args])
    if is_fusing:
      # Have to use a different name to make upper level tt_einsum visible.
      tt_einsum_, args = _fuse_tt_einsums(tt_einsum_, args)
    einsum = tt_einsum_.to_vanilla_einsum()
    num_batch_dims = args[0].num_batch_dims
    # TODO: support broadcasting.
    res_batch_shape = list(args[0].batch_shape)
    # TODO: do in parallel w.r.t. cores.
    # TODO: use optimal einsum.
    res_cores = []
    for i in range(len(args[0].tt_cores)):
      curr_input_cores = [tt.tt_cores[i] for tt in args]
      core = oe.contract(einsum, *curr_input_cores, backend='jax')
      shape = core.shape[num_batch_dims:]
      num_left_rank_dims = len(tt_einsum_.output[0])
      num_tensor_dims = len(tt_einsum_.output[1])
      split_points = (num_left_rank_dims, num_left_rank_dims + num_tensor_dims)
      new_shape = np.split(shape, split_points)
      left_rank = np.prod(new_shape[0])
      right_rank = np.prod(new_shape[2])
      new_shape = [left_rank] + new_shape[1].tolist() + [right_rank]
      new_shape = res_batch_shape + new_shape
      res_cores.append(core.reshape(new_shape))

    if are_tt_matrix_inputs:
      res = TTMatrix(res_cores)
    else:
      res = TT(res_cores)

    if is_fusing:
      res = WrappedTT(res, args, tt_einsum_)
    return res

  return new_func


def compile_cumulative(tt_einsum: TTEinsum) -> Callable:
  def new_func(*args):
    are_tt_matrix_inputs = args[0].is_tt_matrix
    tt_einsum_ = tt_einsum.resolve_i_or_ij(are_tt_matrix_inputs)

    is_fusing = any([isinstance(tt, WrappedTT) for tt in args])
    if is_fusing:
      # Have to use a different name to make upper level tt_einsum visible.
      tt_einsum_, args = _fuse_tt_einsums(tt_einsum_, args)
    einsum = tt_einsum_.to_vanilla_einsum()

    res = jnp.ones([1] * len(args), args[0].tt_cores[0].dtype)
    res_list = []
    for core_idx in range(len(args[0].tt_cores)):
      curr_tensors = [a.tt_cores[core_idx] for a in args]
      curr_tensors.append(res)
      res = oe.contract(einsum, *curr_tensors, backend='jax',
                        optimize='optimal')
      res_list.append(res)
    return res_list

  return new_func


def fuse(func):
  """Fuse a composite function to make it faster.

  Example:
    def f(a, b, c):
      # Computes
      # <a * b, c> =
      #   sum_{i_1, ..., i_d} a[i_1, ..., i_d] b[i_1, ..., i_d] c[i_1, ..., i_d]
      return ttax.flat_inner(a * b, c)

  Function `f` can be suboptimal for some inputs. For example, if `a` and `b`
  are of large TT-rank, and `c` is of low TT-rank, implementing the same
  operation as
    ttax.flat_inner(a * c, b)
  would be much more efficient.

  `fuse` automates such optimizations. You can build an optimal implementation
  of this function for any inputs by doing
    faster_f = ttax.fuse(f)
  Finally, don't forget that in Jax to get good speed you need to wrap you
  highest level function in jit, e.g.
    faster_f = jax.jit(faster_f)
  Now, by using `faster_f(a, b, c)` instead of `f(a, b, c)` you can achieve
  a much faster cumulative time for any inputs.
  """

  def _func(*args):
    wrapped_args = [WrappedTT(arg) for arg in args]
    res = func(*wrapped_args)
    if isinstance(res, WrappedTT):
      res = res.tt
    return res

  return _func
