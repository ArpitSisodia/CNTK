import sys
import os
from datetime import datetime
import numpy as np
from cntk import Trainer, Axis, device, combine
from cntk.layers.blocks import Stabilizer, _initializer_for,  _INFERRED, Parameter, Placeholder
from cntk.layers import Recurrence, Convolution
from cntk.ops import input, cross_entropy_with_softmax, classification_error, sequence, reduce_sum, \
    parameter, times, element_times, past_value, plus, placeholder, reshape, constant, sigmoid, convolution, tanh, times_transpose, greater, cosine_distance, element_divide, element_select, exp, future_value, past_value
from cntk.internal import _as_tuple, sanitize_input
from cntk.initializer import uniform, glorot_uniform

def cosine_similarity(src, tgt, name=''):
  """
  Compute the cosine similarity of two squences.
  Src is a sequence of length 1
  Tag is a sequence of lenght >=1
  """
  src_br = sequence.broadcast_as(src, tgt, name='src_broadcast')
  sim = cosine_distance(src_br, tgt, name)
  return sim

def project_cosine_sim(att_dim, init = glorot_uniform(), name=''):
  """
  Compute the project cosine similarity of two input sequences, where each of the input will be projected to a new dimention space (att_dim) via Wi/Wm
  """
  Wi = Parameter(_INFERRED + tuple((att_dim,)), init = init, name='Wi')
  Wm = Parameter(_INFERRED + tuple((att_dim,)), init = init, name='Wm')
  status = placeholder(name='status')
  memory = placeholder(name='memory')
  projected_status = times(status, Wi, name = 'projected_status')
  projected_memory = times(memory, Wm, name = 'projected_memory')
  sim = cosine_similarity(projected_status, projected_memory, name= name+ '_sim')
  return sequence.softmax(sim, name = name)

def termination_gate(init = glorot_uniform(), name=''):
  Wt = Parameter( _INFERRED + tuple((1,)), init = init, name='Wt')
  status = placeholder(name='status')
  return sigmoid(times(status, Wt), name=name)
