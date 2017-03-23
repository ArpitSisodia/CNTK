import os
import numpy as np
from datetime import datetime
import math
from .wordvocab import *

class logger:
  __name=''
  __logfile=''

  @staticmethod
  def init(name=''):
    if not os.path.exists("model"):
      os.mkdir("model")
    if not os.path.exists("log"):
      os.mkdir("log")
    if name=='' or name is None:
      logger.__name='train'
    logger.__logfile = 'log/{}_{}.log'.format(name, datetime.now().strftime("%m-%d_%H.%M.%S"))
    if os.path.exists(logger.__logfile):
      os.remove(logger.__logfile)
    print('Log with log file: {0}'.format(logger.__logfile))

  @staticmethod
  def log(message, toconsole=True):
    if logger.__logfile == '' or logger.__logfile is None:
      logger.init()
    if toconsole:
      print(message)
    with open(logger.__logfile, 'a') as logf:
      logf.write("{}| {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message))

class uniform_initializer:
  def __init__(self, scale=1, bias=0, seed=0):
    self.seed = seed
    self.scale = scale
    self.bias = bias
    np.random.seed(self.seed)

  def reset(self):
    np.random.seed(self.seed)

  def next(self, size=None):
    return np.random.uniform(0, 1, size)*self.scale + self.bias

def create_random_matrix(rows, columns):
  scale = math.sqrt(6/(rows+columns))*2
  rand = uniform_initializer(scale, -scale/2)
  embedding = [None]*rows
  for i in range(rows):
    embedding[i] = np.array(rand.next(columns), dtype=np.float32)
  return np.ndarray((rows, columns), dtype=np.float32, buffer=np.array(embedding))

def load_embedding(embedding_path, vocab_path, dim, init=None):
  entity_vocab, word_vocab = Vocabulary.load_vocab(vocab_path)
  vocab_dim = len(entity_vocab) + len(word_vocab) + 1
  entity_size = len(entity_vocab)
  item_embedding = [None]*vocab_dim
  with open(embedding_path, 'r') as embedding:
    for line in embedding.readlines():
      line = line.strip('\n')
      item = line.split(' ')
      if item[0] in word_vocab:
        item_embedding[word_vocab[item[0]].id + entity_size + 1] = np.array(item[1:], dtype="|S").astype(np.float32)
  if init != None:
    init.reset()

  for i in range(vocab_dim):
    if item_embedding[i] is None:
      if init:
        item_embedding[i] = np.array(init.next(dim), dtype=np.float32)
      else:
        item_embedding[i] = np.array([0]*dim, dtype=np.float32)
  return np.ndarray((vocab_dim, dim), dtype=np.float32, buffer=np.array(item_embedding))
