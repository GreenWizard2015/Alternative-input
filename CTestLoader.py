import tensorflow as tf
import numpy as np
import Utils
import os
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
import math

class CTestLoader(tf.keras.utils.Sequence):
  def __init__(self, 
    filename='test.npz',
    batch_size=64,
    padding=True
  ):
    with np.load(filename) as f:
      self._dataset = list(f.values())
      
    self.batch_size = batch_size
    N = len(self._dataset[0])
    self._indexes = np.arange(batch_size * math.ceil(N / batch_size)) % N
    if not padding:
      self._indexes = self._indexes[:N]
    self.on_epoch_end()
    return
  
  def on_epoch_end(self):
    return

  def __len__(self):
    return math.ceil(len(self._indexes) / self.batch_size)
  
  def __getitem__(self, idx):
    indices = self._indexes[idx*self.batch_size:(idx + 1)*self.batch_size]
    res = [x[indices] for x in self._dataset]
    *X, Y = res
    return(X, (Y, ))

if __name__ == '__main__':
  import cv2
  folder = os.path.dirname(__file__)
  ds = CTestLoader(os.path.join(folder, 'test.npz'))
  print(len(ds))
  batch = ds[0]
  for x in batch:
    for xx in x:
      print(xx.shape)
    print()
  pass