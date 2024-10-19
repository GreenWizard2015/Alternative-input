import tensorflow as tf
import numpy as np
import os, glob
from functools import lru_cache

class CTestLoader(tf.keras.utils.Sequence):
  def __init__(self, testFolder):
    self._batchesNpz = [
      f for f in glob.glob(os.path.join(testFolder, 'test-*.npz'))
    ]
    self.on_epoch_end()
    return

  @lru_cache(maxsize=1)
  def parametersIDs(self):
    batch, _ = self[0]
    userId = batch['userId'][0, 0, 0]
    placeId = batch['placeId'][0, 0, 0]
    screenId = batch['screenId'][0, 0, 0]
    return placeId, userId, screenId
      
  def on_epoch_end(self):
    return

  def __len__(self):
    return len(self._batchesNpz)
  
  def __getitem__(self, idx):
    with np.load(self._batchesNpz[idx]) as res:
      res = {k: v for k, v in res.items()}
      
    Y = res.pop('y')
    return(res, (Y, ))