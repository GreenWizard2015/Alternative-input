import tensorflow as tf
import numpy as np
import os, glob

class CTestLoader(tf.keras.utils.Sequence):
  def __init__(self, testFolder, contextsStartIndex):
    self._batchesNpz = [
      f for f in glob.glob(os.path.join(testFolder, 'test-*.npz'))
    ]
    self._contextsStartIndex = np.array(contextsStartIndex, dtype=np.int32)[None, None]
    self.on_epoch_end()
    return
  
  def on_epoch_end(self):
    return

  def __len__(self):
    return len(self._batchesNpz)
  
  def __getitem__(self, idx):
    with np.load(self._batchesNpz[idx]) as res:
      res = {k: v for k, v in res.items()}
    
    res['ContextID'] = self._contextsStartIndex + res['ContextID']
    Y = res.pop('y')
    return(res, (Y, ))

if __name__ == '__main__':
  folder = os.path.dirname(__file__)
  ds = CTestLoader(os.path.join(folder, 'test'))
  print(len(ds))
  batch, (y,) = ds[0]
  for k, v in batch.items():
    print(k, v.shape)
  print()
  print(y.shape)
  pass