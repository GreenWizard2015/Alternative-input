import tensorflow as tf
import numpy as np
import os

class CPretrainLoader(tf.keras.utils.Sequence):
  def __init__(self, filename, contextsStartIndex, batch_size):
    self._batchSize = batch_size
    with np.load(filename) as res:
      self._data = {k: v for k, v in res.items()}
    
    lastDim = self._data['ContextID'].shape[-1]
    dims = len(self._data['ContextID'].shape)
    assert len(contextsStartIndex) == lastDim
    shp = [1] * (dims - 1) + [lastDim]
    self._data['ContextID'] += np.array(contextsStartIndex, dtype=np.int32).reshape(shp)

    self.on_epoch_end()
    return
  
  def on_epoch_end(self):
    return

  def __len__(self):
    return len(self._data['y']) // self._batchSize
  
  def __getitem__(self, idx):
    idx = slice(idx * self._batchSize, (idx + 1) * self._batchSize)
    res = {k: v[idx] for k, v in self._data.items()}
    Y = res.pop('y')
    return(res, (Y, ))

if __name__ == '__main__':
  folder = os.path.dirname(__file__)
  ds = CPretrainLoader(os.path.join(folder, 'Data', 'test'))
  print(len(ds))
  batch, (y,) = ds[0]
  for k, v in batch.items():
    print(k, v.shape)
  print()
  print(y.shape)
  pass