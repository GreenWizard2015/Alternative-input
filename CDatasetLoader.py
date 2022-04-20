import tensorflow as tf
import numpy as np
import Utils
import math
import glob
import os
from collections import defaultdict

class CDatasetLoader(tf.keras.utils.Sequence):
  def __init__(self, folder, batch_size, pointsDropout=0.0):
    self.batch_size = batch_size
    self._pointsDropout = pointsDropout
    #######
    dataset = defaultdict(list) # eat memory but ok
    for fn in glob.iglob(os.path.join(folder, '*.npz')):
      with np.load(fn) as data:
        for k, v in data.items():
          dataset[k].append(v)
      continue
    self._dataset = {k: np.concatenate(v, axis=0) for k, v in dataset.items()}
    N = len(list(self._dataset.values())[0])
    self._indexes = np.arange(math.ceil(N / float(batch_size)) * batch_size) % N
    
    self.on_epoch_end()
    return
  
  def on_epoch_end(self):
    np.random.shuffle(self._indexes)
    return

  def __len__(self):
    return math.ceil(len(self._indexes) / self.batch_size)

  def __getitem__(self, idx):
    indexes = self._indexes[idx*self.batch_size:(idx + 1)*self.batch_size]
    samples = [
      {k: v[i] for k, v in self._dataset.items()}
      for i in indexes
    ]
    return(
      Utils.samples2inputs(samples, dropout=self._pointsDropout), 
      ( np.array([x['goal'] for x in samples], np.float32), )
    )
    
if __name__ == '__main__':
  import cv2
  folder = os.path.dirname(__file__)
  ds = CDatasetLoader(os.path.join(folder, 'Dataset'), batch_size=1)
  batchX, batchY = ds[0]
  print(batchX)
  print(batchY[0].shape)
  cv2.imshow('L', cv2.resize(batchX[1][0], (256, 256)))
  
  img = batchX[2][0]*255
#   img = cv2.convertScaleAbs(img, alpha=2, beta=0.0)
  img = cv2.Canny(img.astype(np.uint8), 100,200)
  cv2.imshow('R', cv2.resize(img, (256, 256)))
  cv2.waitKey()
  pass