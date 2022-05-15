import tensorflow as tf
import numpy as np
import Utils
import math
import os
import random

def _dict2arrays(d):
  if isinstance(d, list): return np.array(d, np.uint32)
  return [_dict2arrays(x) for x in d.values()]

class CDatasetLoaderBalanced(tf.keras.utils.Sequence):
  def __init__(self, 
    folder, batch_size, pointsDropout=0.0, eyeDropout=0.0,
    goalHashScale=30, batchPerEpoch=None
  ):
    self.batch_size = batch_size
    self._pointsDropout = pointsDropout
    self._eyeDropout = eyeDropout
    self._dataset = Utils.datasetFromFolder(folder)
    ##################
    samplesByGoals = {}
    def getBucket(*hashes):
      s = samplesByGoals
      for h in hashes[:-1]:
        if not(h in s): s[h] = {}
        s = s[h]
        continue
      h = hashes[-1]
      if not(h in s): s[h] = []
      return s[h]
    
    for index, goal in enumerate(self._dataset['goal']):
      goal = np.clip(goal, 0.0, 1.0)
      x = np.floor((goalHashScale / 2.0) - (goal * goalHashScale))
      hashA = np.abs(x).max() # lvl
      hashB = 0 < x[0] # left/right
      hashC = 0 < x[1] # top/bottom
      hashD = np.abs(x).max() - np.abs(x).min()
      getBucket(hashA, hashB, hashC, hashD).append(index)
      continue
    samplesByGoals = _dict2arrays(samplesByGoals)
    self._samplesByGoals = samplesByGoals
    ###################
    N = len(samplesByGoals)
    self._indexes = np.arange(batch_size * batchPerEpoch) % N
    
    self.on_epoch_end()
    return
  
  def on_epoch_end(self):
    np.random.shuffle(self._indexes)
    return

  def __len__(self):
    return math.ceil(len(self._indexes) / self.batch_size)

  def __getitem__(self, idx):
    buckets = self._indexes[idx*self.batch_size:(idx + 1)*self.batch_size]
    indexes = []
    for ind in buckets:
      subInd = samples = self._samplesByGoals[ind]
      while np.iterable(samples):
        subInd = samples = random.choice(samples)
      indexes.append(subInd)
      continue

    samples = [
      {k: v[i] for k, v in self._dataset.items()}
      for i in indexes
    ]
    
    X = Utils.samples2inputs(samples, dropout=self._pointsDropout)
    if 0.0 < self._eyeDropout:
      imgA = X[1]
      imgB = X[2]
      
      mask = np.random.random((len(samples),)) < self._eyeDropout
      maskA = 0.5 < np.random.random((len(samples),))
      maskB = np.logical_not(maskA)
      
      imgA[np.where(np.logical_and(mask, maskA))] = 0.0
      imgB[np.where(np.logical_and(mask, maskB))] = 0.0
      
      X = (X[0], imgA, imgB) 
      
    return(
      X, 
      ( np.array([x['goal'] for x in samples], np.float32), )
    )
  
if __name__ == '__main__':
  import cv2
  folder = os.path.dirname(__file__)
  ds = CDatasetLoaderBalanced(os.path.join(folder, 'Dataset'), batch_size=16, pointsDropout=0.0, batchPerEpoch=1)
  print(len(ds))
  batchX, batchY = ds[0]
  print(batchX[0].shape)
  img = batchX[1][0]
  cv2.imshow('L', cv2.resize(img, (256, 256)))
  
  pass