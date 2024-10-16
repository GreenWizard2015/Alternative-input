import Core.Utils as Utils
import os, glob
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
import numpy as np
import tensorflow as tf
from enum import Enum

class ESampling(Enum):
  AS_IS = 'as_is'
  UNIFORM = 'uniform'
  
class CDatasetLoader:
  def __init__(self, folder, samplerArgs, sampling, stats):
    # recursively find all 'train.npz' files
    trainFiles = glob.glob(os.path.join(folder, '**', 'train.npz'), recursive=True)
    if 0 == len(trainFiles):
      raise Exception('No training dataset found in "%s"' % (folder, ))
      exit(1)
    
    print('Found %d training datasets' % (len(trainFiles), ))

    self._datasets = []
    for trainFile in trainFiles:
      print('Loading %s' % (trainFile, ))
      # extract the placeId, userId, and screenId
      parts = os.path.split(trainFile)[0].split(os.path.sep)
      placeId, userId, screenId = parts[-3], parts[-2], parts[-1]
      ds = CDataSampler(
        CSamplesStorage(
          placeId=stats['placeId'].index(placeId),
          userId=stats['userId'].index(userId),
          screenId=stats['screenId'].index('%s/%s' % (placeId, screenId))
        ),
        **samplerArgs
      )
      ds.addBlock(Utils.datasetFrom(trainFile))
      self._datasets.append(ds)
      continue
    
    print('Loaded %d datasets' % (len(self._datasets), ))
    validSamples = {
      i: len(ds.validSamples())
      for i, ds in enumerate(self._datasets)
    }
    dtype = np.uint8 if len(self._datasets) < 256 else np.uint32
    # create an array of dataset indices to sample from
    sampling = ESampling(sampling)
    if ESampling.AS_IS == sampling: # just concatenate the indices
      self._indices = np.concatenate([
        np.full((v, ), k, dtype=dtype) # id of the dataset
        for k, v in validSamples.items()
      ])
    if ESampling.UNIFORM == sampling:
      maxSize = max(validSamples.values())
      chunks = []
      for k, size in validSamples.items():
        # all datasets should have the same number of samples represented in the indices
        # so that the sampling is uniform
        chunk = np.full((maxSize, ), k, dtype=dtype)
        chunks.append(chunk)
        continue
      self._indices = np.concatenate(chunks)
      
    self._currentId = 0

    self._batchSize = samplerArgs.get('batch_size', 16)
    return
  
  def on_epoch_start(self):
    # shuffle the indices at the beginning of each epoch
    self._currentId = 0
    np.random.shuffle(self._indices)
    # reset all datasets
    for ds in self._datasets:
      ds.reset()
      continue
    return
  
  def on_epoch_end(self):
    return

  def __len__(self):
    return 1 + (len(self._indices) // self._batchSize)
  
  def _getBatchStats(self, batchSize):
    batch = self._indices[self._currentId:self._currentId + batchSize]
    self._currentId = (self._currentId + batchSize) % len(self._indices)
    while len(batch) < batchSize:
      batch2 = self._indices[:batchSize - len(batch)]
      self._currentId = (self._currentId + len(batch2)) % len(self._indices)
      batch = np.concatenate([batch, batch2], axis=0) # concatenate the two batches
      continue

    assert len(batch) == batchSize, 'Invalid batch size: %d != %d' % (len(batch), batchSize)
    datasetIds, counts = np.unique(batch, return_counts=True)
    return datasetIds, counts

  def sample(self, **kwargs):
    batchSize = kwargs.get('batch_size', self._batchSize)
    resX = []
    resY = []
    totalSamples = 0
    # find the datasets ids and the number of samples to take from each dataset
    datasetIds, counts = self._getBatchStats(batchSize)
    for datasetId, N in zip(datasetIds, counts):
      dataset = self._datasets[datasetId]
      x, (y, ) = dataset.sample(N=N, **kwargs)
      assert len(y) == N, 'Invalid number of samples: %d != %d' % (len(y), N)
      resX.append(x)
      resY.append(y)
      totalSamples += len(y)
      continue
    
    resY = np.concatenate(resY, axis=0)
    assert resY.shape[0] == batchSize, 'Invalid shape: %s' % (resY.shape, )
    assert resY.shape[-1] == 2, 'Invalid shape: %s' % (resY.shape, )
    assert len(resY.shape) == 4, 'Invalid shape: %s' % (resY.shape, )

    # sampled data has 'clean' and 'augmented' keys
    output = {}
    for nm in ['clean', 'augmented']:
      keys = resX[0][nm].keys()
      output[nm] = {k: tf.concat([x[nm][k] for x in resX], axis=0) for k in keys}
      continue
    return output, (resY,)
  
if __name__ == '__main__':
  import cv2
  folder = os.path.dirname(__file__)
  ds = CDatasetLoader(
    os.path.join(folder, 'Dataset'), batch_size=16, 
    batchPerEpoch=1, steps=5
  )
  print(len(ds))
  batchX, batchY = ds[0]
  print(batchY[0].shape)
  print(batchX[1].shape)
  img = batchX[1][0, 0]
  cv2.imshow('L', cv2.resize(img, (256, 256)))
  cv2.waitKey()
  pass