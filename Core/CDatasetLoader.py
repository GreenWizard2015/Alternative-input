import Core.Utils as Utils
import os, glob
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
import numpy as np
import tensorflow as tf

class CDatasetLoader:
  def __init__(self, folder, samplerArgs, stats):
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
    print('Total samples: %d' % sum(ds.totalSamples for ds in self._datasets))

    self._batchSize = samplerArgs.get('batch_size', 16)
    return
  
  def on_epoch_start(self):
    for ds in self._datasets:
      ds.reset()
      continue
    return
  
  def on_epoch_end(self):
    return

  def __len__(self):
    return sum(len(ds) for ds in self._datasets)
  
  def sample(self, **kwargs):
    batchSize = kwargs.get('batch_size', self._batchSize)
    N = batchSize // len(self._datasets)
    if N < 1: N = 1
    resX = []
    resY = []
    totalSamples = 0
    while totalSamples < batchSize:
      datasetIdx = np.random.randint(0, len(self._datasets))
      dataset = self._datasets[datasetIdx]
      x, (y, ) = dataset.sample(N=min(N, batchSize - totalSamples), **kwargs)
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