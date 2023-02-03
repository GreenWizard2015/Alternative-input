import Core.Utils as Utils
import os
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
import numpy as np

class CDatasetLoader:
  def __init__(self,
    folderOrFile, batch_size, batchPerEpoch,
    samplerArgs
  ):
    self.batch_size = batch_size
    self._batchPerEpoch = batchPerEpoch
    
    self._dataset = CDataSampler(
      CSamplesStorage(),
      **samplerArgs
    )
    if os.path.isfile(folderOrFile):
      self._dataset.addBlock(np.load(folderOrFile))
    else:
      self._dataset.addBlock(Utils.datasetFrom(folderOrFile))
    self.on_epoch_end()
    return
  
  def on_epoch_end(self):
    return

  def __len__(self):
    return self._batchPerEpoch
  
  def __getitem__(self, idx):
    return self._dataset.sample(self.batch_size)
  
  def sample(self, **kwargs):
    if not('N' in kwargs):
      kwargs['N'] = self.batch_size
    return self._dataset.sample(**kwargs)

  @property
  def contexts(self):
    return self._dataset.contexts
  
if __name__ == '__main__':
  import cv2
  folder = os.path.dirname(__file__)
  ds = CDatasetLoader(
    os.path.join(folder, 'Dataset'), batch_size=16, 
    pointsDropout=0.0, batchPerEpoch=1, steps=5
  )
  print(len(ds))
  batchX, batchY = ds[0]
  print(batchY[0].shape)
  print(batchX[1].shape)
  img = batchX[1][0, 0]
  cv2.imshow('L', cv2.resize(img, (256, 256)))
  cv2.waitKey()
  pass