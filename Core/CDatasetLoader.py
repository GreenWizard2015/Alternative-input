import Core.Utils as Utils
import os
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
import numpy as np

class CDatasetLoader:
  def __init__(self, folderOrFile, samplerArgs):
    self._dataset = CDataSampler(
      CSamplesStorage(),
      **samplerArgs
    )
    if os.path.isfile(folderOrFile):
      self._dataset.addBlock(np.load(folderOrFile))
    else:
      self._dataset.addBlock(Utils.datasetFrom(folderOrFile))
    return
  
  def on_epoch_start(self):
    self._dataset.reset()
    return
  
  def on_epoch_end(self):
    return

  def __len__(self):
    return len(self._dataset)
  
  def __getitem__(self, idx):
    return self._dataset.sample()
  
  def sample(self, **kwargs):
    return self._dataset.sample(**kwargs)

  @property
  def contexts(self):
    return self._dataset.contexts
  
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