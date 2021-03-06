import tensorflow as tf
import numpy as np
import Utils
import math
import os

class CDatasetLoader(tf.keras.utils.Sequence):
  def __init__(self, folder, batch_size, pointsDropout=0.0, eyeDropout=0.0):
    self.batch_size = batch_size
    self._pointsDropout = pointsDropout
    self._eyeDropout = eyeDropout
    self._dataset = Utils.datasetFromFolder(folder)
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
  ds = CDatasetLoader(os.path.join(folder, 'Dataset'), batch_size=16, pointsDropout=0.0)
  print(len(ds))
  batchX, batchY = ds[0]
  print(batchX[0].shape)
  img = batchX[1][0]
  cv2.imshow('L', cv2.resize(img, (256, 256)))
  
#   kernels = createSobelsConv(1 + 2*np.arange(1, 8))
#   for i, k in enumerate(kernels):
#     filtered = cv2.filter2D(img, kernel=k, ddepth=-1)
#     print(i, filtered.min(), filtered.max())
#     cv2.imshow('L%d_%d'% (0, i), cv2.resize(np.clip(filtered, 0, 1.), (256, 256)))
#     continue
#   cv2.waitKey()
#   
  import networks
  transform = networks.EyeEnricher()
  images = transform([batchX[1]]).numpy()
  for i in range(images.shape[-1]):
    img = images[0, ..., i, None]
    cv2.imshow('R%d' % i, cv2.resize(img, (256, 256)))
    continue
  cv2.waitKey()
#   x = batchX[0]
#   encoded = networks.PointsEnricher(x.shape[1:])([x]).numpy()
#   np.set_printoptions(precision=2, threshold=77777777, linewidth=545556)
#   print(encoded[0])
#   import matplotlib.pyplot as plt
#   plt.hist(encoded.ravel())
#   plt.show()
  pass