import tensorflow as tf
import numpy as np
from CDatasetLoader import CDatasetLoader
from Core.CFakeModel import CFakeModel
import os
from collections import defaultdict
import time

folder = os.path.dirname(__file__)
model_h5 = os.path.join(folder, 'simple.h5')

model = CFakeModel(model='NerfLike')#, weights=model_h5)

ds = CDatasetLoader(os.path.join(folder, 'Dataset'), batch_size=16, pointsDropout=0.25, eyeDropout=0.25)
batchN = len(ds)

model.save(model_h5)
losses = ['loss']
for epoch in range(1):
  history = defaultdict(list)
  T = time.time()
  for batchId in range(batchN):
    stats = model.fit(ds[batchId])
    for k in ['time'] + losses:
      history[k].append(stats[k])
      
    if (0 == (batchId % 100)) and (0 < batchId):
      lossStr = ', '.join(['%s: %.5f' % (k, np.mean(history[k][-100:])) for k in losses])
      print(
        '%d epoch | %d/%d | %s (%d ms, %.2f sec)' % 
        (epoch, batchId, batchN, lossStr, np.mean(history['time'][-100:]), time.time() - T)
      )
      T = time.time()
    continue
  
  lossStr = ', '.join(['%s: %.5f' % (k, np.mean(history[k])) for k in losses])
  print(
    '%d epoch | %s' % 
    (epoch, lossStr, )
  )
  ds.on_epoch_end()
  model.save(model_h5)
  continue

m = model._model

def debug(l):
  if hasattr(l, 'layers'):
    for ll in l.layers:
      debug(ll)
  if 'c_coords_encoding_layer' in l.name:
    print(l.coefs.numpy())
    print('shifts', l.shifts.numpy())
    print('.......')
    pass
  return

for l in m.layers:
  debug(l)
  continue
