import tensorflow as tf
import numpy as np
from CDatasetLoader import CDatasetLoader
from Core.CFakeModel import CFakeModel
import os
from collections import defaultdict
import time

model = CFakeModel(model='NerfLike', depth=5)
folder = os.path.dirname(__file__)
ds = CDatasetLoader(os.path.join(folder, 'Dataset'), batch_size=16, pointsDropout=0.25, eyeDropout=0.25)
batchN = len(ds)
# model.sampleNerf([x[:3] for x in ds[0][0] ])
# exit()
model_h5 = os.path.join(folder, 'nerf.h5')
for epoch in range(5):
  history = defaultdict(list)
  T = time.time()
  for batchId in range(batchN):
    stats = model.fit(ds[batchId])
    for k in ['loss', 'time']:
      history[k].append(stats[k])
      
    if (0 == (batchId % 100)) and (0 < batchId):
      print(
        '%d epoch | %d/%d | %.5f (%d ms, %.2f sec)' % 
        (epoch, batchId, batchN, np.mean(history['loss'][-100:]), np.mean(history['time'][-100:]), time.time() - T)
      )
      # model.debug([x[:1] for x in ds[0][0] ])
      T = time.time()
    continue
  
  print('%d epoch | %.5f (%d ms)' % (epoch, np.mean(history['loss']), np.mean(history['time'])))
  ds.on_epoch_end()
#   model.save(model_h5)
  continue