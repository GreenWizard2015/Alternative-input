import tensorflow as tf
import numpy as np
from CDatasetLoader import CDatasetLoader
from Core.CFakeModel import CFakeModel
import os
from collections import defaultdict
import time

model = CFakeModel()
folder = os.path.dirname(__file__)
ds = CDatasetLoader(os.path.join(folder, 'Dataset'), batch_size=16, pointsDropout=0.5)
batchN = len(ds)

for epoch in range(1):
  history = defaultdict(list)
  T = time.time()
  for batchId in range(batchN):
    stats = model.fit(ds[batchId])
    for k in ['loss', 'time']:
      history[k].append(stats[k])
      
    if 0 == (batchId % 100):
      print(
        '%d epoch | %d/%d | %.5f (%d ms, %.2f sec)' % 
        (epoch, batchId, batchN, np.mean(history['loss']), np.mean(history['time']), time.time() - T)
      )
      history = defaultdict(list)
      model.debug([x[:1] for x in ds[0][0] ])
      T = time.time()
    continue
  print('%d epoch | %d/%d | %.5f (%d ms)' % (epoch, batchId, batchN, np.mean(history['loss']), np.mean(history['time'])))
  ds.on_epoch_end()
  continue
