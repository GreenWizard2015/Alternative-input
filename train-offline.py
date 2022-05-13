#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
from CDatasetLoader import CDatasetLoader
from CDatasetLoaderBalanced import CDatasetLoaderBalanced
from Core.CFakeModel import CFakeModel
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from CCoordsEncodingLayer import CCoordsEncodingLayer

np.set_printoptions(precision=4, threshold=7777, suppress=True, linewidth=120)
folder = os.path.dirname(__file__)
model_h5 = os.path.join(folder, 'autoregressive.h5')

model = CFakeModel(model='autoregressive')#, weights=model_h5, trainable=True)
coefsHistory = defaultdict(list)
def debug():
  def debugL(l):
    if hasattr(l, 'layers'):
      for ll in l.layers:
        debugL(ll)
      return
    if isinstance(l, CCoordsEncodingLayer):
      coefs = l.coefs.numpy()
      coefsHistory[l.name].append(coefs)
      l.debug()
      #l._fussion.summary()
      pass
    return
  
  for nm in ['_model', '_encoder', '_decoder']:
    if hasattr(model, nm):
      print(nm)
      debugL(getattr(model, nm))
    continue
  return

debug()
# exit()
trainDataset = CDatasetLoaderBalanced(
  os.path.join(folder, 'Dataset'), 
  batch_size=16, pointsDropout=0.25, eyeDropout=0.25,
  batchMult=512*8*4
)
batchN = len(trainDataset)

losses = ['loss']
for epoch in range(5):
  history = defaultdict(list)
  T = time.time()
  for batchId in range(batchN):
    stats = model.fit(trainDataset[batchId])
    for k in ['time'] + losses:
      history[k].append(stats[k])
      
    if (0 == (batchId % 100)) and (0 < batchId):
      lossStr = ', '.join(['%s: %.5f' % (k, np.mean(history[k][-100:])) for k in losses])
      print(
        '%d epoch | %d/%d | %s (%d ms, %.2f sec)' % 
        (epoch, batchId, batchN, lossStr, np.mean(history['time'][-100:]), time.time() - T)
      )
      debug()
      model.save(model_h5)
      T = time.time()
    continue
  
  lossStr = ', '.join(['%s: %.5f' % (k, np.mean(history[k])) for k in losses])
  print(
    '%d epoch | %s' % 
    (epoch, lossStr, )
  )
  trainDataset.on_epoch_end()
  model.save(model_h5)
  continue

fig, splt = plt.subplots(len(coefsHistory), 2, sharex=True)
# splt = [splt]
for i, (k, values) in enumerate(coefsHistory.items()):
  byCoef = np.transpose([x.ravel() for x in values])
  for x in byCoef:
    splt[i][0].plot(np.arange(x.size), x)
    mx = 1e-8 + x.max()
    x /= mx
    splt[i][1].plot(np.arange(x.size), x)
    splt[i][1].set_ylim(0, 1.05)
  continue
plt.show()