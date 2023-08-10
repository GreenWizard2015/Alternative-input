#!/usr/bin/env python
# -*- coding: utf-8 -*-.
# TODO: rewrite this file to support the command line parameters and the W&B logging
import numpy as np
from Core.CDatasetLoader import CDatasetLoader
from Core.CTestLoader import CTestLoader
import os
from collections import defaultdict
import time
from Core.CDemoModel import CDemoModel
import tensorflow as tf

BATCH_PER_EPOCH = 15_000
EVAL_EVERY = 1_000
# setup numpy printing options for debugging
np.set_printoptions(precision=4, threshold=7777, suppress=True, linewidth=120)
folder = os.path.dirname(__file__)
folder = os.path.join(folder, 'Data')

trainDataset = CDatasetLoader(
  os.path.join(folder, 'train.npz'), 
  batch_size=16, batchPerEpoch=BATCH_PER_EPOCH,
  samplerArgs=dict(
    defaults=dict(
      timesteps=5,
      stepsSampling={'max frames': 5, 'include last': False},
      # augmentations
      pointsDropout=0.2, pointsNoise=0.015,
      eyesDropout=0., eyesAdditiveNoise=0.1, brightnessFactor=3.0, lightBlobFactor=3.0
    ),
  )
)

testDataset = CTestLoader(os.path.join(folder, 'test'))

model = CDemoModel(
  timesteps=5,
#   weights=dict(folder=folder), 
  trainable=True,
)
model._model.summary()

def evaluate():
  T = time.time()

  # evaluate the model on the val dataset
  lossPerSample = {'loss': [], 'pos': []}
  predV = []
  predDist = []
  Y = []
  for batchId in range(len(testDataset)):
    _, (y,) = batch = testDataset[batchId]
    loss, predP, dist = model.eval(batch)
    predV.append(predP)
    predDist.append(dist)
    Y.append(y[:, -1, 0])
    for l, pos in zip(loss, y[:, -1]):
      lossPerSample['loss'].append(l)
      lossPerSample['pos'].append(pos[0])
      continue
    continue
  
  def unbatch(qu):
    qu = np.array(qu)
    qu = qu.transpose(1, 0, *np.arange(2, len(qu.shape)))
    return qu.reshape((qu.shape[0], -1, qu.shape[-1]))

  if True:
    Y = unbatch(Y).reshape((-1, 2))
    predV = unbatch(predV).reshape((-1, 2))
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.figure(figsize=(8, 8))
    plt.plot(Y[:, 0], Y[:, 1], 'o', markersize=1)
    plt.plot(predV[:, 0], predV[:, 1], 'o', markersize=1)
    for i in range(5):
      d = i * 0.1
      plt.gca().add_patch(
        patches.Rectangle(
          (d,d), 1-2*d, 1-2*d,
          linewidth=1,edgecolor='r',facecolor='none'
        )
      )

    plt.savefig(os.path.join(folder, 'pred.png'))
    plt.clf()
    plt.close()
    pass
  
  loss = np.mean(lossPerSample['loss'])
  print('Test | %.2f sec | Loss: %.5f. Distance: %.5f' % (time.time() - T, loss, np.mean(predDist)))
  return loss

bestLoss = evaluate()
for epoch in range(10):
  timesteps = 5
  batchSize = 8*8
  print('Epoch: %d, timesteps: %d, batchSize: %d' % (epoch, timesteps, batchSize))
  history = defaultdict(list)
  T = time.time()
  evaluated = False
  for batchId in range(len(trainDataset)):
    stats = model.fit(trainDataset.sample(
      N=batchSize,
      timesteps=timesteps
    ))
    history['time'].append(stats['time'])
    for k in stats['losses'].keys():
      history[k].append(stats['losses'][k])
      
    if (0 == (batchId % EVAL_EVERY)) and (0 < batchId):
      model.save(folder, postfix='latest')
      statsStr = ', '.join(['%s: %.5f' % (k, np.mean(v[-EVAL_EVERY:])) for k, v in history.items()])
      print(
        '%d epoch | %d/%d | %s (%.2f sec)' % 
        (epoch, batchId, len(trainDataset), statsStr, time.time() - T)
      )
      
      testLoss = evaluate()
      evaluated = True
      if testLoss < bestLoss:
        print('Improved %.5f => %.5f' % (bestLoss, testLoss))
        bestLoss = testLoss
        model.save(folder)
      T = time.time()
      pass
    continue
  
  model.save(folder, postfix='latest')
  if not evaluated:
    testLoss = evaluate()
    if testLoss < bestLoss:
      print('Improved %.5f => %.5f' % (bestLoss, testLoss))
      bestLoss = testLoss
      model.save(folder)
    pass

  statsStr = ', '.join(['%s: %.5f' % (k, np.mean(v)) for k, v in history.items()])
  print('%d epoch | %s' % (epoch, statsStr, ))
  trainDataset.on_epoch_end()
  continue