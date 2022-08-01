#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
from CDatasetLoader import CDatasetLoader
from CTestLoader import CTestLoader
from Core.CFakeModel import CFakeModel
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import networks
import Utils

np.set_printoptions(precision=4, threshold=7777, suppress=True, linewidth=120)
folder = os.path.dirname(__file__)

model = CFakeModel(
  model='simple', depth=15,
  F2LArgs=dict(steps=5),
  weights=dict(folder=folder), 
  trainable=True
)
trainDataset = CDatasetLoader(
  os.path.join(folder, 'Dataset'), 
  batch_size=16, batchPerEpoch=16 * 1024,
  samplerArgs=dict(
    defaults=dict(
      timesteps=model.timesteps,
      return_tensors=True,
      stepsSampling={'max frames': 2, 'include last': True},
      # augmentations 
      pointsDropout=0.25, pointsNoise=0.002,
      eyesDropout=0.05, eyesAdditiveNoise=0.02,
      forecast=dict(
        past=False, future=True, maxT=0.2,
        keypoints=10
      )
    ),
    debugDistribution=False,
    adjustDistribution=dict(mu='auto', sigma=0.05, noise=0.05)
  )
)
batchN = len(trainDataset)

testDataset = CTestLoader(
  os.path.join(folder, 'test.npz'), 
  batch_size=64*8
)

def evaluate():
  T = time.time()
  lossPerSample = {'loss': [], 'pos': []}
  predV = []
  predDist = []
  # Y = []
  for batchId in range(len(testDataset)):
    _, (y,) = batch = testDataset[batchId]
    loss, predP, dist = model.eval(batch)
    predV.append(predP)
    predDist.append(dist)
    # Y.append(y[:, -1, 0])
    for l, pos in zip(loss, y[:, -1]):
      lossPerSample['loss'].append(l)
      lossPerSample['pos'].append(pos[0])
      continue
    continue
  
  def unbatch(qu):
    qu = np.array(qu)
    qu = qu.transpose(1, 0, *np.arange(2, len(qu.shape)))
    return qu.reshape((qu.shape[0], -1, qu.shape[-1]))

  predV = unbatch(predV).reshape((-1, 2))
  # Y = unbatch(Y)
  if True:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.figure(figsize=(8, 8))
    plt.plot(predV[:, 0], predV[:, 1], 'o', markersize=1)
    for i in range(5):
      d = i * 0.1
      plt.gca().add_patch(
        patches.Rectangle(
          (d,d), 1-2*d, 1-2*d,
          linewidth=1,edgecolor='r',facecolor='none'
        )
      )

    plt.savefig('pred.png')
    plt.clf()
    plt.close()
    pass
  
  trainDataset.updateDistribution(lossPerSample['pos'], lossPerSample['loss'])
  loss = np.mean(lossPerSample['loss'])
  print('Test | %.2f sec | Loss: %.5f. Distance: %.5f' % (time.time() - T, loss, np.mean(predDist)))
  return loss

N = 1000
bestLoss = evaluate() # np.inf
for epoch in range(10):
  timesteps = 5
  batchSize = 16*4
  print('Epoch: %d, timesteps: %d, batchSize: %d' % (epoch, timesteps, batchSize))
  history = defaultdict(list)
  T = time.time()
  for batchId in range(batchN):
    stats = model.fit(trainDataset.sample(
      N=batchSize,
      timesteps=timesteps
    ))
    history['time'].append(stats['time'])
    for k in stats['losses'].keys():
      history[k].append(stats['losses'][k])
      
    if (0 == (batchId % N)) and (0 < batchId):
      statsStr = ', '.join(['%s: %.5f' % (k, np.mean(v[-N:])) for k, v in history.items()])
      print(
        '%d epoch | %d/%d | %s (%.2f sec)' % 
        (epoch, batchId, batchN, statsStr, time.time() - T)
      )

      testLoss = evaluate()
      if testLoss < bestLoss:
        print('Improved %.5f => %.5f' % (bestLoss, testLoss))
        bestLoss = testLoss
        model.save(folder)
      T = time.time()
    continue
  
  testLoss = evaluate()
  if testLoss < bestLoss:
    print('Improved %.5f => %.5f' % (bestLoss, testLoss))
    bestLoss = testLoss
    model.save(folder)

  statsStr = ', '.join(['%s: %.5f' % (k, np.mean(v)) for k, v in history.items()])
  print('%d epoch | %s' % (epoch, statsStr, ))
  trainDataset.on_epoch_end()
  continue