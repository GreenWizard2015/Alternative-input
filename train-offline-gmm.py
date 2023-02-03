#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
from Core.CDatasetLoader import CDatasetLoader
from Core.CTestLoader import CTestLoader
from Core.CPretrainLoader import CPretrainLoader
import os
from collections import defaultdict
import time
from Core.CGMModel import CGMModel
import tensorflow as tf

BATCH_PER_EPOCH = 15_000
EVAL_EVERY = 1_000
SHIFT_AUGMENTATION = 512
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
      pointsDropout=0.2, pointsNoise=0.005,
      eyesDropout=0., eyesAdditiveNoise=0.05, brightnessFactor=2., lightBlobFactor=2.,
      shiftsN=SHIFT_AUGMENTATION,
      radialShiftsN=SHIFT_AUGMENTATION,
    ),
  )
)

trainContexts = trainDataset.contexts + [SHIFT_AUGMENTATION, SHIFT_AUGMENTATION]
print('Train contexts:', trainContexts)

pretrainDataset = CPretrainLoader(
  os.path.join(folder, 'pretrain.npz'),
  batch_size=16,
  contextsStartIndex=trainContexts,
)
testDataset = CTestLoader(os.path.join(folder, 'test'), contextsStartIndex=trainContexts)
testContexts = [2, 2, 3] + [2, 2] # testDataset.contexts

print('Test contexts:', testContexts)
contextsAll = [a + b for a, b in zip(trainContexts, testContexts)]
print('All contexts:', contextsAll)

model = CGMModel(
  F2LArgs=dict(steps=5, contexts=contextsAll),
  # weights=dict(folder=folder), 
  trainable=True,
  useDiscriminator=False,
)

def evaluate():
  T = time.time()

  # drop the embeddings
  embeddings = model._face2latent.get_layer('c_context_encoder').embeddings
  if False:
    for emb, trainCtxSize, ctxSize in zip(embeddings, trainContexts, contextsAll):
      W = emb.get_weights()
      assert len(W) == 1
      oldWeights = W[0]
      assert np.allclose(oldWeights.shape, (ctxSize, emb.output_dim))
      
      oldWeights[trainCtxSize:] = oldWeights[:trainCtxSize].mean(axis=0, keepdims=True)
      emb.set_weights([oldWeights])
      continue
  # pretrain the embeddings on the test dataset
  pretrainEpochs = 1
  optimizer = tf.keras.optimizers.Adam(
    # linearly decay the learning rate from 1e-3 to 1e-5 in len(pretrainDataset) steps
    learning_rate=1e-4,
  )
  @tf.function
  def pretrain_step(batch):
    print('pre')
    TV = sum([x.trainable_variables for x in embeddings], [])
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(TV)
      loss = model.fit_embeddings(batch)
      pass
    optimizer.minimize(loss, TV, tape=tape)
    return loss
    
  for epoch in range(pretrainEpochs):
    history = []
    for batchId in range(len(pretrainDataset)):
      batch = pretrainDataset[batchId]
      loss = pretrain_step(batch)
      history.append(loss.numpy())
      continue

    print('Pretrain | %d | %.2f sec | Loss: %.5f' % (
      epoch,
      time.time() - T,
      np.mean(history)
    ))
    continue
  # evaluate the model on the val dataset
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

  if False:
    # Y = unbatch(Y)
    predV = unbatch(predV).reshape((-1, 2))
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

    plt.savefig(os.path.join(folder, 'pred.png'))
    plt.clf()
    plt.close()
    pass
  
  loss = np.mean(lossPerSample['loss'])
  print('Test | %.2f sec | Loss: %.5f. Distance: %.5f' % (time.time() - T, loss, np.mean(predDist)))
  return loss

bestLoss = evaluate() # np.inf
# bestLoss = np.inf
for epoch in range(10):
  timesteps = 5
  batchSize = 64
  print('Epoch: %d, timesteps: %d, batchSize: %d' % (epoch, timesteps, batchSize))
  print('lr: %.2e' % (model.learning_rate,))
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
      statsStr = ', '.join(['%s: %.5f' % (k, np.mean(v[-EVAL_EVERY:])) for k, v in history.items()])
      print(
        '%d epoch | %d/%d | %s (%.2f sec)' % 
        (epoch, batchId, len(trainDataset), statsStr, time.time() - T)
      )
      # print current learning rate in scientific notation
      print('lr: %.2e' % (model.learning_rate,))
      
      testLoss = evaluate()
      evaluated = True
      if testLoss < bestLoss:
        print('Improved %.5f => %.5f' % (bestLoss, testLoss))
        bestLoss = testLoss
        model.save(folder)
      T = time.time()
      pass
    continue
  
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