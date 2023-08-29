#!/usr/bin/env python
# -*- coding: utf-8 -*-.
# TODO: add the W&B integration
import numpy as np
from Core.CDatasetLoader import CDatasetLoader
from Core.CTestLoader import CTestLoader
import os, argparse
from collections import defaultdict
import time
from Core.CDemoModel import CDemoModel
import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def unbatch(qu):
  qu = np.array(qu)
  qu = qu.transpose(1, 0, *np.arange(2, len(qu.shape)))
  return qu.reshape((qu.shape[0], -1, qu.shape[-1]))

def plotPrediction(Y, predV, filename):
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

  plt.savefig(filename)
  plt.clf()
  plt.close()
  return

def _eval(dataset, model, plotFilename):
  T = time.time()
  # evaluate the model on the val dataset
  lossPerSample = {'loss': [], 'pos': []}
  predV = []
  predDist = []
  Y = []
  for batchId in range(len(dataset)):
    _, (y,) = batch = dataset[batchId]
    loss, predP, dist = model.eval(batch)
    predV.append(predP)
    predDist.append(dist)
    Y.append(y[:, -1, 0])
    for l, pos in zip(loss, y[:, -1]):
      lossPerSample['loss'].append(l)
      lossPerSample['pos'].append(pos[0])
      continue
    continue

  # plot the predictions and the ground truth
  Y = unbatch(Y).reshape((-1, 2))
  predV = unbatch(predV).reshape((-1, 2))
  plotPrediction(Y, predV, plotFilename)
  
  loss = np.mean(lossPerSample['loss'])
  dist = np.mean(predDist)
  T = time.time() - T
  return loss, dist, T

def evaluate(datasets, model, folder):
  totalLoss = 0.0
  for i, dataset in enumerate(datasets):
    loss, dist, T = _eval(dataset, model, os.path.join(folder, 'pred-%d.png' % i))
    print('Test %d / %d | %.2f sec | Loss: %.5f. Distance: %.5f' % (i + 1, len(datasets), T, loss, dist))
    totalLoss += loss
    continue
  print('Mean loss: %.5f' % (totalLoss / len(datasets), ))
  return totalLoss / len(datasets)

def _modelTrainingLoop(model, dataset):
  def F(desc, timesteps):
    T = timesteps if callable(timesteps) else lambda: timesteps
    history = defaultdict(list)
    # use the tqdm progress bar
    with tqdm.tqdm(total=len(dataset), desc=desc) as pbar:
      dataset.on_epoch_start()
      for _ in range(len(dataset)):
        stats = model.fit(dataset.sample(
          timesteps=T()
        ))
        history['time'].append(stats['time'])
        for k in stats['losses'].keys():
          history[k].append(stats['losses'][k])
        # add stats to the progress bar (mean of each history)
        pbar.set_postfix({k: '%.5f' % np.mean(v) for k, v in history.items()})
        pbar.update(1)
        continue
      dataset.on_epoch_end()
    return
  return F

def main(args):
  timesteps = args.steps
  # setup numpy printing options for debugging
  np.set_printoptions(precision=4, threshold=7777, suppress=True, linewidth=120)
  folder = os.path.dirname(__file__)
  folder = os.path.join(folder, 'Data')

  trainDataset = CDatasetLoader(
    os.path.join(folder, 'train.npz'),
    samplerArgs=dict(
      batch_size=args.batch_size,
      minFrames=timesteps,
      defaults=dict(
        timesteps=timesteps,
        stepsSampling={'max frames': 5, 'include last': True},
        # augmentations
        pointsDropout=0.0, pointsNoise=0.01,
        eyesDropout=0.0, eyesAdditiveNoise=0.01, brightnessFactor=1.0, lightBlobFactor=1.0,
      ),
    )
  )
  model = dict(timesteps=timesteps, trainable=True)
  if args.model is not None:
    model['weights'] = dict(folder=folder, postfix=args.model)

  model = CDemoModel(**model)
  model._model.summary()

  evalDatasets = [
    CTestLoader(os.path.join(folder, nm))
    for nm in os.listdir(folder) if nm.startswith('test-')
  ]
  bestLoss = evaluate(evalDatasets, model, folder)
  bestEpoch = 0
  trainStep = _modelTrainingLoop(model, trainDataset)
  for epoch in range(args.epochs):
    trainStep(
      desc='Epoch %.*d / %d' % (len(str(args.epochs)), epoch, args.epochs),
      timesteps=timesteps
    )
    model.save(folder, postfix='latest')
    
    testLoss = evaluate(evalDatasets, model, folder)
    if testLoss < bestLoss:
      print('Improved %.5f => %.5f' % (bestLoss, testLoss))
      bestLoss = testLoss
      bestEpoch = epoch
      model.save(folder, postfix='best')
      continue

    print('Passed %d epochs since the last improvement' % (epoch - bestEpoch, ))
    if args.patience <= (epoch - bestEpoch):
      print('Early stopping')
      break
    continue
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=1000)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--patience', type=int, default=15)
  parser.add_argument('--steps', type=int, default=5)
  parser.add_argument('--model', type=str)

  main(parser.parse_args())
  pass