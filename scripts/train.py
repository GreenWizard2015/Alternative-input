#!/usr/bin/env python
# -*- coding: utf-8 -*-.
# TODO: add the W&B integration
import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

import numpy as np
from Core.CDatasetLoader import CDatasetLoader
from Core.CTestLoader import CTestLoader
from collections import defaultdict
import time
from Core.CModelTrainer import CModelTrainer
import tqdm
import json
import glob

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

def _eval(dataset, model, plotFilename, args):
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

  if args.debug: # plot the predictions and the ground truth
    Y = unbatch(Y).reshape((-1, 2))
    predV = unbatch(predV).reshape((-1, 2))
    plotPrediction(Y, predV, plotFilename)
  
  loss = np.mean(lossPerSample['loss'])
  dist = np.mean(predDist)
  T = time.time() - T
  return loss, dist, T

def evaluator(datasets, model, folder, args):
  losses = [np.inf] * len(datasets) # initialize with infinity
  def evaluate():
    totalLoss = totalDist = 0.0
    for i, dataset in enumerate(datasets):
      loss, dist, T = _eval(dataset, model, os.path.join(folder, 'pred-%d.png' % i), args)
      print('Test %d / %d | %.2f sec | Loss: %.5f (%.5f). Distance: %.5f' % (
        i + 1, len(datasets), T, loss, losses[i], dist
      ))
      if loss < losses[i]:
        print('Improved %.5f => %.5f' % (losses[i], loss))
        model.save(folder, postfix='best-%d' % i) # save the model separately
        losses[i] = loss
        pass

      totalLoss += loss
      totalDist += dist
      continue
    print('Mean loss: %.5f | Mean distance: %.5f' % (
      totalLoss / len(datasets), totalDist / len(datasets)
    ))
    return totalLoss / len(datasets)
  return evaluate

def _modelTrainingLoop(model, dataset):
  def F(desc, sampleParams):
    history = defaultdict(list)
    # use the tqdm progress bar
    with tqdm.tqdm(total=len(dataset), desc=desc) as pbar:
      dataset.on_epoch_start()
      for _ in range(len(dataset)):
        sampled = dataset.sample(**sampleParams)
        assert 2 == len(sampled), 'The dataset should return a tuple with the input and the output'
        X, Y = sampled
        assert 'clean' in X, 'The input should contain the clean data'
        assert 'augmented' in X, 'The input should contain the augmented data'
        for nm in ['clean', 'augmented']:
          item = X[nm]
          assert 'points' in item, 'The input should contain the points'
          assert 'left eye' in item, 'The input should contain the left eye'
          assert 'right eye' in item, 'The input should contain the right eye'
          assert 'time' in item, 'The input should contain the time'
          assert 'userId' in item, 'The input should contain the userId'
          assert 'placeId' in item, 'The input should contain the placeId'
          assert 'screenId' in item, 'The input should contain the screenId'
          continue
        stats = model.fit(sampled)
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

def _defaultSchedule(args):
  return lambda epoch: dict()

def _schedule_from_json(args):
  with open(args.schedule, 'r') as f:
    schedule = json.load(f)

  # schedule is a dictionary of dictionaries, where the keys are the epochs
  # transform it into a sorted list of tuples (epoch, params)
  for k, v in schedule.items():
    v = [(int(epoch), p) for epoch, p in v.items()]
    schedule[k] = sorted(v, key=lambda x: x[0])
    continue

  def F(epoch):
    res = {}
    for k, v in schedule.items():
      assert isinstance(v, list), 'The schedule should be a list of parameters'
      # find the first epoch that is less or equal to the current one
      smallest = [i for i, (e, _) in enumerate(v) if e <= epoch]
      if 0 == len(smallest): continue
      smallest = smallest[-1]

      startEpoch, p = v[smallest]
      value = p
      # p could be a dictionary or value
      if isinstance(p, list) and (2 == len(p)):
        assert smallest + 1 < len(v), 'The last epoch should be the last one'
        minV, maxV = [float(x) for x in p]
        nextEpoch, _ = v[smallest + 1]
        # linearly interpolate between the values
        value = minV + (maxV - minV) * (epoch - startEpoch) / (nextEpoch - startEpoch)
        pass
      
      res[k] = float(value)
      continue

    if args.debug and res:
      print('Parameters for epoch %d:' % (epoch, ))
      for k, v in res.items():
        print('  %s: %.5f' % (k, v))
        continue
    return res
  return F

def _trainer_from(args):
  if args.trainer == 'default': return CModelTrainer
  raise Exception('Unknown trainer: %s' % (args.trainer, ))

def averageModels(folder, model, noiseStd=0.0):
  TV = [np.zeros_like(x) for x in model._model.get_weights()]
  N = 0
  for nm in glob.glob(os.path.join(folder, '*.h5')):
    if not('best' in nm): continue # only the best models
    model.load(nm, embeddings=True)
    # add the weights to the total
    weights = model._model.get_weights()
    for i in range(len(TV)):
      TV[i] += weights[i]
      continue
    N += 1
    continue

  # average the weights
  TV = [(x / N) + np.random.normal(0.0, noiseStd, x.shape) for x in TV]
  model._model.set_weights(TV)
  return

def main(args):
  timesteps = args.steps
  folder = os.path.join(args.folder, 'Data')
  if args.schedule is None:
    getSampleParams = _defaultSchedule(args)
  else:
    getSampleParams = _schedule_from_json(args)

  stats = None
  with open(os.path.join(folder, 'remote', 'stats.json'), 'r') as f:
    stats = json.load(f)

  trainer = _trainer_from(args)
  trainDataset = CDatasetLoader(
    os.path.join(folder, 'remote'),
    stats=stats,
    samplerArgs=dict(
      batch_size=args.batch_size,
      minFrames=timesteps,
      maxT=1.0,
      defaults=dict(
        timesteps=timesteps,
        stepsSampling='uniform',
        # no augmentations by default
        pointsNoise=0.02, pointsDropout=0.1,
        eyesDropout=0.1, eyesAdditiveNoise=0.01, brightnessFactor=5.0, lightBlobFactor=5.0,
      ),
    )
  )
  model = dict(timesteps=timesteps, stats=stats)
  if args.model is not None:
    model['weights'] = dict(folder=folder, postfix=args.model, embeddings=args.embeddings)
  if args.modelId is not None:
    model['model'] = args.modelId

  model = trainer(**model)
  model._model.summary()

  if args.average:
    averageModels(folder, model)
  # find folders with the name "/test-*/"
  evalDatasets = [
    CTestLoader(nm)
    for nm in glob.glob(os.path.join(folder, 'test-main', 'test-*/'))
  ]
  eval = evaluator(evalDatasets, model, folder, args)
  bestLoss = eval()
  bestEpoch = 0
  trainStep = _modelTrainingLoop(model, trainDataset)
  for epoch in range(args.epochs):
    trainStep(
      desc='Epoch %.*d / %d' % (len(str(args.epochs)), epoch, args.epochs),
      sampleParams=getSampleParams(epoch)
    )
    model.save(folder, postfix='latest')
    
    testLoss = eval()
    if testLoss < bestLoss:
      print('Improved %.5f => %.5f' % (bestLoss, testLoss))
      bestLoss = testLoss
      bestEpoch = epoch
      model.save(folder, postfix='best')
      continue

    print('Passed %d epochs since the last improvement (best: %.5f)' % (epoch - bestEpoch, bestLoss))
    if args.patience <= (epoch - bestEpoch):
      if 'stop' == args.on_patience:
        print('Early stopping')
        break
      if 'reset' == args.on_patience:
        print('Resetting the model to the average of the best models')
        # and add some noise
        averageModels(folder, model, noiseStd=0.01)
        bestEpoch = epoch
        continue
    continue
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=1000)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--patience', type=int, default=5)
  parser.add_argument('--on-patience', type=str, default='stop', choices=['stop', 'reset'])
  parser.add_argument('--steps', type=int, default=5)
  parser.add_argument('--model', type=str)
  parser.add_argument('--embeddings', default=False, action='store_true')
  parser.add_argument(
    '--average', default=False, action='store_true',
    help='Load each model from the folder and average them weights'
  )
  parser.add_argument('--folder', type=str, default=ROOT_FOLDER)
  parser.add_argument('--modelId', type=str)
  parser.add_argument(
    '--trainer', type=str, default='default',
    choices=['default']
  )
  parser.add_argument(
    '--schedule', type=str, default=None,
    help='JSON file with the scheduler parameters for sampling the training dataset'
  )
  parser.add_argument('--debug', action='store_true')

  main(parser.parse_args())
  pass