#!/usr/bin/env python
# -*- coding: utf-8 -*-.
# TODO: add the W&B integration
import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

import numpy as np
from Core.CDataSamplerInpainting import CDataSamplerInpainting
from Core.CDatasetLoader import CDatasetLoader
from Core.CTestInpaintingLoader import CTestInpaintingLoader
from collections import defaultdict
import time
from Core.CInpaintingTrainer import CInpaintingTrainer
import tqdm
import json
import glob

def _eval(dataset, model):
  T = time.time()
  # evaluate the model on the val dataset
  loss = []
  for batchId in range(len(dataset)):
    batch = dataset[batchId]
    loss_value = model.eval(batch)
    loss.append(loss_value)
    continue

  loss = np.mean(loss)
  T = time.time() - T
  return loss, T

def evaluator(datasets, model, folder, args):
  losses = [np.inf] * len(datasets) # initialize with infinity
  def evaluate(onlyImproved=False):
    totalLoss = []
    losses_dist = []
    for i, dataset in enumerate(datasets):
      loss, T = _eval(dataset, model)
      isImproved = loss < losses[i]
      if (not onlyImproved) or isImproved:
        print('Test %d / %d | %.2f sec | Loss: %.5f (%.5f).' % (
          i + 1, len(datasets), T, loss, losses[i],
        ))
      if isImproved:
        print('Test %d / %d | Improved %.5f => %.5f,' % (
          i + 1, len(datasets), losses[i], loss,
        ))
        # model.save(folder, postfix='best-%d' % i) # save the model separately
        losses[i] = loss
        pass

      totalLoss.append(loss)
      continue
    if not onlyImproved:
      print('Mean loss: %.5f' % (
        np.mean(totalLoss)
      ))
    return np.mean(totalLoss)
  return evaluate

def _modelTrainingLoop(model, dataset):
  def F(desc):
    history = defaultdict(list)
    # use the tqdm progress bar
    with tqdm.tqdm(total=len(dataset), desc=desc) as pbar:
      dataset.on_epoch_start()
      for _ in range(len(dataset)):
        sampled = dataset.sample()
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

def _trainer_from(args):
  if args.trainer == 'default': return CInpaintingTrainer
  raise Exception('Unknown trainer: %s' % (args.trainer, ))

def averageModels(folder, model, noiseStd=0.0):
  TV = [np.zeros_like(x) for x in model.trainable_variables()]
  N = 0
  for nm in glob.glob(os.path.join(folder, '*.h5')):
    if not('best' in nm): continue # only the best models
    model.load(nm, embeddings=True)
    # add the weights to the total
    weights = model.trainable_variables()
    for i in range(len(TV)):
      TV[i] += weights[i].numpy()
      continue
    N += 1
    continue

  # average the weights
  TV = [(x / N) + np.random.normal(0.0, noiseStd, x.shape) for x in TV]
  for v, new in zip(model.trainable_variables(), TV):
    v.assign(new)
    continue
  model.compile() # recompile the model with the new weights
  return

def main(args):
  timesteps = args.steps
  folder = os.path.join(args.folder, 'Data')

  stats = None
  with open(os.path.join(folder, 'remote', 'stats.json'), 'r') as f:
    stats = json.load(f)

  trainer = _trainer_from(args)
  trainDataset = CDatasetLoader(
    os.path.join(folder, 'remote'),
    stats=stats,
    sampling=args.sampling,
    samplerArgs=dict(
      batch_size=args.batch_size,
      minFrames=timesteps,
      maxT=1.0,
      defaults=dict(
        timesteps=timesteps,
        stepsSampling='uniform',
        # no augmentations by default
        pointsNoise=0.01, pointsDropout=0.0,
        eyesDropout=0.1, eyesAdditiveNoise=0.01, brightnessFactor=1.5, lightBlobFactor=1.5,
        targets=dict(keypoints=3, total=10),
      ),
      keys=['clean', 'augmented'],
    ),
    sampler_class=CDataSamplerInpainting,
  )
  model = dict(timesteps=timesteps, stats=stats)
  if args.model is not None:
    model['weights'] = dict(folder=folder, postfix=args.model, embeddings=args.embeddings)
  if args.modelId is not None:
    model['model'] = args.modelId

  model = trainer(**model)
#   model._model.summary()

  # find folders with the name "/test-*/"
  evalDatasets = [
    CTestInpaintingLoader(nm)
    for nm in glob.glob(os.path.join(folder, 'test-inpainting', 'test-*/'))
  ]
  eval = evaluator(evalDatasets, model, folder, args)
  bestLoss = eval() # evaluate loaded model
  bestEpoch = 0
  # wrapper for the evaluation function. It saves the model if it is better
  def evalWrapper(eval):
    def f(epoch, onlyImproved=False):
      nonlocal bestLoss, bestEpoch
      newLoss = eval(onlyImproved=onlyImproved)
      if newLoss < bestLoss:
        print('Improved %.5f => %.5f' % (bestLoss, newLoss))
        if onlyImproved: #details
          for i, (loss, bestLoss_, dist, bestDist) in enumerate(losses):
            print('Test %d | Loss: %.5f (%.5f). Distance: %.5f (%.5f)' % (i + 1, loss, bestLoss_, dist, bestDist))
            continue
          print('-' * 80)
        bestLoss = newLoss
        bestEpoch = epoch
        # model.save(folder, postfix='best')
      return
    return f
  
  eval = evalWrapper(eval)
  trainStep = _modelTrainingLoop(model, trainDataset)
  for epoch in range(args.epochs):
    trainStep(
      desc='Epoch %.*d / %d' % (len(str(args.epochs)), epoch, args.epochs),
    )
    # model.save(folder, postfix='latest')
    eval(epoch)

    print('Passed %d epochs since the last improvement (best: %.5f)' % (epoch - bestEpoch, bestLoss))
    if args.patience <= (epoch - bestEpoch):
      if 'stop' == args.on_patience:
        print('Early stopping')
        break
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
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--noise', type=float, default=1e-4)
  parser.add_argument(
    '--restarts', type=int, default=1,
    help='Number of times to restart the model reinitializing the weights'
  )
  parser.add_argument(
    '--sampling', type=str, default='uniform',
    choices=['uniform', 'as_is'],
  )

  main(parser.parse_args())
  pass