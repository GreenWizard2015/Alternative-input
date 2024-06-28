#!/usr/bin/env python
# -*- coding: utf-8 -*-.
'''
This script performs the following steps:
- Load the best model from the Data folder
- Load the test datasets from the Data/test-main folder
- Evaluate the model on the test datasets
- Add each test dataset to blacklists if the model mean loss is greater than the threshold
'''
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

def _eval(dataset, model):
  T = time.time()
  # evaluate the model on the val dataset
  losses = []
  predDist = []
  for batchId in range(len(dataset)):
    batch = dataset[batchId]
    loss, _, dist = model.eval(batch)
    predDist.append(dist)
    losses.append(loss)
    continue
  
  loss = np.mean(losses)
  dist = np.mean(predDist)
  T = time.time() - T
  return loss, dist, T

def evaluate(dataset, model):
  loss, dist, T = _eval(dataset, model)
  print('Test | %.2f sec | Loss: %.5f. Distance: %.5f' % (
    T, loss, dist,
  ))
  return loss, dist

def main(args):
  timesteps = args.steps
  folder = args.folder
  stats = None
  with open(os.path.join(folder, 'remote', 'stats.json'), 'r') as f:
    stats = json.load(f)
    
  badDatasets = [] # list of tuples (userId, placeId, screenId)
  if os.path.exists(os.path.join(folder, 'blacklist.json')):
    with open(os.path.join(folder, 'blacklist.json'), 'r') as f:
      badDatasets = json.load(f)
    pass

  model = dict(timesteps=timesteps, stats=stats, use_encoders=False)
  assert args.model is not None, 'The model should be specified'
  if args.model is not None:
    model['weights'] = dict(folder=folder, postfix=args.model, embeddings=True)

  model = CModelTrainer(**model)
  # find folders with the name "/test-*/"
  for nm in glob.glob(os.path.join(folder, 'test-main', 'test-*/')):
    evalDataset = CTestLoader(nm)
    loss, dist = evaluate(evalDataset, model)
    if args.threshold < dist:
      badDatasets.append(evalDataset.parametersIDs())
    continue
  # convert indices to the strings
  res = []
  for userId, placeId, screenId in badDatasets:
    userId = stats['userId'][userId]
    placeId = stats['placeId'][placeId]
    screenId = stats['screenId'][screenId]
    res.append((userId, placeId, screenId))
    continue
  print('Blacklisted datasets:')
  print(json.dumps(res, indent=2))
  # save the blacklisted datasets
  with open(os.path.join(folder, 'blacklist.json'), 'w') as f:
    json.dump(res, f, indent=2)
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=5)
  parser.add_argument('--model', type=str, default='best')
  parser.add_argument('--folder', type=str, default=os.path.join(ROOT_FOLDER, 'Data'))
  parser.add_argument(
    '--threshold', type=float, required=True,
  )

  main(parser.parse_args())
  pass