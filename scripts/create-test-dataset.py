#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

import numpy as np
import Core.Utils as Utils
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
from collections import defaultdict
import glob
import json
import shutil

BATCH_SIZE = 128 * 4
trainIndx = 0

def samplesStream(params, take, filename, stats):
  if not isinstance(take, list): take = [take]
  # filename is "{placeId}/{userId}/{screenId}/train.npz"
  # extract the placeId, userId, and screenId
  parts = os.path.split(filename)[0].split(os.path.sep)
  placeId, userId, screenId = parts[-3], parts[-2], parts[-1]
  # use the stats to get the numeric values of the placeId, userId, and screenId  
  ds = CDataSampler(
    CSamplesStorage(
      placeId=stats['placeId'].index(placeId),
      userId=stats['userId'].index(userId),
      screenId=stats['screenId'].index('%s/%s' % (placeId, screenId))
    ),
    defaults=params, 
    batch_size=BATCH_SIZE, minFrames=params['timesteps'] 
  )
  ds.addBlock(Utils.datasetFrom(filename))
  
  N = ds.totalSamples
  for i in range(0, N, BATCH_SIZE):
    indices = list(range(i, min(i + BATCH_SIZE, N)))
    batch, rejected, accepted = ds.sampleByIds(indices)
    if batch is None: continue

    # main batch
    x, (y, ) = batch
    for idx in range(len(y)):
      for kind in take:
        res = {k: v[idx, None].numpy() for k, v in x[kind].items()}
        res['y'] = y[idx, None]
        yield res
      continue
    continue
  return

def batches(*params):
  data = defaultdict(list)
  for sample in samplesStream(*params):
    for k, v in sample.items():
      data[k].append(v)
      continue

    if BATCH_SIZE <= len(data['y']):
      yield data
      data = defaultdict(list)
    continue

  if 0 < len(data['y']):
    # copy data to match batch size
    for k, v in data.items():
      while len(v) < BATCH_SIZE: v.extend(v)
      data[k] = v[:BATCH_SIZE]
      continue
    yield data
  return
############################################
def generateTestDataset(params, filename, stats, outputFolder):
  # generate test dataset
  global trainIndx
  ONE_MB = 1024 * 1024
  totalSize = 0
  if not os.path.exists(outputFolder):
    os.makedirs(outputFolder, exist_ok=True)
  for bIndex, batch in enumerate(batches(params, ['clean'], filename, stats)):
    fname = os.path.join(outputFolder, 'test-%d.npz' % trainIndx)
    # concatenate all arrays
    batch = {k: np.concatenate(v, axis=0) for k, v in batch.items()}
    np.savez_compressed(fname, **batch)
    # get fname size
    size = os.path.getsize(fname)
    totalSize += size
    print('%d | Size: %.1f MB | Total: %.1f MB' % (bIndex + 1, size / ONE_MB, totalSize / ONE_MB))
    trainIndx += 1
    continue
  print('Done')
  return

def main(args):
  augm = lambda x: dict(timesteps=args.steps, stepsSampling={'max frames': x})
  PARAMS = [
    dict(timesteps=args.steps, stepsSampling='last'),
    # augm(3), augm(4), augm(5),
  ]
  folder = os.path.join(ROOT_FOLDER, 'Data', 'remote')

  stats = None
  with open(os.path.join(folder, 'stats.json'), 'r') as f:
    stats = json.load(f)

  # remove all content from the output folder
  shutil.rmtree(args.output, ignore_errors=True)
  # recursively find the train file
  trainFilename = glob.glob(os.path.join(folder, '**', 'test.npz'), recursive=True)
  print('Found test files:', len(trainFilename))
  for idx, filename in enumerate(trainFilename):
    print('Processing', filename)
    for params in PARAMS:
      targetFolder = os.path.join(args.output, 'test-%d' % idx)
      generateTestDataset(params, filename, stats, outputFolder=targetFolder)
      continue
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=5, help='Number of timesteps')
  parser.add_argument('--batch-size', type=int, default=512, help='Batch size of the test dataset')
  parser.add_argument(
    '--output', type=str, help='Output folder',
    default=os.path.join(ROOT_FOLDER, 'Data', 'test-main')
  )
  args = parser.parse_args()
  BATCH_SIZE = args.batch_size # TODO: fix this hack
  main(args)
  pass