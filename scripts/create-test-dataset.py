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

BATCH_SIZE = 128 * 4
folder = os.path.join(ROOT_FOLDER, 'Data')

def samplesStream(params, take):
  if not isinstance(take, list): take = [take]
  ds = CDataSampler( CSamplesStorage(), defaults=params, batch_size=BATCH_SIZE, minFrames=params['timesteps'] )
  ds.addBlock(Utils.datasetFrom(os.path.join(folder, 'test.npz')))
  
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

def batches(params, take):
  data = defaultdict(list)
  for sample in samplesStream(params, take):
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
def generateTestDataset(params, outputFolder):
  # clear output folder
  if os.path.exists(outputFolder):
    # remove test-*.npz files
    for fname in os.listdir(outputFolder):
      if fname.startswith('test-') and fname.endswith('.npz'):
        os.remove(os.path.join(outputFolder, fname))
      continue
  else:
    os.makedirs(outputFolder)
  # generate test dataset
  print('Generating test dataset to "%s"' % (os.path.basename(outputFolder)))
  ONE_MB = 1024 * 1024
  totalSize = 0
  for bIndex, batch in enumerate(batches(params, ['clean'])):
    fname = os.path.join(outputFolder, 'test-%d.npz' % bIndex)
    # concatenate all arrays
    batch = {k: np.concatenate(v, axis=0) for k, v in batch.items()}
    np.savez(fname, **batch)
    # get fname size
    size = os.path.getsize(fname)
    totalSize += size
    print('%d | Size: %.1f MB | Total: %.1f MB' % (bIndex + 1, size / ONE_MB, totalSize / ONE_MB))
    continue
  print('Done')
  return

def main(args):
  augm = lambda x: dict(timesteps=args.steps, stepsSampling={'max frames': x})
  PARAMS = [
    dict(timesteps=args.steps, stepsSampling='last'),
    # augm(3), augm(4), augm(5),
  ]

  for i, params in enumerate(PARAMS):
    outputFolder = os.path.join(folder, 'test-%d' % i)
    generateTestDataset(params, outputFolder)
    continue
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=5, help='Number of timesteps')
  parser.add_argument('--batch-size', type=int, default=512, help='Batch size of the test dataset')

  args = parser.parse_args()
  BATCH_SIZE = args.batch_size # TODO: fix this hack
  main(args)
  pass