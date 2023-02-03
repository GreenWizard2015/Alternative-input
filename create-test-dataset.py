#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
import random
import os
import Core.Utils as Utils
Utils.setupGPU(memory_limit=1024)
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
from collections import defaultdict

BATCH_SIZE = 128 * 4
SAMPLES_PER_BUCKET = 5
SAMPLE_ONCE = True
PARAMS = dict(
  timesteps=5,
  stepsSampling='last',
  maxT=1.0,
)

folder = os.path.dirname(__file__)
folder = os.path.join(folder, 'Data')
outputFolder = os.path.join(folder, 'test')

def samplesStream():
  ds = CDataSampler( CSamplesStorage(), defaults=PARAMS )
  ds.addBlock(Utils.datasetFrom(os.path.join(folder, 'test.npz')))

  for indices in ds.lowlevelSamplesIndexes():
    indices = list(indices)
    samplesN = SAMPLES_PER_BUCKET
    # take indices from the bucket by chunks of SAMPLES_PER_BUCKET
    while (0 < len(indices)) and (0 < samplesN):
      random.shuffle(indices)
      indicesChunk = indices[:SAMPLES_PER_BUCKET]
      batch, rejected, accepted = ds.sampleByIds(indicesChunk)

      indicesToRemove = indicesChunk if SAMPLE_ONCE else rejected
      for idx in indicesToRemove: indices.remove(idx)

      if batch is None: continue

      # main batch
      x, (y, ) = batch
      N = min(len(y), samplesN)
      samplesN -= N
      for idx in range(N):
        for kind in ['clean']:
          res = {k: v[idx, None].numpy() for k, v in x[kind].items()}
          res['y'] = y[idx, None]
          yield res
        continue
    continue
  return

def batches():
  data = defaultdict(list)
  for sample in samplesStream():
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
# clear output folder
if os.path.exists(outputFolder):
  # remove test-*.npz files
  for fname in os.listdir(outputFolder):
    if fname.startswith('test-') and fname.endswith('.npz'):
      os.remove(os.path.join(outputFolder, fname))
    continue
else:
  os.makedirs(outputFolder)

totalSize = 0
for bIndex, batch in enumerate(batches()):
  fname = os.path.join(outputFolder, 'test-%d.npz' % bIndex)
  # concatenate all arrays
  batch = {k: np.concatenate(v, axis=0) for k, v in batch.items()}
  np.savez(fname, **batch)
  # get fname size
  size = os.path.getsize(fname)
  totalSize += size
  print('%d | Size: %d MB | Total: %d MB' % (bIndex + 1, size // 1024 // 1024, totalSize // 1024 // 1024))
  continue
print('Done')