#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
import random
import os, shutil
import Core.Utils as Utils
Utils.setupGPU(memory_limit=1024)
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
from collections import defaultdict

SAMPLES_PER_BUCKET = 5
SAMPLE_ONCE = not True
PARAMS = dict(
  timesteps=5,
  stepsSampling={'max frames': 5, 'include last': False},
  maxT=1.0,
  # augmentations
  pointsDropout=0.1, pointsNoise=0.001,
  eyesDropout=0., eyesAdditiveNoise=0.01, brightnessFactor=1.1, lightBlobFactor=1.1,
  shiftsN=1,
  radialShiftsN=1,
)

folder = os.path.dirname(__file__)
folder = os.path.join(folder, 'Data')
output = os.path.join(folder, 'pretrain.npz')

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
        for kind in ['clean']: #['augmented']:
          res = {k: v[idx, None].numpy() for k, v in x[kind].items()}
          res['y'] = y[idx, None]
          yield res
        continue
      continue
    continue
  return
############################################
N = 0
data = defaultdict(list)
for sample in samplesStream():
  for k, v in sample.items():
    data[k].append(v)
    continue
  N += 1
  if (N % 1000) == 0: print(N)
  continue

data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
# shuffle
indices = np.arange(len(data['y']))
np.random.shuffle(indices)
for k, v in data.items():
  data[k] = v[indices]
  continue

np.savez_compressed(output, **data)
print('Done')
