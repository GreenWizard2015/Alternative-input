#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
import random
import os
import Utils
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
from collections import defaultdict
import numpy as np

SAMPLES_PER_BUCKET = 5
SAMPLE_ONCE = True
PARAMS = {
  'timesteps': 5,
  'stepsSampling': 'last',
  'maxT': 1.0,
  'onlyFinalPos': False,
  'pointsDropout': 0.0,
  'eyesDropout': 0.0,
}

folder = os.path.dirname(__file__)
ds = CDataSampler( CSamplesStorage(), defaults=PARAMS )
ds.addBlock(Utils.datasetFromFolder(os.path.join(folder, 'Dataset')))

XY = None
for indices in ds.lowlevelSamplesIndexes():
  indices = list(indices)
  random.shuffle(indices)
  samples = []
  while (0 < len(indices)) and (len(samples) < SAMPLES_PER_BUCKET):
    indx = indices.pop() if SAMPLE_ONCE else random.choice(indices)
    sample = ds.sampleById(indx)
    if sample:
      samples.append(sample)
    else:
      if not SAMPLE_ONCE: indices.remove(indx)
    continue
  
  if 0 < len(samples):
    if XY is None:
      XY = [[] for x in samples[0] for _ in x]
    
    for sample in samples:
      flat = [v for x in sample for v in x]
      for x, v in zip(XY, flat): x.append(v)
      continue
  continue

XY = [np.concatenate(x, axis=0) for x in XY]
for x in XY: print(x.shape)
np.savez(os.path.join(folder, 'test.npz'), *XY)
print('Done')