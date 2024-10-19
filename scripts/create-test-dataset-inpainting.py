#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

import numpy as np
import Core.Utils as Utils
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSamplerInpainting import CDataSamplerInpainting
from collections import defaultdict
import json
import tensorflow as tf

def samplesStream(params, take, filename, ID, batch_size):
  if not isinstance(take, list): take = [take]
  placeId, userId, screenId = ID
  # use the stats to get the numeric values of the placeId, userId, and screenId  
  ds = CDataSamplerInpainting(
    CSamplesStorage(
      placeId=placeId,
      userId=userId,
      screenId=screenId,
    ),
    defaults=params, 
    batch_size=batch_size, minFrames=params['timesteps'],
    keys=take
  )
  ds.addBlock(Utils.datasetFrom(filename))
  
  N = ds.totalSamples
  for i in range(0, N, batch_size):
    indices = list(range(i, min(i + batch_size, N)))
    batch, rejected, accepted = ds.sampleByIds(indices)
    if batch is None: continue

    # main batch
    x, y = batch
    for idx in range(len(x['points'])):
      resX = {}
      for k, v in x.items():
        item = v[idx, None]
        if tf.is_tensor(item): item = item.numpy()
        resX[f'X_{k}'] = item
        continue

      resY = {}
      for k, v in y.items():
        item = v[idx, None]
        if tf.is_tensor(item): item = item.numpy()
        resY[f'Y_{k}'] = item
        continue
        
      yield dict(**resX, **resY)
      continue
    continue
  return

def batches(stream, batch_size):
  data = defaultdict(list)
  for sample in stream:
    for k, v in sample.items():
      data[k].append(v)
      continue

    if batch_size <= len(data['X_points']):
      yield data
      data = defaultdict(list)
    continue

  if 0 < len(data['X_points']):
    # copy data to match batch size
    for k, v in data.items():
      while len(v) < batch_size: v.extend(v)
      data[k] = v[:batch_size]
      continue
    yield data
  return
############################################
def generateTestDataset(outputFolder, stream):
  # generate test dataset
  ONE_MB = 1024 * 1024
  totalSize = 0
  if not os.path.exists(outputFolder):
    os.makedirs(outputFolder, exist_ok=True)
  for bIndex, batch in enumerate(stream):
    fname = os.path.join(outputFolder, 'test-%d.npz' % bIndex)
    # concatenate all arrays
    batch = {k: np.concatenate(v, axis=0) for k, v in batch.items()}
    np.savez_compressed(fname, **batch)
    # get fname size
    size = os.path.getsize(fname)
    totalSize += size
    print('%d | Size: %.1f MB | Total: %.1f MB' % (bIndex + 1, size / ONE_MB, totalSize / ONE_MB))
    continue
  print('Done')
  return

def main(args):
  PARAMS = [
    dict(      
      timesteps=args.steps,
      stepsSampling='uniform',
      # no augmentations by default
      pointsNoise=0.01, pointsDropout=0.0,
      eyesDropout=0.1, eyesAdditiveNoise=0.01, brightnessFactor=1.5, lightBlobFactor=1.5,
      targets=dict(keypoints=3, total=10),
    ),
  ]
  folder = os.path.join(ROOT_FOLDER, 'Data', 'remote')

  stats = None
  with open(os.path.join(folder, 'stats.json'), 'r') as f:
    stats = json.load(f)

  for datasetFolder, (place_id_index, user_id_index, screen_id_index) in Utils.dataset_from_stats(stats, folder):
    trainFile = os.path.join(datasetFolder, 'test.npz')
    if not os.path.exists(trainFile):
      continue
    print('Processing', trainFile)

    for i, params in enumerate(PARAMS):
      output = args.output
      if 0 < i: output += '-%d' % i
      targetFolder = os.path.join(datasetFolder, output)
      ID = (place_id_index, user_id_index, screen_id_index)
      stream = samplesStream(params, ['clean'], trainFile, ID=ID, batch_size=args.batch_size)
      stream = batches(stream, batch_size=args.batch_size)
      generateTestDataset(outputFolder=targetFolder, stream=stream)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=5, help='Number of timesteps')
  parser.add_argument('--batch-size', type=int, default=512, help='Batch size of the test dataset')
  parser.add_argument(
    '--output', type=str, help='Output folder name',
    default='test-inpainting'
  )
  args = parser.parse_args()
  main(args)
  pass