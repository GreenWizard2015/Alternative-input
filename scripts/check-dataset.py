#!/usr/bin/env python
# -*- coding: utf-8 -*-.
'''
This script is load one by one the datasets and check how many unique samples are there
'''
import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

from Core.CDataSamplerInpainting import CDataSamplerInpainting
from Core.CDataSampler import CDataSampler
import Core.Utils as Utils
import json
from Core.CSamplesStorage import CSamplesStorage

def samplesStream(params, filename, ID, batch_size, is_inpainting):
  placeId, userId, screenId = ID
  storage = CSamplesStorage(placeId=placeId, userId=userId, screenId=screenId)
  if is_inpainting:
    ds = CDataSamplerInpainting(
      storage,
      defaults=params, 
      batch_size=batch_size, minFrames=params['timesteps'],
      keys=['clean']
    )
  else:
    ds = CDataSampler(
      storage,
      defaults=params, 
      batch_size=batch_size, minFrames=params['timesteps'],
    )
  ds.addBlock(Utils.datasetFrom(filename))
  
  N = ds.totalSamples
  for i in range(0, N, batch_size):
    indices = list(range(i, min(i + batch_size, N)))
    batch, rejected, accepted = ds.sampleByIds(indices)
    if batch is None: continue

    # main batch
    x, y = batch
    if not is_inpainting:
      x = x['clean']
    for idx in range(len(x['points'])):
      yield idx
  return

def main(args):
  params = dict(
    timesteps=args.steps,
    stepsSampling='uniform',
    # no augmentations by default
    pointsNoise=0.0, pointsDropout=0.0,
    eyesDropout=0.0, eyesAdditiveNoise=0.0, brightnessFactor=1.0, lightBlobFactor=1.0,
    targets=dict(keypoints=3, total=10),
  )
  folder = os.path.join(args.folder, 'Data', 'remote')

  stats = None
  with open(os.path.join(folder, 'stats.json'), 'r') as f:
    stats = json.load(f)

  # enable all disabled datasets
  stats['blacklist'] = []
  for datasetFolder, ID in Utils.dataset_from_stats(stats, folder):
    trainFile = os.path.join(datasetFolder, 'train.npz')
    if not os.path.exists(trainFile):
      continue
    print('Processing', trainFile)

    stream = samplesStream(params, trainFile, ID=ID, batch_size=64, is_inpainting=args.inpainting)
    samplesN = 0
    for _ in stream:
      samplesN += 1
      continue
    print(f'Dataset has {samplesN} valid samples')
    if samplesN <= args.min_samples:
      print(f'Warning: dataset has less or equal to {args.min_samples} samples and will be disabled')
      stats['blacklist'].append(ID)

  with open(os.path.join(folder, 'stats.json'), 'w') as f:
    json.dump(stats, f, indent=2, sort_keys=True, default=str)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=5)
  parser.add_argument('--folder', type=str, default=ROOT_FOLDER)
  parser.add_argument('--min-samples', type=int, default=0)
  parser.add_argument('--inpainting', action='store_true', default=False)
  main(parser.parse_args())
  pass