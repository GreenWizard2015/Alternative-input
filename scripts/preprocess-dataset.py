#!/usr/bin/env python
# -*- coding: utf-8 -*-.
'''
This script performs the following steps:
  1. Load the dataset from the source folder
  2. Split the dataset into "sessions" (i.e. consecutive frames) with a gap of N seconds
  3. Split each session into training and testing sets
  4. Save the training.npz and testing.npz files
'''

import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

import numpy as np
import Core.Utils as Utils

def splitSession(start, end, ratio, framesPerChunk, splits):
  N = end - start
  idx = np.arange(start, end)
  # pad with -1 to make it divisible by framesPerChunk
  pad = framesPerChunk - (N % framesPerChunk)
  idx = np.pad(idx, (0, pad), 'constant', constant_values=-1)
  # reshape into chunks
  chunks = idx.reshape(-1, framesPerChunk)
  # shuffle chunks by axis 0
  idx = np.arange(len(chunks))
  np.random.shuffle(idx)
  chunks = chunks[idx]
  # split into training and testing chunks
  split = int(ratio * len(chunks))
  training = chunks[:split]
  testing = chunks[split:]
  # split training into splits
  trainingN = len(training)
  trainingSets = [training[i::splits] for i in range(splits)]
  # shuffle training sets using the numpy methods
  idx = np.arange(len(trainingSets))
  np.random.shuffle(idx) # shuffle indices according to the seed
  trainingSets = [trainingSets[i] for i in idx]
  ####
  print(
    'Session {}-{}: {} chunks, {} training{}, {} testing'.format(
      start, end, len(chunks),
      trainingN,
      ' ({})'.format(', '.join([str(len(training)) for training in trainingSets])) if 1 < splits else '',
      len(testing)
    )
  )
  # remove padding
  def F(x):
    if len(x) == 0: return []
    x = x.reshape(-1) # flatten
    return x[x != -1] # remove padding
  return [F(training) for training in trainingSets], F(testing)

def splitDataset(dataset, ratio, framesPerChunk, skipAction, splits):
  # split each session into training and testing sets
  trainingSets = [[] for _ in range(splits)] # list of (start, end) tuples for each split
  testing = [] # list of (start, end) tuples
  for i, (start, end) in enumerate(dataset):
    trainingIdxSets = []
    testingIdx = []
    if (end - start) < 2 * framesPerChunk:
      print('Session %d is too short. Action: %s' % (i, skipAction))
      if 'drop' == skipAction: continue

      rng = np.arange(start, end)
      if 'train' == skipAction: trainingIdxSets = [rng for _ in range(splits)]
      if 'test' == skipAction: testingIdx = rng
    else:
      trainingIdxSets, testingIdx = splitSession(start, end, ratio, framesPerChunk, splits=splits)

    # store training and testing sets if they are not empty
    for tIdx, trainingIdx in enumerate(trainingIdxSets):
      if 0 < len(trainingIdx): trainingSets[tIdx].append(trainingIdx)
      continue
    if 0 < len(testingIdx): testing.append(testingIdx)
    continue
  # save training and testing sets
  testing = np.sort(np.concatenate(testing))
  trainingSets = [np.sort(np.concatenate(training)) for training in trainingSets]
  
  # check that training and testing sets are disjoint
  for training in trainingSets:
    intersection = np.intersect1d(training, testing)
    if 0 < len(intersection):
      print('Training and testing sets are not disjoint!')
      print(intersection)
      raise Exception('Training and testing sets are not disjoint!')
    continue

  return trainingSets, testing

def dropPadding(idx, padding):
  res = []
  # find consecutive frames chunks, save their start and end indices
  gaps = np.where(1 < np.diff(idx))[0]
  gaps = np.concatenate(([0], 1 + gaps, [len(idx)]))
  for a, b in zip(gaps[:-1], gaps[1:]):
    chunk = idx[a:b]
    assert np.all(1 == np.diff(chunk)), 'Should be consecutive'
    res.append(chunk)
    continue
  assert np.all(np.concatenate(res) == idx), 'Should be equal'
  res = [chunk[padding:-padding] for chunk in res]
  # remove chunks that are too short
  res = [chunk for chunk in res if padding < len(chunk)]
  res = np.concatenate(res)
  print('Frames before: {}. Frames after: {}'.format(len(idx), len(res)))
  return res

def main(args):
  # set random seed (I hope this is enough to make the results reproducible)
  np.random.seed(args.seed)
  ####
  folder = os.path.join(ROOT_FOLDER, 'Data')
  src = os.path.join(folder, 'Dataset')
  # load dataset
  dataset = Utils.datasetFrom(src)
  # split dataset into sessions
  sessions = Utils.extractSessions(dataset, float(args.time_delta))
  # print sessions and their durations for debugging
  print('Found {} sessions'.format(len(sessions)))
  for i, (start, end) in enumerate(sessions):
    duration = dataset['time'][end - 1] - dataset['time'][start]
    print('Session {}: {} - {} ({}, {})'.format(i, start, end, end - start, duration))
    continue
  ######################################################
  # split each session into training and testing sets
  trainingSets, testing = splitDataset(
    sessions,
    ratio=1.0 - float(args.test_ratio),
    framesPerChunk=int(args.frames_per_chunk),
    skipAction=args.skipped_frames,
    splits=int(args.splits)
  )
  
  testPadding = int(args.test_padding)
  if 0 < testPadding:
    testing = dropPadding(testing, testPadding)

  def saveSubset(filename, idx):
    print('%s: %d frames' % (filename, len(idx)))
    subset = {k: v[idx] for k, v in dataset.items()}
    assert np.all(np.diff(subset['time']) > 0), 'Time is not monotonically increasing!'
    np.savez(os.path.join(folder, filename), **subset)
    return
  
  # remove any train files that might be there
  toRemove = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith('train') and f.endswith('.npz')]
  for f in toRemove: os.remove(f)
  
  for i, training in enumerate(trainingSets):
    name = 'train-%d.npz' % i if 1 < len(trainingSets) else 'train.npz'
    saveSubset(name, training)
    continue
  saveSubset('test.npz', testing)
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Preprocess the dataset')
  parser.add_argument('--time-delta', type=float, default=1.0, help='Time delta in seconds')
  parser.add_argument('--test-ratio', type=float, default=0.2, help='Ratio of testing samples')
  parser.add_argument('--frames-per-chunk', type=int, default=25, help='Number of frames per chunk')
  parser.add_argument('--test-padding', type=int, default=5, help='Number of frames to skip at the beginning/end of each session')
  parser.add_argument('--seed', type=int, default=42, help='Random seed')
  parser.add_argument(
    '--skipped-frames', type=str, default='train', choices=['train', 'test', 'drop'],
    help='What to do with skipped frames ("train", "test", or "drop")'
  )
  parser.add_argument('--splits', type=int, default=1, help='Number of splits of training set')
  args = parser.parse_args()
  main(args)
  pass
