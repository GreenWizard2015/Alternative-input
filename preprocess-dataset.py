#!/usr/bin/env python
# -*- coding: utf-8 -*-.
'''
This script performs the following steps:
  1. Load the dataset from the source folder
  2. Split the dataset into "sessions" (i.e. consecutive frames) with a gap of N seconds
  3. Split each session into training and testing sets
  4. Save the training.npz and testing.npz files
'''
import numpy as np
import os
import Core.Utils as Utils
import argparse

def extractSessions(dataset, TDelta):
  res = []
  T = 0
  prevSession = 0
  for i, t in enumerate(dataset['time']):
    if TDelta < (t - T):
      if 1 < (i - prevSession):
        res.append((prevSession, i))
      prevSession = i
      pass
    T = t
    continue
  # if last session is not empty, then append it
  if prevSession < len(dataset['time']):
    res.append((prevSession, len(dataset['time'])))
    pass
  return res

def splitSession(start, end, ratio, framesPerChunk):
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
  print(
    'Session {}-{}: {} chunks, {} training, {} testing'.format(
      start, end, len(chunks), len(training), len(testing)
    )
  )
  # remove padding
  F = lambda x: np.sort(x[x != -1].reshape(-1))
  training = F(training)
  testing = F(testing)
  return training, testing

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
  MAIN_FOLDER = os.path.join(os.path.dirname(__file__), 'Data')
  src = os.path.join(MAIN_FOLDER, 'Dataset')
  # load dataset
  dataset = Utils.datasetFrom(src)
  # split dataset into sessions
  sessions = extractSessions(dataset, float(args.time_delta))
  # print sessions and their durations for debugging
  print('Found {} sessions'.format(len(sessions)))
  for i, (start, end) in enumerate(sessions):
    duration = dataset['time'][end - 1] - dataset['time'][start]
    print('Session {}: {} - {} ({}, {})'.format(i, start, end, end - start, duration))
    continue
  ######################################################
  # split each session into training and testing sets
  ratio = 1.0 - float(args.test_ratio)
  framesPerChunk = int(args.frames_per_chunk)
  training = [] # list of (start, end) tuples
  testing = [] # list of (start, end) tuples
  for i, (start, end) in enumerate(sessions):
    N = end - start
    if N < 2 * framesPerChunk:
      print('Session {} is too short, skipping'.format(i))
      continue
    trainingIdx, testingIdx = splitSession(start, end, ratio, framesPerChunk)
    training.append(trainingIdx)
    testing.append(testingIdx)
    continue
  # save training and testing sets
  training = np.sort(np.concatenate(training))
  testing = np.sort(np.concatenate(testing))

  # check that training and testing sets are disjoint
  intersection = np.intersect1d(training, testing)
  if 0 < len(intersection):
    print('Training and testing sets are not disjoint!')
    print(intersection)
    raise Exception('Training and testing sets are not disjoint!')
  
  testPadding = int(args.test_padding)
  if 0 < testPadding:
    testing = dropPadding(testing, testPadding)

  print('Training set: {} frames'.format(len(training)))
  print('Testing set: {} frames'.format(len(testing)))

  trainingData = {k: v[training] for k, v in dataset.items()}
  assert np.all(np.diff(trainingData['time']) > 0), 'Time is not monotonically increasing!'
  np.savez(os.path.join(MAIN_FOLDER, 'train.npz'), **trainingData)

  testingData = {k: v[testing] for k, v in dataset.items()}
  assert np.all(np.diff(testingData['time']) > 0), 'Time is not monotonically increasing!'
  np.savez(os.path.join(MAIN_FOLDER, 'test.npz'), **testingData)
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Preprocess the dataset')
  parser.add_argument('--time-delta', type=float, default=1.0, help='Time delta in seconds')
  parser.add_argument('--test-ratio', type=float, default=0.2, help='Ratio of testing samples')
  parser.add_argument('--frames-per-chunk', type=int, default=25, help='Number of frames per chunk')
  parser.add_argument('--test-padding', type=int, default=5, help='Number of frames to skip at the beginning/end of each session')
  args = parser.parse_args()
  main(args)
  pass
