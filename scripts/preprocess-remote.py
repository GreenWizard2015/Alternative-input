#!/usr/bin/env python
# -*- coding: utf-8 -*-.
'''
This script performs the following steps:
  1. Load the npz files recursively from the "Data/remote" folder
  2. Combine the npz files into a single dataset
  3. Drop the "userId", "placeId", and "screenId" fields
  4. Split the dataset into "sessions" (i.e. consecutive frames) with a gap of N seconds
  5. Split each session into training and testing sets
  6. Save the training.npz and testing.npz files
  7. Remove the npz files from the current folder

  Also, the script generates "Data/remote/stats.json" file with the following structure:
  {
    "placeId": [ ... ],
    "userId": [ ... ],
    "screenId": [ ... ],
  }
'''

import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

import numpy as np
import Core.Utils as Utils
import json

def loadNpz(path):
  res = Utils.datasetFrom(path)
  # validate the dataset and remove the "userId", "placeId", and "screenId" fields
  for nm in ['userId', 'placeId', 'screenId']:
    if nm in res:
      v = res.pop(nm)
      v = np.unique(v)
      assert 1 == len(v), 'Expecting a single value for %s' % nm

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
  # split training into splits
  trainingN = len(training)
  # shuffle training sets using the numpy methods
  np.random.shuffle(training)
  ####
  print(
    'Session {}-{}: {} chunks, {} training, {} testing'.format(
      start, end, len(chunks),
      trainingN,
      len(testing)
    )
  )
  # remove padding
  def F(x):
    if len(x) == 0: return []
    x = x.reshape(-1) # flatten
    return x[x != -1] # remove padding
  return F(training), F(testing)

def splitDataset(dataset, ratio, framesPerChunk, skipAction):
  # split each session into training and testing sets
  trainingSet = [] # list of (start, end) tuples for each split
  testing = [] # list of (start, end) tuples
  for i, (start, end) in enumerate(dataset):
    trainingIdx = testingIdx = []
    if (end - start) < 2 * framesPerChunk:
      # print('Session %d is too short. Action: %s' % (i, skipAction))
      if 'drop' == skipAction: continue

      rng = np.arange(start, end)
      if 'train' == skipAction: trainingIdx = rng
      if 'test' == skipAction: testingIdx = rng
    else:
      trainingIdx, testingIdx = splitSession(start, end, ratio, framesPerChunk)

    # store training and testing sets if they are not empty
    if 0 < len(trainingIdx): trainingSet.append(trainingIdx)
    if 0 < len(testingIdx): testing.append(testingIdx)
    continue
  if (0 == len(trainingSet)) or (0 == len(testing)):
    print('No training or testing sets was created!')
    return [], []
  # save training and testing sets
  testing = np.sort(np.concatenate(testing))
  training = np.sort(np.concatenate(trainingSet))
  
  # check that training and testing sets are disjoint
  intersection = np.intersect1d(training, testing)
  if 0 < len(intersection):
    print('Training and testing sets are not disjoint!')
    print(intersection)
    raise Exception('Training and testing sets are not disjoint!')

  return training, testing

def dropPadding(idx, padding):
  res = []
  if len(idx) < 2: return res
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
  res = np.concatenate(res) if 0 < len(res) else []
  print('Frames before: {}. Frames after: {}'.format(len(idx), len(res)))
  return res

def processFolder(folder, timeDelta, testRatio, framesPerChunk, testPadding, skippedFrames, minimumFrames, dropZeroDeltas):
  print('Processing', folder)
  # load all.npz file if it exists
  all_file = os.path.join(folder, 'all.npz')
  if os.path.exists(all_file):
    dataset = loadNpz(all_file)
  else:
    dataset = loadNpz(folder)
    np.savez(all_file, **dataset)

  # remove the npz files, except for all.npz
  files = os.listdir(folder)
  for fn in files:
    if fn.endswith('.npz') and not ('all.npz' == fn):
      os.remove(os.path.join(folder, fn))
  print('Removed', len(files), 'files')

  if dropZeroDeltas: # drop frames with zero time deltas
    deltas = np.diff(dataset['time'])
    idx = np.where(0 == deltas)[0]
    print('Dropping {} frames with zero time deltas'.format(len(idx)))
    dataset = {k: np.delete(v, idx) for k, v in dataset.items()}

  N = len(dataset['time'])    
  # print total deltas statistics
  print('Dataset: {} frames'.format(N))
  deltas = np.diff(dataset['time'])
  print('Total time deltas: min={}, max={}, mean={}'.format(np.min(deltas), np.max(deltas), np.mean(deltas)))
  deltas = None

  if N < minimumFrames:
    print('Dataset is too short. Skipping...')
    return 0, 0, True
  # split dataset into sessions
  sessions = Utils.extractSessions(dataset, float(timeDelta))
  # print sessions and their durations for debugging
  print('Found {} sessions'.format(len(sessions)))
  for i, (start, end) in enumerate(sessions):
    idx = np.arange(start, end)
    session_time = dataset['time'][idx]
    delta = np.diff(session_time)
    duration = session_time[-1] - session_time[0]
    # print also min, max, and mean time deltas
    print('Session {} - {}: min={}, max={}, mean={}, frames={}, duration={} sec'.format(
      start, end, np.min(delta), np.max(delta), np.mean(delta), len(session_time), duration
    ))
    continue
  ######################################################
  # split each session into training and testing sets
  training, testing = splitDataset(
    sessions,
    ratio=1.0 - float(testRatio),
    framesPerChunk=int(framesPerChunk),
    skipAction=skippedFrames,
  )
  if 0 < testPadding:
    testing = dropPadding(testing, testPadding)

  if (0 == len(training)) or (0 == len(testing)):
    print('No training or testing sets found!')
    return 0, 0, True

  def saveSubset(filename, idx):
    print('%s: %d frames' % (filename, len(idx)))
    subset = {k: v[idx] for k, v in dataset.items()}
    time = subset['time']
    diff = np.diff(time)
    assert np.all(diff >= 0), 'Time is not monotonically increasing!'
    np.savez(os.path.join(folder, filename), **subset)
    return
  # save training and testing sets 
  saveSubset('train.npz', training)
  saveSubset('test.npz', testing)

  print('Processing ', folder, 'done')
  return len(testing), len(training), False

def main(args):
  stats = {
    'placeId': [],
    'userId': [],
    'screenId': [],
  }
  testFrames = trainFrames = 0
  framesPerChunk = {}
  # subfolders: PlaceId -> UserId -> ScreenId -> start_time.npz
  folder = args.folder
  foldersList = lambda x: [nm for nm in os.listdir(x) if os.path.isdir(os.path.join(x, nm))]
  subfolders = foldersList(folder)
  for placeId in subfolders:
    if not (placeId in stats['placeId']):
      stats['placeId'].append(placeId)
    userIds = foldersList(os.path.join(folder, placeId))
    for userId in userIds:
      if not (userId in stats['userId']):
        stats['userId'].append(userId)
      screenIds = foldersList(os.path.join(folder, placeId, userId))
      for screenId in screenIds:
        sid = '%s/%s' % (placeId, screenId)
        if not (sid in stats['screenId']):
          stats['screenId'].append(sid)
        path = os.path.join(folder, placeId, userId, screenId)
        testFramesN, trainFramesN, isSkipped = processFolder(
          path, 
          args.time_delta, args.test_ratio, args.frames_per_chunk,
          args.test_padding, args.skipped_frames,
          minimumFrames=args.minimum_frames,
          dropZeroDeltas=args.drop_zero_deltas
        )
        if not isSkipped:
          testFrames += testFramesN
          trainFrames += trainFramesN
          # store the number of frames per chunk
          sid = '%s/%s/%s' % (placeId, userId, screenId)
          framesPerChunk[sid] = testFramesN + trainFramesN
      continue
  print('Total: %d training frames, %d testing frames' % (trainFrames, testFrames))

  # save the stats
  with open(os.path.join(folder, 'stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

  print('-' * 80)
  for k, v in framesPerChunk.items():
    print('%s: %d frames' % (k, v))
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Preprocess the dataset')
  parser.add_argument(
    '--folder', type=str, 
    default=os.path.join(ROOT_FOLDER, 'Data', 'remote'),
    help='Folder with the npz files'
  )
  parser.add_argument('--time-delta', type=float, default=3.0, help='Time delta in seconds')
  parser.add_argument('--test-ratio', type=float, default=0.2, help='Ratio of testing samples')
  parser.add_argument('--frames-per-chunk', type=int, default=25, help='Number of frames per chunk')
  parser.add_argument('--test-padding', type=int, default=5, help='Number of frames to skip at the beginning/end of each session')
  parser.add_argument(
    '--skipped-frames', type=str, default='train', choices=['train', 'test', 'drop'],
    help='What to do with skipped frames ("train", "test", or "drop")'
  )
  parser.add_argument('--minimum-frames', type=int, default=0, help='Minimum number of frames in a dataset')
  parser.add_argument('--drop-zero-deltas', action='store_true', help='Drop frames with zero time deltas')
  args = parser.parse_args()
  main(args)
  pass
