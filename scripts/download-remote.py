#!/usr/bin/env python
# -*- coding: utf-8 -*-.
'''
Downloads the data from the remote server and saves it in the local folder
In structure:
Data
  remote
    PlaceId
      UserId
        ScreenId
          start_time.npz
'''
import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

import io as IO
import gzip
import shutil
import numpy as np
import requests

folder = os.path.join(ROOT_FOLDER, 'Data')

def deserialize(buffer):
  offset = 0
  samples = []
  while offset < len(buffer):
    sample = {}
    
    # Read time (uint32)
    time_data = np.frombuffer(buffer, dtype='>I', count=1, offset=offset)
    sample['time'] = time_data[0]
    offset += 4
    
    # Read leftEye (1024 uint8)
    sample['leftEye'] = np.frombuffer(buffer, dtype=np.uint8, count=32*32, offset=offset) \
      .reshape(32, 32)
    offset += 32 * 32
    
    # Read rightEye (1024 uint8)
    sample['rightEye'] = np.frombuffer(buffer, dtype=np.uint8, count=32*32, offset=offset) \
      .reshape(32, 32)
    offset += 32 * 32
    
    # Read points (float32)
    sample['points'] = np.frombuffer(buffer, dtype='>f4', count=2*478, offset=offset) \
      .reshape(478, 2)
    assert np.all(-2 <= sample['points']) and np.all(sample['points'] <= 2), 'Invalid points'
    offset += 4 * 2 * 478
    
    # Read goal (float32)
    sample['goal'] = goal = np.frombuffer(buffer, dtype='>f4', count=2, offset=offset)
    offset += 4 * 2
    
    # Read userId (36 bytes as string)
    sample['userId'] = buffer[offset:offset+36].decode('utf-8')
    offset += 36
    
    # Read placeId (36 bytes as string)
    sample['placeId'] = buffer[offset:offset+36].decode('utf-8')
    offset += 36
    
    # Read screenId (int32)
    sample['screenId'] = buffer[offset:offset+36].decode('utf-8')
    offset += 36
    
    if (-2 < sample['goal']).all() and (sample['goal'] < 2).all():
      samples.append(sample)
    else:
      print('Invalid goal:', sample['goal'])
    continue

  # Transpose them to the make columnwise
  res = {}
  for k in samples[0].keys():
    res[k] = np.array([sample[k] for sample in samples])
    assert res[k].shape[0] == len(samples), 'Invalid shape for %s' % k
    continue
  # convert the time to float32
  res['time'] = res['time'].astype(np.float32) / 1000.0
  # rename "leftEye" and "rightEye" to "left eye" and "right eye"
  res['left eye'] = res.pop('leftEye')
  res['right eye'] = res.pop('rightEye')
  return res

def saveChunk(samples, folder):
  # check time is increasing monotonically
  time = samples['time']
  assert np.all(time[1:] >= time[:-1]), 'The time should be increasing monotonically'

  # save the samples in '{placeId}/{userId}/{screenId}/{start_time}.npz'
  userId = np.unique(samples['userId'])
  assert 1 == len(userId), 'Expecting a single userId'
  placeId = np.unique(samples['placeId'])
  assert 1 == len(placeId), 'Expecting a single placeId'
  screenId = np.unique(samples['screenId'])
  assert 1 == len(screenId), 'Expecting a single screenId'

  myfolder = os.path.join(
    folder, placeId[0], userId[0], str(screenId[0])
  )
  if not os.path.exists(myfolder): os.makedirs(myfolder, exist_ok=True)
  start_time = samples['time'][0]
  np.savez_compressed(os.path.join(myfolder, str(start_time) + '.npz'), **samples)
  return

def splitByID(samples):
  res = {}
  keys = list(samples.keys())
  N = len(samples['time'])
  for i in range(N):
    sample = {k: samples[k][i] for k in keys} # copy
    userId = sample['userId']
    placeId = sample['placeId']
    screenId = sample['screenId']
    key = '%s/%s/%s' % (placeId, userId, screenId)
    if key not in res:
      res[key] = []
    res[key].append(sample)
    continue

  # transpose them to the make columnwise
  lst = list(res.values())
  res = []
  for values in lst:
    newValues = {}
    for k in keys:
      newValues[k] = np.array([sample[k] for sample in values])
      continue
    res.append(newValues)
    continue

  return res

def main(args):
  # Clear the folder
  shutil.rmtree(os.path.join(folder, 'remote'), ignore_errors=True)
  # get the list of files from the remote server
  urls = requests.get(args.url).json()
  print('Found %d files on the remote server' % len(urls))
  for file in urls:
    response = requests.get(file)
    content = IO.BytesIO(response.content)
    # read first file in the gz archive
    with gzip.open(content, 'rb') as f:
      first_file = f.read()
      samples = deserialize(first_file)
      print('Read %d samples from %s' % (len(samples['time']), file))

      # don't want to messing up with such cases
      userId = np.unique(samples['userId'])
      placeId = np.unique(samples['placeId'])
      screenId = np.unique(samples['screenId'])
      
      chunks = [samples]
      needSplit = (1 < len(userId)) or (1 < len(placeId)) or (1 < len(screenId))
      if needSplit:
        chunks = splitByID(samples)

      for chunk in chunks:
        saveChunk(chunk, os.path.join(folder, 'remote'))
        continue
      pass

  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--url', type=str, help='URL to the list of files')

  args = parser.parse_args()
  main(args)
  pass