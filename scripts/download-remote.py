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
  # read header (uint8)
  version = np.frombuffer(buffer, dtype=np.uint8, count=1, offset=offset)[0]
  if not (version in [2]): # only version 2 is supported
    raise ValueError('Invalid version %d' % version[0])
  offset += 1

  userId = buffer[offset:offset+36].decode('utf-8')
  offset += 36
  placeId = buffer[offset:offset+36].decode('utf-8')
  offset += 36
  screenId = buffer[offset:offset+36].decode('utf-8')
  offset += 36

  EYE_SIZE = 32 if 1 == version else 48
  # read samples
  while offset < len(buffer):
    sample = {
      'userId': userId,
      'placeId': placeId,
      'screenId': screenId,
    }
    
    # Read time (uint32)
    time_data = np.frombuffer(buffer, dtype='>I', count=1, offset=offset)
    sample['time'] = time_data[0]
    offset += 4
    
    # Read leftEye (uint8)
    EYE_COUNT = EYE_SIZE * EYE_SIZE
    sample['leftEye'] = np.frombuffer(buffer, dtype=np.uint8, count=EYE_COUNT, offset=offset) \
      .reshape(EYE_SIZE, EYE_SIZE)
    offset += EYE_COUNT
    
    # Read rightEye (uint8)
    sample['rightEye'] = np.frombuffer(buffer, dtype=np.uint8, count=EYE_COUNT, offset=offset) \
      .reshape(EYE_SIZE, EYE_SIZE)
    offset += EYE_COUNT
    
    # Read points (float32)
    sample['points'] = np.frombuffer(buffer, dtype='>f4', count=2*478, offset=offset) \
      .reshape(478, 2)
    assert np.all(-2 <= sample['points']) and np.all(sample['points'] <= 2), 'Invalid points'
    offset += 4 * 2 * 478
    
    # Read goal (float32)
    sample['goal'] = goal = np.frombuffer(buffer, dtype='>f4', count=2, offset=offset)
    offset += 4 * 2
    
    if (-2 < goal).all() and (goal < 2).all():
      samples.append(sample)
    else:
      print('Invalid goal:', goal)
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
  if 1 == version: # upscale to 48x48
    import cv2
    res['left eye'] = np.stack(
      [cv2.resize(img[..., None], (48, 48)) for img in res['left eye']]
    )
    res['right eye'] = np.stack([
      cv2.resize(img[..., None], (48, 48)) for img in res['right eye']
    ])
    pass
  
  assert res['left eye'].shape[1:] == (48, 48), 'Invalid shape for left eye. Got %s' % str(res['left eye'].shape)
  assert res['right eye'].shape[1:] == (48, 48), 'Invalid shape for right eye. Got %s' % str(res['right eye'].shape)
  return res

def find_free_name(folder, base_name, extension=".npz"):
  counter = 0
  while True:
    if counter == 0:
      file_name = f"{base_name}{extension}"
    else:
      file_name = f"{base_name}_{counter}{extension}"
    
    file_path = os.path.join(folder, file_name)
    if not os.path.exists(file_path):
      return file_path
    counter += 1

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
  fname = find_free_name(myfolder, str(start_time), extension='.npz')
  np.savez_compressed(fname, **samples)
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

def fetch(cache=None):
  def fromServer(url):
    response = requests.get(url)
    return IO.BytesIO(response.content), False
  
  def cached(url):
    name = os.path.basename(url)
    cache_file = os.path.join(cache, name)
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as f:
        return IO.BytesIO(f.read()), True # read from the cache
      
    response, _ = fromServer(url)
    with open(cache_file, 'wb') as f: # save to the cache
      f.write(response)
    return response, False
  
  if cache is not None:
    return cached
  return fromServer

def main(args):
  # Clear the folder
  shutil.rmtree(os.path.join(folder, 'remote'), ignore_errors=True)
  # get the list of files from the remote server
  urls = requests.get(args.url).json()
  N = len(urls)
  L = len(str(N))
  print('Found %d files on the remote server' % N)
  fetcher = fetch(args.cache)
  for i, file in enumerate(urls):
    content, isCached = fetcher(file)
    # read first file in the gz archive
    with gzip.open(content, 'rb') as f:
      first_file = f.read()
      samples = deserialize(first_file)
      src = 'cache' if isCached else file
      print(f'[{i:0{L}d}/{N:0{L}d}] Readed {len(samples["time"])} samples from {src}')

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
  parser.add_argument('--cache', type=str, help='Path to the cache folder', default=None)

  args = parser.parse_args()
  main(args)
  pass