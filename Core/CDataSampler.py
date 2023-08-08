import numpy as np
import random
import Core.Utils as Utils
from functools import lru_cache
import collections
import Core.CDataSampler_utils as DSUtils

class CDataSampler:
  def __init__(self, storage, defaults={}):
    self._storage = storage
    self._defaults = defaults
    self._samplesByHash = {}
    self._mainHashes = []
    return

  def __len__(self):
    return len(self._storage)
  
  def _hashesFor(self, goal):
    goal = np.array(goal) - 0.5 # centered
    # return [ str(np.trunc(goal * s)) for s in [3, 7, 17, 37]]
    RHash = '%d' % (np.sqrt(np.square(goal).sum()) / 0.1, )
    # angel in degrees
    angel = np.arctan2(goal[1], goal[0]) / np.pi * 180
    AHash = '%d' % (angel / 10, )
    return [RHash, AHash]
    
  def _bucketFor(self, *args, **kwargs):
    hashes = self._hashesFor(*args, **kwargs)
    #######
    hashA = hashes[0]
    if not(hashA in self._mainHashes):
      self._mainHashes.append(hashA)
    
    s = self._samplesByHash
    for h in hashes[:-1]:
      if not(h in s): s[h] = {}
      s = s[h]
      continue
    h = hashes[-1]
    if not(h in s): s[h] = []
    return s[h]

  def add(self, sample):
    idx = self._storage.add(sample)
    sample = self._storage[idx]
    self._bucketFor(
      goal=sample['goal'],
    ).append(idx)
    return idx
  
  def addBlock(self, samples):
    indexes = self._storage.addBlock(samples)
    for idx in indexes:
      sample = self._storage[idx]
      self._bucketFor(
        goal=sample['goal'],
      ).append(idx)
      continue
    return
  
  def _sampleIndexFrom(self, bucketID):
    samples = self._samplesByHash[bucketID]
    while isinstance(samples, dict):
      key = random.choice(list(samples.keys()))
      samples = samples[key]
      continue
    return random.choice(samples)

  @lru_cache(None)
  def _trajectoryRange(self, mainInd, maxT):
    mainT = self._storage[mainInd]['time']
    minT = mainT - maxT
    maxT = mainT + maxT
    
    minInd = maxInd = mainInd
    for ind in range(mainInd - 1, -1, -1):
      if self._storage[ind]['time'] < minT: break
      minInd = ind
      continue

    for ind in range(mainInd, len(self._storage)):
      if maxT < self._storage[ind]['time']: break
      maxInd = ind
      continue
    
    return minInd, maxInd

  def _trajectory(self, mainInd, maxT):
    minInd, maxInd = self._trajectoryRange(mainInd, maxT)
    return list(range(minInd, mainInd)), list(range(mainInd + 1, maxInd + 1))
  
  def _trajectory2keypoints(self, before, mainInd, after, N):
    mainPt = self._storage[mainInd]['goal']
    if 1 < N:
      trajectory = []
      trajectory.extend(before)
      trajectory.append(mainInd)
      trajectory.extend(after)
      trajectory = np.array([self._storage[ind]['goal'] for ind in trajectory])
      chunksN = max((1, len(trajectory) // N))
      keypoints = [(mainPt, 0.0)]
      for i in range(0, len(trajectory), chunksN):
        x = trajectory[i:i+chunksN]
        if 0 < len(x):
          pt = np.mean(x, axis=0)
          d = np.linalg.norm(pt - mainPt)
          keypoints.append((pt, d))
        continue
      while len(keypoints) < N: keypoints.append(keypoints[0])
      keypoints = sorted(keypoints, key=lambda x: x[1])
      keypoints = [pt for pt, _ in keypoints]
      keypoints = np.array(keypoints[:N])
    else:
      keypoints = np.array([mainPt])
    return keypoints

  def _stepsFor(self, mainInd, steps, stepsSampling='uniform', maxT=1.0, **_):
    if (steps is None) or (1 == steps): return [mainInd]
    if mainInd < steps: return False
    
    samples, _ = self._trajectory(mainInd, maxT)
    if len(samples) < (steps - 1): return False
    if 'uniform' == stepsSampling:
      samples = random.sample(samples, steps - 1)
    if 'last' == stepsSampling:
      samples = samples[-(steps - 1):]
      
    if isinstance(stepsSampling, dict):
      candidates = list(samples)
      maxFrames = stepsSampling['max frames']
      shift = 1
      if stepsSampling.get('include last', True):
        candidates.append(mainInd)
        shift = 0
      candidates = candidates[::-1]
      samples = []
      left = steps - 1
      for _ in range(left):
        avl = min((maxFrames, 1 + len(candidates) - left))
        ind = random.randint(0, avl - 1)
        samples.append(candidates[ind])
        candidates = candidates[ind+shift:]
        left -= 1
        continue
      pass
      
    res = list(sorted(samples + [mainInd]))
    assert len(res) == steps
    ###########################
    T = np.array([self._storage[ind]['time'] for ind in res])
    T -= T[0]
    diff = np.diff(T, 1)
    idx = np.nonzero(diff)[0]
    if len(idx) < 2: return False
    if len(diff) == len(idx):
      T = diff
    else:
      # avg non-zero diff
      dT = np.min(diff[idx])
      T = np.append(T, T[-1] + dT)
      idx = [0, *(1 + idx), len(T) - 1]
      T = np.interp(np.arange(len(T) - 1), idx, T[idx])
      T = np.diff(T, 1)
      pass
    T = np.insert(T, 0, 0.0)
    assert len(res) == len(T)
    return [tuple(x) for x in zip(res, T)]
  
  def _seedsStream(self, N):
    while 0 < len(self._mainHashes):
      seeds = random.choices(self._mainHashes, k=2*N)
      for value, _ in collections.Counter(seeds).most_common():
        yield value
      continue
    return
  
  def sample(self, N, **kwargs):
    kwargs = {**self._defaults, **kwargs}
    timesteps = kwargs.get('timesteps', None)

    indexes = []
    tries = 10 * N
    for bucketID in self._seedsStream(N):
      tries -= 1
      if tries <= 0: return None
      
      indx = self._sampleIndexFrom(bucketID)
      sampledSteps = self._stepsFor( indx, steps=timesteps, **kwargs )
      if sampledSteps:
        indexes.extend(sampledSteps)
        N -= 1
        if N <= 0: break
      continue
    if 0 < N: return None
    
    return self._indexes2XY(indexes, kwargs)

  def sampleById(self, idx, **kwargs):
    kwargs = {**self._defaults, **kwargs}
    timesteps = kwargs.get('timesteps', None)
    sampledSteps = self._stepsFor(idx, steps=timesteps, **kwargs)
    if not sampledSteps: return None
    return self._indexes2XY([*sampledSteps], kwargs)
  
  def checkById(self, idx, **kwargs):
    kwargs = {**self._defaults, **kwargs}
    timesteps = kwargs.get('timesteps', None)
    sampledSteps = self._stepsFor(idx, steps=timesteps, **kwargs)
    if not sampledSteps: return False
    return True
    
  def sampleByIds(self, ids, **kwargs):
    kwargs = {**self._defaults, **kwargs}
    timesteps = kwargs.get('timesteps', None)
    sampledSteps = []
    rejected = []
    accepted = []
    for idx in ids:
      sample = self._stepsFor(idx, steps=timesteps, **kwargs)
      if sample:
        accepted.append(idx)
        sampledSteps.extend(sample)
      else:
        rejected.append(idx)
        pass
      continue
    
    res = None
    if 0 < len(sampledSteps):
      res = self._indexes2XY(sampledSteps, kwargs)
    return res, rejected, accepted
  
  def _reshapeSteps(self, values, steps):
    if steps is None: return values
    
    res = []
    for x in values:
      B, *s = x.shape
      newShape = (B // steps, steps, *s)
      res.append(x.reshape(newShape))
      continue
    return tuple(res)
  
  @lru_cache(None)
  def _targetFor(self, ind, maxT, keypoints=1, past=True, future=True, **_):
    before, after = self._trajectory(ind, maxT=maxT)
    if not past: before = []
    if not future: after = []
    return self._trajectory2keypoints(before, ind, after, N=keypoints)

  def _indexes2XY(self, indexesAndTime, kwargs):
    timesteps = kwargs.get('timesteps', None)
    samples = [self._storage[i] for i, _ in indexesAndTime]

    forecast = kwargs.get('forecast', {})
    maxT = kwargs.get('maxT', 1.0)
    forecast['maxT'] = forecast.get('maxT', maxT)
    Y = ( np.array([
      self._targetFor(i, **forecast) 
      for i, _ in indexesAndTime
    ], np.float32), )
    Y = self._reshapeSteps(Y, timesteps)
    ##############
    X = DSUtils.toTensor(
      (
        np.array([x['points'] for x in samples], np.float32),
        np.array([x['left eye'] for x in samples]),
        np.array([x['right eye'] for x in samples]),
        np.array([T for _, T in indexesAndTime], np.float32).reshape((-1, 1)),
      ),
      (
        kwargs.get('pointsDropout', 0.0),
        kwargs.get('pointsNoise', 0.0),
      
        kwargs.get('eyesAdditiveNoise', 0.0),
        kwargs.get('eyesDropout', 0.0),
        kwargs.get('brightnessFactor', 0.0),
        kwargs.get('lightBlobFactor', 0.0),

        timesteps
      )
    )
    ###############
    (Y, ) = Y
    return(X, (Y.astype(np.float32), ))
  
  @property
  def mainSeedsCount(self):
    return len(self._mainHashes)
  
  def lowlevelSamplesIndexes(self):
    LLBuckets = []
    def F(bucket):
      if isinstance(bucket, dict):
        for v in bucket.values():
          F(v)
        return
      LLBuckets.append(bucket)
      return
    F(self._samplesByHash)
    return LLBuckets

##############
if __name__ == '__main__':
  import tensorflow as tf
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024//2)]
  )
  import os
  from Core.CSamplesStorage import CSamplesStorage
  folder = os.path.dirname(os.path.dirname(__file__))
  ds = CDataSampler( CSamplesStorage() )
  dsBlock = Utils.datasetFrom(os.path.join(folder, 'Data', 'Dataset'))
  ds.addBlock(dsBlock)
  # print count of samples in each bucket
  for lA, vA in ds._samplesByHash.items():
    print(lA, len(vA))
    # for lB, vB in vA.items():
    #   print('  ', lB, len(vB))
  exit(0)

  import matplotlib.pyplot as plt
  data = ds._getShifts(512)
  plt.scatter(data[:, 0], data[:, 1], .3)
  plt.axis('equal')
  plt.show()
  exit(0)

  # blankLeft = np.all(dsBlock['left eye'] < 0.1*255, axis=(1, 2))
  # blankRight = np.all(dsBlock['right eye'] < 0.1*255, axis=(1, 2))
  # print('Blank left', blankLeft.sum())
  # print('Blank right', blankRight.sum())
  # both = np.logical_and(blankLeft, blankRight)
  # print('Both', both.sum())
  # exit(0)

  xy = ds.sample(
    4, timesteps=5, 
    stepsSampling={'max frames': 5, 'include last': True},
    forecast=dict(
      past=True, future=True, maxT=2.,
      keypoints=128
    ),
    shiftsN=10,
  )
  xy = ds.sample(
    4, timesteps=5, 
    stepsSampling={'max frames': 5, 'include last': True},
    forecast=dict(
      past=True, future=True, maxT=2.,
      keypoints=128
    ),
    shiftsN=10,
  )
  for v in xy[:1]:
    for k, x in v['clean'].items():
      print(k, x.shape)
    print('........')
  print(xy[1][0].shape)
  print(xy[0]['clean']['ContextID'])
  exit(0)

  import cv2
  def show(x, nm):
    leftEye = x['left eye'][0, -1].numpy()
    rightEye = x['right eye'][0, -1].numpy()
    combined = np.concatenate([leftEye, rightEye], axis=1)
    # upscale x8
    combined = cv2.resize(combined, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(nm, combined)
    return

  data, _ = ds.sampleById(96, timesteps=5)
  show(data['clean'], 'clean')
  show(data['augmented'], 'augmented 1')
  for i in range(1, 4):
    data, _ = ds.sampleById(96, timesteps=5)
    show(data['augmented'], 'augmented %d' % (i + 1))
    continue
  
  cv2.waitKey(0)
  pass