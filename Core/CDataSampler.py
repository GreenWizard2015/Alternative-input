import numpy as np
import random
from math import ceil
import Core.Utils as Utils
from functools import lru_cache
import Core.CDataSampler_utils as DSUtils

class CDataSampler:
  def __init__(self, storage, batch_size, minFrames, defaults={}, maxT=1.0, cumulative_time=True):
    '''
    If cumulative_time is True, then time is a cumulative time from the start of the trajectory i.e. [0, 0.1, 0.2, 0.3, ...]
    If cumulative_time is False, then time is a time delta between frames i.e. [0, 0.1, 0.1, 0.1, ...]
    '''
    self._storage = storage
    self._defaults = defaults
    self._batchSize = batch_size
    self._maxT = maxT
    self._minFrames = minFrames
    self._samples = []
    self._currentSample = None
    self._cumulative_time = cumulative_time
    return
  
  def reset(self):
    random.shuffle(self._samples)
    self._currentSample = 0
    return

  def __len__(self):
    return ceil(len(self._samples) / self._batchSize)

  def _storeSample(self, idx):
    # store sample if it has enough frames
    minInd = self._getTrajectoryBefore(idx)
    if self._minFrames <= (idx - minInd):
      self._samples.append(idx)
    return
  
  def add(self, sample):
    idx = self._storage.add(sample)
    self._storeSample(idx)
    return idx
  
  def addBlock(self, samples):
    indexes = self._storage.addBlock(samples)
    for idx in indexes:
      self._storeSample(idx)
      continue
    return

  def _getTrajectoryBefore(self, mainInd):
    mainT = self._storage[mainInd]['time']
    minT = mainT - self._maxT
    
    minInd = mainInd
    for ind in range(mainInd - 1, -1, -1):
      if self._storage[ind]['time'] < minT: break
      minInd = ind
      continue
    return minInd
  
  @lru_cache(None)
  def _trajectoryRange(self, mainInd):
    '''
      Returns indexes of samples that are in the range of maxT from the mainInd
      Returns (minInd, maxInd) where minInd <= mainInd <= maxInd
    '''
    mainT = self._storage[mainInd]['time']
    maxT = mainT + self._maxT
    maxInd = mainInd
    for ind in range(mainInd, len(self._storage)):
      if maxT < self._storage[ind]['time']: break
      maxInd = ind
      continue
    
    minInd = self._getTrajectoryBefore(mainInd)
    return minInd, maxInd

  def _trajectory(self, mainInd):
    minInd, maxInd = self._trajectoryRange(mainInd)
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

  def _prepareT(self, res):
    T = np.array([self._storage[ind]['time'] for ind in res])
    T -= T[0]
    diff = np.diff(T, 1)
    idx = np.nonzero(diff)[0]
    if len(idx) < 1: return None # all frames have the same time
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
    # T is an array of time deltas like [0, 0.1, 0.1, 0.1, ...], convert it to cumulative time
    if self._cumulative_time:
      T = np.cumsum(T)
    return T
  
  def _framesFor(self, mainInd, samples, steps, stepsSampling):
    if 'uniform' == stepsSampling:
      samples = random.sample(samples, steps - 1)
    if 'last' == stepsSampling:
      samples = samples[-(steps - 1):]
      
    if isinstance(stepsSampling, dict):
      candidates = list(samples)
      maxFrames = stepsSampling['max frames']
      candidates = candidates[::-1]
      samples = []
      left = steps - 1
      for _ in range(left):
        avl = min((maxFrames, 1 + len(candidates) - left))
        ind = random.randint(0, avl - 1)
        samples.append(candidates[ind])
        candidates = candidates[ind+1:]
        left -= 1
        continue
      pass
      
    res = list(sorted(samples + [mainInd]))
    assert len(res) == steps
    return res
  
  def _stepsFor(self, mainInd, steps, stepsSampling='uniform', **_):
    if (steps is None) or (1 == steps): return [(mainInd, 0.0)]
    if mainInd < steps: return False
    
    samples, _ = self._trajectory(mainInd)
    if len(samples) < (steps - 1): return False
    # Try to sample valid frames
    for _ in range(10):
      res = self._framesFor(mainInd, samples, steps, stepsSampling)
      T = self._prepareT(res)
      if T is not None:
        assert len(res) == len(T)
        return [tuple(x) for x in zip(res, T)]
      continue
    return False
  
  def sample(self, **kwargs):
    kwargs = {**self._defaults, **kwargs}
    timesteps = kwargs.get('timesteps', None)
    N = kwargs.get('N', self._batchSize)
    indexes = []
    for _ in range(N):
      added = False
      while not added:
        idx = self._samples[self._currentSample]
        self._currentSample = (self._currentSample + 1) % len(self._samples)

        sampledSteps = self._stepsFor(idx, steps=timesteps, **kwargs)
        if sampledSteps:
          # TODO: remove from samples?
          indexes.extend(sampledSteps)
          added = True
      continue

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
  def _targetFor(self, ind, keypoints=1, past=True, future=True, **_):
    before, after = self._trajectory(ind)
    if not past: before = []
    if not future: after = []
    return self._trajectory2keypoints(before, ind, after, N=keypoints)

  def _indexes2XY(self, indexesAndTime, kwargs):
    timesteps = kwargs.get('timesteps', None)
    samples = [self._storage[i] for i, _ in indexesAndTime]

    forecast = kwargs.get('forecast', {})
    Y = ( np.array([
      self._targetFor(i, **forecast) 
      for i, _ in indexesAndTime
    ], np.float32), )
    Y = self._reshapeSteps(Y, timesteps)
    ##############
    userIds = np.unique([x['userId'] for x in samples])
    assert 1 == len(userIds), 'Only one user is supported. Found: ' + str(userIds)
    placeIds = np.unique([x['placeId'] for x in samples])
    assert 1 == len(placeIds), 'Only one place is supported. Found: ' + str(placeIds)
    screenIds = np.unique([x['screenId'] for x in samples])
    assert 1 == len(screenIds), 'Only one screen is supported. Found: ' + str(screenIds)

    X = DSUtils.toTensor(
      (
        np.array([x['points'] for x in samples], np.float32),
        np.array([x['left eye'] for x in samples]),
        np.array([x['right eye'] for x in samples]),
        np.array([T for _, T in indexesAndTime], np.float32).reshape((-1, 1)),
      ),
      (
        kwargs.get('pointsNoise', 0.0),
        kwargs.get('pointsDropout', 0.0),
      
        kwargs.get('eyesAdditiveNoise', 0.0),
        kwargs.get('eyesDropout', 0.0),
        kwargs.get('brightnessFactor', 0.0),
        kwargs.get('lightBlobFactor', 0.0),

        timesteps
      ),
      userIds[0], placeIds[0], screenIds[0]
    )
    ###############
    (Y, ) = Y
    return(X, (Y.astype(np.float32), ))

  @property
  def totalSamples(self):
    return len(self._storage)
  
  def validSamples(self):
    return list(sorted(self._samples))
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
  ds = CDataSampler( CSamplesStorage(), balancingMethod=dict(context='all') )
  dsBlock = Utils.datasetFrom(os.path.join(folder, 'Data', 'Dataset'))
  ds.addBlock(dsBlock)
  exit(0)