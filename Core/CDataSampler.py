import numpy as np
import random
import Utils
import matplotlib.pyplot as plt
from functools import lru_cache
import collections

def gaussian(x, mu, sig):
  return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

import tensorflow as tf

@tf.function(
  input_signature=[
    (
      tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(5,), dtype=tf.float32)
  ]
)
def _toTensor(data, params):
  print('Instantiate _toTensor')
  pointsDropout, pointsNoise, eyesAdditiveNoise, eyesDropout, timesteps = tf.unstack(params)
  timesteps = tf.cast(timesteps, tf.int32)
  points, imgA, imgB = data
  N = tf.shape(points)[0]
  imgA = tf.cast(imgA, tf.float32) / 255.
  imgB = tf.cast(imgB, tf.float32) / 255.
  cleanData = (points, imgA, imgB)
  ##########################
  if 0.0 < eyesAdditiveNoise:
    imgA = imgA + tf.random.normal(tf.shape(imgA), stddev=eyesAdditiveNoise)
    imgA = tf.clip_by_value(imgA, 0.0, 1.0)

    imgB = imgB + tf.random.normal(tf.shape(imgB), stddev=eyesAdditiveNoise)
    imgB = tf.clip_by_value(imgB, 0.0, 1.0)
  ##########################
  if 0.0 < eyesDropout:
    mask = tf.random.uniform((N,)) < eyesDropout
    maskA = 0.5 < tf.random.uniform((N,))
    maskB = tf.logical_not(maskA)
    imgA = tf.where(tf.logical_and(mask, maskA)[:, None, None], 0.0, imgA)
    imgB = tf.where(tf.logical_and(mask, maskB)[:, None, None], 0.0, imgB)
  ##########################
  validPointsMask = tf.reduce_all(-1.0 < points, axis=-1, keepdims=True)
  if 0.0 < pointsNoise:
    points += tf.random.normal(tf.shape(points), stddev=pointsNoise)
    
  if 0.0 < pointsDropout:
    mask = tf.random.uniform(tf.shape(points)[:-1])[..., None] < pointsDropout
    points = tf.where(mask, -1.0, points)
  
  points = tf.where(validPointsMask, points, -1.0)
  ##########################
  reshape = lambda x: tf.reshape(
    x,
    tf.concat([(N // timesteps, timesteps), tf.shape(x)[1:]], axis=-1)
  )
  augmented = tuple(reshape(x) for x in (points, imgA, imgB))
  clean = tuple(reshape(x) for x in cleanData)
  return{'augmented': augmented, 'clean': clean}

class CDataSampler:
  def __init__(
    self, storage, defaults={}, goalHashScale=100, 
    debugDistribution=True, adjustDistribution=False
  ):
    self._storage = storage
    self._defaults = defaults
    self._samplesByGoals = {}
    self._mainHashes = []
    self._samplingDistribution = None
    self._adjustDistribution = adjustDistribution
    
    self._goalHashScale = goalHashScale
    self._debugDistribution = debugDistribution
    return

  def __len__(self):
    return len(self._storage)
  
  def add(self, sample):
    idx = self._storage.add(sample)
    goal = sample['goal']
    self._bucketFor(goal).append(idx)
    return idx
  
  def addBlock(self, samples):
    indexes = self._storage.addBlock(samples)
    for idx, goal in zip(indexes, samples['goal']):
      self._bucketFor(goal).append(idx)
      continue
    return
  
  def _sampleIndexFrom(self, bucketID):
    samples = self._samplesByGoals[bucketID]
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
  
  def _trajectory2keypoints(self, before, mainInd, after, N, debug=False):
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
    
    if debug:
      trajectory = before
      trajectory = np.array([self._storage[ind]['goal'] for ind in trajectory])
      plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo')
      
      trajectory = after
      trajectory = np.array([self._storage[ind]['goal'] for ind in trajectory])
      plt.plot(trajectory[:, 0], trajectory[:, 1], 'go')
      
      plt.plot(mainPt[0], mainPt[1], 'ro')
  
      ax = plt.gca()
      plt.plot(keypoints[:, 0], keypoints[:, 1], 'r*')
      ax.axis('equal')
      plt.show()
      pass
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
    return res
  
  def _seedsStream(self, N):
    while 0 < len(self._mainHashes):
      seeds = random.choices(
        self._mainHashes, k=2*N,
        cum_weights=self._samplingDistribution
      )
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

  def _applyEyeDropout(self, X, dropout):
    if 0.0 < dropout:
      imgA = X[1]
      imgB = X[2]
      N = len(imgA)
      
      mask = np.random.random((N,)) < dropout
      maskA = 0.5 < np.random.random((N,))
      maskB = np.logical_not(maskA)
      
      imgA[np.where(np.logical_and(mask, maskA))] = 0.0
      imgB[np.where(np.logical_and(mask, maskB))] = 0.0
      
      return (X[0], imgA, imgB)
    return X
  
  def _applyEyeAdditiveNoise(self, X, noise):
    if noise <= 0.0: return X

    points, imgA, imgB = X
    imgA += np.random.normal(scale=noise, size=imgA.shape)
    np.clip(imgA, 0, 1, out=imgA)
    
    imgB += np.random.normal(scale=noise, size=imgB.shape)
    np.clip(imgB, 0, 1, out=imgB)
    return(points, imgA, imgB)
  
  def _applyPointsNoise(self, X, noise):
    if noise <= 0.0: return X

    points, imgA, imgB = X
    index = np.where(np.all(-1 < points, axis=-1))
    points[index] += np.random.normal(scale=noise, size=(len(index[0]), 2))
    return(points, imgA, imgB)
  
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
  
  def _indexes2XY(self, indexes, kwargs):
    timesteps = kwargs.get('timesteps', None)
    return_tensors = kwargs.get('return_tensors', False)
    samples = [self._storage[i] for i in indexes]
    
    if return_tensors:
      X = _toTensor(
        (
          np.array([x['points'] for x in samples], np.float32),
          np.array([x['left eye'] for x in samples]),
          np.array([x['right eye'] for x in samples]),
        ),
        (
          kwargs.get('pointsDropout', 0.0),
          kwargs.get('pointsNoise', 0.0),
        
          kwargs.get('eyesAdditiveNoise', 0.0),
          kwargs.get('eyesDropout', 0.0),
        
          timesteps
        )
      )
    else:
      X = Utils.samples2inputs(samples, dropout=kwargs.get('pointsDropout', 0.0))
      X = self._applyPointsNoise(X, noise=kwargs.get('pointsNoise', 0.0))
      
      X = self._applyEyeAdditiveNoise(X, noise=kwargs.get('eyesAdditiveNoise', 0.0))
      X = self._applyEyeDropout(X, kwargs.get('eyesDropout', 0.0))
      X = self._reshapeSteps(X, timesteps)
    ###############
    forecast = kwargs.get('forecast', {})
    maxT = kwargs.get('maxT', 1.0)
    forecast['maxT'] = forecast.get('maxT', maxT)
    Y = ( np.array([
      self._targetFor(i, **forecast) 
      for i in indexes
    ], np.float32), )
    Y = self._reshapeSteps(Y, timesteps)
    return(X, Y)
  
  def _hashesFor(self, goal):
    goal = np.array(goal) # np.clip(goal, 0.0, 1.0)
    goal = goal - 0.5 # centered
    x = np.trunc(goal * self._goalHashScale)
#     ax = np.trunc(np.abs(goal) * self._goalHashScale)
#     
#     xx = np.abs(x)
#     isCorner = xx[0] == xx[1]
#     cornerID = 'No'
#     if isCorner and np.alltrue(0 < xx):
#       cornerID = str(x / xx)
#     lvl = np.greater_equal([14, 9, 6], ax.max())
#     
    return [
#       isCorner,
#       cornerID,
#       *lvl,
#       ax.max(),
#       str(0 < x), # left/right, top/bottom
      str(x),
    ]
    
  def _bucketFor(self, goal):
    hashes = self._hashesFor(goal)
    #######
    hashA = hashes[0]
    if not(hashA in self._mainHashes):
      self._mainHashes.append(hashA)
    
    s = self._samplesByGoals
    for h in hashes[:-1]:
      if not(h in s): s[h] = {}
      s = s[h]
      continue
    h = hashes[-1]
    if not(h in s): s[h] = []
    return s[h]
  
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
    F(self._samplesByGoals)
    return LLBuckets
  
  def updateDistribution(self, pos, loss):
    if not(self._debugDistribution or self._adjustDistribution): return
    
    N = len(self._mainHashes)
    statsPerBuckets = {}
    for p, l in zip(pos, loss):
      h = self._hashesFor(p)[0]
      bucket = statsPerBuckets.get(h, None)
      if bucket is None:
        statsPerBuckets[h] = bucket = {'loss': [], 'pos': []}
      bucket['loss'].append(l)
      bucket['pos'].append(p)
      continue
    
    statsPerBuckets = {
      h: {
        'loss': np.mean(v['loss']),
        'pos': np.mean(v['pos'], axis=0)
      }
      for h, v in statsPerBuckets.items()
    }
    self._debugLoss(statsPerBuckets)
    
    if self._adjustDistribution:
      statsPerBuckets = {
        h: v
        for h, v in statsPerBuckets.items()
        if h in self._mainHashes
      }
      unknownLoss = np.mean([x['loss'] for x in statsPerBuckets.values()])
      distribution = np.full((N, ), unknownLoss)
      for h, v in statsPerBuckets.items():
        distribution[self._mainHashes.index(h)] = v['loss']
        continue
      
      ordered = np.argsort(distribution)
      if isinstance(self._adjustDistribution, dict):
        mu = self._adjustDistribution['mu']
        if 'auto' == mu:
          muPos = np.searchsorted(distribution[ordered], distribution.mean())
          mu = float(muPos) / N
          pass
        
        fakeDistr = gaussian(
          np.linspace(0., 1., N), 
          mu,
          self._adjustDistribution['sigma']
        )
        noise = self._adjustDistribution.get('noise', 0.0)
        if 0.0 < noise:
          fakeDistr += np.random.uniform(size=N, high=noise)
        distribution[ordered] = fakeDistr
      else:
        h, l = distribution.max(), distribution.min()
        distribution = (distribution - l) / (h - l)
        pass

      if self._debugDistribution:
        plt.figure(figsize=(8, 8))
        plt.bar(np.arange(N), distribution[ordered], width=1)
        plt.savefig('sampling-distr.png')
        plt.clf()
        plt.close()
        pass
      
      self._samplingDistribution = np.cumsum(distribution)
      pass
    return
  
  def _debugLoss(self, statsPerBuckets):
    if not self._debugDistribution: return
    
    cmb = list(statsPerBuckets.values())
    d = np.array([x['loss'] for x in cmb]).reshape(-1)
    
    plt.figure(figsize=(8, 8))
    plt.bar(np.arange(len(d)), np.sort(d), width=1)
    plt.savefig('loss-distr.png')
    plt.clf()
    plt.close()
  
    plt.figure(figsize=(8, 8))
    c = np.array([x['loss'] for x in cmb]).reshape(-1)
    pos = np.array([x['pos'] for x in cmb])
    sc = plt.scatter(
      pos[:, 0], pos[:, 1],
      c=c,
      cmap='tab10',
      vmin=0.0, vmax=0.01,
    )
    plt.colorbar(sc)
    plt.savefig('loss-heatmap.png')
    plt.clf()
    plt.close()
    return
##############
if __name__ == '__main__':
  import os
  from Core.CSamplesStorage import CSamplesStorage
  folder = os.path.dirname(os.path.dirname(__file__))
  ds = CDataSampler( CSamplesStorage() )
  ds.addBlock(Utils.datasetFromFolder(os.path.join(folder, 'Dataset')))

  xy = ds.sample(64, timesteps=5, stepsSampling={'max frames': 22, 'include last': True})
  for v in xy:
    for x in v:
      print(x.shape)
    print('........')

  xy = ds.sampleById(8, timesteps=5)
  x = xy[0][0][0, 0]
  plt.plot(x[:, 0], x[:, 1], 'o', markersize=1)
  plt.axis('equal')
  plt.show()
#   print(x)
  pass