import threading
import Core.Utils as Utils
import numpy as np
from collections import defaultdict
import os
import time
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
 
class CDataset:
  def __init__(self, folder, timesteps):
    self._lock = threading.RLock()
    self._samples = CDataSampler(
      CSamplesStorage(),
      defaults=dict(
        timesteps=timesteps,
        stepsSampling={'max frames': 2, 'include last': True},
        # augmentations 
        pointsDropout=0.25, pointsNoise=0.002,
        eyesDropout=0.05, eyesAdditiveNoise=0.02,
      )
    )
    self._timesteps = timesteps
    
    self._totalSamples = 0
    self._storedSamples = []
    self._samplesPerChunk = 1000
    os.makedirs(folder, exist_ok=True)
    self._storeTo = lambda: os.path.join(folder, '%d.npz' % int(time.time() * 1000))
    ################
    self._distrMap = self._makeDistrMap()
    for data in Utils.dataFromFolder(folder):
      for goal in data['goal']:
        self._putToDistribution(goal)
        continue
      continue
    return

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    if 0 < len(self._storedSamples):
      self._saveSamples()
    return

  def store(self, data, goal):
    data = Utils.tracked2sample(data)
    sample = {**data, 'goal': goal}
    with self._lock:
      self._samples.add(sample)
      
      self._putToDistribution(goal)
      self._storedSamples.append(sample)
      if self._samplesPerChunk <= len(self._storedSamples):
        self._saveSamples()
    return

  def sample(self, N=16):
    with self._lock:
      res = self._samples.sample(N)

    return res
    
  def __len__(self):
    with self._lock:
      return len(self._samples)
    
  def _saveSamples(self):
    data = defaultdict(list)
    for sample in self._storedSamples:
      for k, v in sample.items():
        data[k].append(v)
      continue
    data = {k: np.array(v) for k, v in data.items()}

    np.savez(self._storeTo(), **data)
    self._storedSamples.clear() 
    return
  
  def _goal2hash(self, goal):
    return str(np.trunc(np.array(goal) * 30))
  
  def _putToDistribution(self, goal):
    M = 30
    h = str(np.clip(np.trunc(goal * M), 0, M - 1))
    self._distrMap[h]['N'] += 1
    self._totalSamples += 1
    return
  
  def distribution(self):
    M = 30
    with self._lock:
      dmap = np.zeros((M, M, 1), np.float32)
      for x in self._distrMap.values():
        goal = np.clip(np.trunc(x['center'] * M), 0, M)
        cy, cx = goal.astype(np.int)
        dmap[cx, cy] = x['N']
        continue
      return [(x['center'], x['N']) for x in self._distrMap.values()], dmap[:-1, :-1, :]
    return
  
  def _makeDistrMap(self):
    M = 30
    dm = {}
    for i in range(M):
      for j in range(M):
        pt = np.array([i, j], np.float)
        c = (0.5 + pt) / M
        dm[str(pt)] = {'N': 0, 'center': c}
        continue
      continue
    return dm
  
  @property
  def totalSamples(self):
    return self._totalSamples