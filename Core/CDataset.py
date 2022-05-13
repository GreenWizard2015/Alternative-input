import threading
import Utils
import numpy as np
import random
from collections import defaultdict
import os
import time
 
class CDataset:
  def __init__(self, folder, maxSamples=20):
    self._lock = threading.RLock()
    self._samples = defaultdict(list)
    self._maxSamples = maxSamples
    
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

  def store(self, data, goal, T):
    data = Utils.tracked2sample(data)
    sample = (data, {'goal': goal, 'time': T})
    with self._lock:
      h = self._goal2hash(goal)
      samples = self._samples[h]

      if self._maxSamples < len(samples):
        if 0.75 < random.random():
          idx = random.randint(0, len(samples) - 1)
          samples[idx] = sample
      else:
        samples.append(sample)
      
      self._putToDistribution(goal)
      self._storedSamples.append(sample)
      if self._samplesPerChunk <= len(self._storedSamples):
        self._saveSamples()
    return

  def sample(self, N=16):
    samples = []
    with self._lock:
      buckets = list(self._samples.values())
      if buckets:
        for bucket in random.choices(buckets, k=N):
          samples.append(random.choice(bucket))

    if len(samples) < N: return None
    return(
      Utils.samples2inputs([x for x, _ in samples], dropout=0.3), 
      ( np.array([x['goal'] for _, x in samples], np.float32), )
    )
    
  def __len__(self):
    with self._lock:
      return sum([len(x) for x in self._samples.values()], 0)
    
  def _saveSamples(self):
    data = defaultdict(list)
    for sampleData in self._storedSamples:
      for sample in sampleData:
        for k, v in sample.items():
          data[k].append(v)
      continue
    data = {k: np.array(v) for k, v in data.items()}

    np.savez(self._storeTo(), **data)
    self._storedSamples.clear() 
    return
  
  def _goal2hash(self, goal):
    return str(np.floor(np.array(goal) * 30))
  
  def _putToDistribution(self, goal):
    M = 30
    h = str(np.clip(np.floor(goal * M), 0, M - 1))
    self._distrMap[h]['N'] += 1
    return
  
  def distribution(self):
    M = 30
    with self._lock:
      dmap = np.zeros((M, M, 1), np.float32)
      for x in self._distrMap.values():
        goal = np.clip(np.floor(x['center'] * M), 0, M)
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