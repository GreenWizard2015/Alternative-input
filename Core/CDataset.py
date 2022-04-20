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
      samples = self._samples[str(np.ceil(np.array(goal) * 20))]

      if self._maxSamples < len(samples):
        if 0.75 < random.random():
          idx = random.randint(0, len(samples) - 1)
          samples[idx] = sample
      else:
        samples.append(sample)
      
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
    data = {k: np.array(v, np.float32) for k, v in data.items()}

    np.savez(self._storeTo(), **data)
    self._storedSamples.clear() 
    return