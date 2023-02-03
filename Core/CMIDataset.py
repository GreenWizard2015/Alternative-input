import threading
import numpy as np
import random
from collections import defaultdict

class CMIDataset:
  def __init__(self):
    self._lock = threading.RLock()
    self._samples = []
    return

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    data = defaultdict(list)
    for sample in self._samples:
      for k, v in sample.items():
        data[k].append(v)
      continue
    data = {k: np.array(v) for k, v in data.items()}

    np.savez('d:/dump.npz', **data)
    return

  def store(self, latent, pos, goal, prev):
    sample = {'latent': latent, 'pos': pos, 'goal': goal, 'prev': prev}
    with self._lock:
      self._samples.append(sample)
    return

  def _sample(self, N):
    indx = random.sample(range(len(self._samples)), N)
    res = defaultdict(list)
    for i in indx:
      sample = self._samples[i]
      nextI = min((len(self._samples) - 1, i + 5))
      nextPos = self._samples[nextI]['prev']
      sample = {**sample, 'pos': nextPos}
      for k, v in sample.items():
        res[k].append(v)
      continue

    return {k: np.array(v, np.float32) for k, v in res.items()}

  def sample(self, N=64):
    with self._lock:
      if len(self._samples) < N: return None
      
      res = tuple(self._sample(N) for _ in range(2))
    return res
    
  def __len__(self):
    with self._lock:
      return len(self._samples)
    return

  @property
  def totalSamples(self):
    return len(self)