import threading
import Utils
import numpy as np
import random
from collections import defaultdict

# TODO: Save samples to disk 
class CDataset:
  def __init__(self, maxSamples=20):
    self._lock = threading.RLock()
    self._samples = defaultdict(list)
    self._maxSamples = maxSamples
    return

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    return

  def store(self, data, goal):
    data = Utils.tracked2sample(data)
    with self._lock:
      samples = self._samples[str(np.ceil(np.array(goal) * 20))]

      if self._maxSamples < len(samples):
        if 0.75 < random.random():
          idx = random.randint(0, len(samples) - 1)
          samples[idx] = (data, goal)
      else:
        samples.append((data, goal))
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
      ( np.array([x for _, x in samples], np.float32), )
    )
    
  def __len__(self):
    with self._lock:
      return sum([len(x) for x in self._samples.values()], 0)