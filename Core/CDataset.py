import threading
import Utils
import numpy as np
import random

# TODO: Add buckets for samples based on goals i. e. split screen into 64 regions and maintain at lease 32 sample in each
# TODO: Save samples to disk 
class CDataset:
  def __init__(self, maxSamples=2000):
    self._lock = threading.Lock()
    self._samples = []
    self._maxSamples = maxSamples
    return

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    return

  def store(self, data, goal):
    data = Utils.tracked2sample(data)
    with self._lock:
      if self._maxSamples < len(self._samples):
        idx = random.randint(0, self._maxSamples)
        self._samples[idx] = (data, goal)
      else:
        self._samples.append((data, goal))
    return

  def sample(self, N=16):
    with self._lock:
      if len(self._samples) < N: return None
      
      idx = np.random.choice(np.arange(len(self._samples)), N)
      samples = [self._samples[i] for i in idx]

    return(
      Utils.samples2inputs([x for x, _ in samples]), 
      ( np.array([x for _, x in samples], np.float32), )
    )
    
  def __len__(self):
    with self._lock:
      return len(self._samples)