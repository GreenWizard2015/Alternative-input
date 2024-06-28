import Core.Utils as Utils
import numpy as np
from collections import defaultdict
import os
import time

class CDataset:
  def __init__(self, folder, timesteps):
    self._timesteps = timesteps
    
    self._totalSamples = Utils.countSamplesIn(folder)
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

  def store(self, data, goal):
    data = Utils.tracked2sample(data)
    sample = {**data, 'goal': goal}
    self._storedSamples.append(sample)
    self._totalSamples += 1

    if self._samplesPerChunk <= len(self._storedSamples):
      self._saveSamples()
    return
    
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

  @property
  def totalSamples(self):
    return self._totalSamples