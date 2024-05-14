import numpy as np
from collections import defaultdict
from functools import lru_cache

class CSamplesStorageChunk:
  def __init__(self, samples=None):
    self._next = None
    self._samples = defaultdict(list) if samples is None else samples
    return
    
  def _canAdd(self):
    if self._next: return False
    T = self._samples['time']
    if not isinstance(T, list): return False
    if 5000 < len(T): return False
    return True
   
  def _pack(self):
    if isinstance(self._samples['time'], list):
      self._samples = {k: np.array(v) for k, v in self._samples.items()}
    return
  
  def _ownCount(self):
    if self._samples:
      return len(self._samples['time'])
    return 0
  
  def count(self):
    cnt = self._next.count() if self._next else 0
    return self._ownCount() + cnt
    
  def add(self, sample, idx=0):
    if self._canAdd():
      idx += self._ownCount()
      for k, v in sample.items():
        self._samples[k].append(v)
      return idx

    if not self._next:
      self._pack()
      self._next = CSamplesStorageChunk()
      
    return self._next.add(sample, idx + self._ownCount())
  
  def get(self, idx):
    N = self._ownCount()
    if idx < N:
      return {k: v[idx] for k, v in self._samples.items()}
    return self._next.get(idx - N)
  
  def append(self, block):
    if self._next and (0 < self._next._ownCount()):
      return self._next.append(block)
    
    self._next = block
    return

class CSamplesStorage:
  def __init__(self, userId, placeId, screenId):
    self._head = None
    self._latestT = -np.inf
    # they are the same for all samples, so we store them here to reduce the memory usage
    self._userId = userId
    self._placeId = placeId
    self._screenId = screenId
    return

  def __len__(self):
    if self._head:
      return self._head.count()
    return 0

  @lru_cache(10000)
  def __getitem__(self, idx):
    res = self._head.get(idx)
    # add userId, placeId, screenId
    res['userId'] = self._userId
    res['placeId'] = self._placeId
    res['screenId'] = self._screenId
    return res
  
  def add(self, sample):
    if not self._head:
      self._head = CSamplesStorageChunk()
    T = sample['time']
    
    self._latestT = T
    return self._head.add(sample)
  
  def addBlock(self, samples):
    assert all(isinstance(v, np.ndarray) for v in samples.values())
    samples = {k: v for k, v in samples.items()} # copy the dictionary
    N = len(samples['time'])

    startIndex = len(self)
    block = CSamplesStorageChunk(samples=samples)
    if not self._head:
      self._head = block
    else:
      self._head.append(block)
    return np.arange(startIndex, startIndex + N)

if __name__ == '__main__':
  import os, Core.Utils as Utils
  folder = os.path.dirname(os.path.dirname(__file__))
  ds = CSamplesStorage()
  # ds.addBlock(Utils.datasetFrom(os.path.join(folder, 'Data', 'Dataset')))
  ds.addBlock(Utils.datasetFrom(os.path.join(folder, 'Data', 'test.npz')))
  print(len(ds))
  s = ds[0]
  print(list(s.keys()))

  pass