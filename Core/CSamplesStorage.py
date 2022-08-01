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
  def __init__(self):
    self._head = None
    self._latestT = -np.inf
    return

  def __len__(self):
    if self._head:
      return self._head.count()
    return 0

  @lru_cache(10000)
  def __getitem__(self, idx):
    return self._head.get(idx)
  
  def add(self, sample):
    if not self._head:
      self._head = CSamplesStorageChunk()
    assert self._latestT < sample['time']
    
    self._latestT = sample['time']
    return self._head.add(sample)
  
  def addBlock(self, samples):
    T = samples['time']
    assert self._latestT < T[0]
    assert all(a < b for a, b in zip(T[:-1], T[1:]))
    
    startIndex = len(self)
    block = CSamplesStorageChunk(samples=samples)
    if not self._head:
      self._head = block
    else:
      self._head.append(block)
      
    self._latestT = T[-1]
    return np.arange(startIndex, startIndex + len(T))
  
if __name__ == '__main__':
  import os, Utils
  folder = os.path.dirname(os.path.dirname(__file__))
  ds = CSamplesStorage()
  ds.addBlock(Utils.datasetFromFolder(os.path.join(folder, 'Dataset')))
  print(len(ds))
  s = ds[0]
  print(list(s.keys()))
  pass