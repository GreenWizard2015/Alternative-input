import Core.Utils as Utils
import os
from Core.CSamplesStorage import CSamplesStorage
import numpy as np
from enum import Enum

class ESampling(Enum):
  AS_IS = 'as_is'
  UNIFORM = 'uniform'
  
class CDatasetLoader:
  def __init__(self, folder, samplerArgs, sampling, stats, sampler_class, test_folders):
    self._datasets = []
    for datasetFolder, ID in Utils.dataset_from_stats(stats, folder):
      (place_id_index, user_id_index, screen_id_index) = ID
      for test_folder in test_folders:
        dataset = os.path.join(datasetFolder, test_folder)
        if not os.path.exists(dataset):
          continue
        print('Loading %s' % (dataset, ))
        print(f'ID: {ID}. Index: {1 + len(self._datasets)}')
        ds = sampler_class(
          CSamplesStorage(
            placeId=place_id_index,
            userId=user_id_index,
            screenId=screen_id_index,
          ),
          **samplerArgs
        )
        ds.addBlock(Utils.datasetFrom(dataset))
        self._datasets.append(ds)

    if 0 == len(self._datasets):
      raise Exception('No training dataset found in "%s"' % (folder, ))
    
    validSamples = {
      i: len(ds.validSamples())
      for i, ds in enumerate(self._datasets)
    }
    # ignore datasets with no valid samples
    validSamples = {k: v for k, v in validSamples.items() if 0 < v}

    print('Loaded %d datasets with %d valid samples' % (len(self._datasets), sum(validSamples.values())))

    dtype = np.uint8 if len(self._datasets) < 256 else np.uint32
    # create an array of dataset indices to sample from
    sampling = ESampling(sampling)
    if ESampling.AS_IS == sampling: # just concatenate the indices
      self._indices = np.concatenate([
        np.full((v, ), k, dtype=dtype) # id of the dataset
        for k, v in validSamples.items()
      ])
    if ESampling.UNIFORM == sampling:
      maxSize = max(validSamples.values())
      chunks = []
      for k, size in validSamples.items():
        # all datasets should have the same number of samples represented in the indices
        # so that the sampling is uniform
        chunk = np.full((maxSize, ), k, dtype=dtype)
        chunks.append(chunk)
        continue
      self._indices = np.concatenate(chunks)
      
    self._currentId = 0

    self._batchSize = samplerArgs.get('batch_size', 16)
    return
  
  def on_epoch_start(self):
    # shuffle the indices at the beginning of each epoch
    self._currentId = 0
    np.random.shuffle(self._indices)
    # reset all datasets
    for ds in self._datasets:
      ds.reset()
      continue
    return
  
  def on_epoch_end(self):
    return

  def __len__(self):
    return 1 + (len(self._indices) // self._batchSize)
  
  def _getBatchStats(self, batchSize):
    batch = self._indices[self._currentId:self._currentId + batchSize]
    self._currentId = (self._currentId + batchSize) % len(self._indices)
    while len(batch) < batchSize:
      batch2 = self._indices[:batchSize - len(batch)]
      self._currentId = (self._currentId + len(batch2)) % len(self._indices)
      batch = np.concatenate([batch, batch2], axis=0) # concatenate the two batches
      continue

    assert len(batch) == batchSize, 'Invalid batch size: %d != %d' % (len(batch), batchSize)
    datasetIds, counts = np.unique(batch, return_counts=True)
    return datasetIds, counts

  def sample(self, **kwargs):
    batchSize = kwargs.get('batch_size', self._batchSize)
    samples = []
    totalSamples = 0
    # find the datasets ids and the number of samples to take from each dataset
    while totalSamples < batchSize:
      datasetIds, counts = self._getBatchStats(batchSize - totalSamples)
      for datasetId, N in zip(datasetIds, counts):
        dataset = self._datasets[datasetId]
        sampled, N = dataset.sample(N=N, **kwargs)
        if 0 < N:
          samples.append(sampled)
        totalSamples += N
        continue
    
    first_dataset = self._datasets[0]
    return first_dataset.merge(samples, batchSize)
