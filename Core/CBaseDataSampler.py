import numpy as np
import random
from math import ceil
from functools import lru_cache

class CBaseDataSampler:
    def __init__(self, storage, batch_size, minFrames, defaults={}, maxT=1.0, cumulative_time=True):
        '''
        Base class for data sampling.

        Parameters:
        - storage: The storage object containing the samples.
        - batch_size: The number of samples per batch.
        - minFrames: The minimum number of frames required in a trajectory.
        - defaults: Default parameters for sampling.
        - maxT: Maximum time window for sampling frames.
        - cumulative_time: If True, time is cumulative; otherwise, it's time deltas.
        '''
        self._storage = storage
        self._defaults = defaults
        self._batchSize = batch_size
        self._maxT = maxT
        self._minFrames = minFrames
        self._samples = []
        self._currentSample = None
        self._cumulative_time = cumulative_time
        return

    def reset(self):
        random.shuffle(self._samples)
        self._currentSample = 0
        return

    def __len__(self):
        return ceil(len(self._samples) / self._batchSize)

    def _storeSample(self, idx):
        # Store sample if it has enough frames
        minInd = self._getTrajectoryBefore(idx)
        if self._minFrames <= (idx - minInd):
            self._samples.append(idx)
        return

    def add(self, sample):
        idx = self._storage.add(sample)
        self._storeSample(idx)
        return idx

    def addBlock(self, samples):
        indexes = self._storage.addBlock(samples)
        for idx in indexes:
            self._storeSample(idx)
            continue
        return

    def _getTrajectoryBefore(self, mainInd):
        mainT = self._storage[mainInd]['time']
        minT = mainT - self._maxT

        minInd = mainInd
        for ind in range(mainInd - 1, -1, -1):
            if self._storage[ind]['time'] < minT: break
            minInd = ind
            continue
        return minInd

    @lru_cache(None)
    def _trajectoryRange(self, mainInd):
        '''
        Returns indexes of samples that are within maxT from mainInd.
        Returns (minInd, maxInd) where minInd <= mainInd <= maxInd
        '''
        mainT = self._storage[mainInd]['time']
        maxT = mainT + self._maxT
        maxInd = mainInd
        for ind in range(mainInd, len(self._storage)):
            if maxT < self._storage[ind]['time']: break
            maxInd = ind
            continue

        minInd = self._getTrajectoryBefore(mainInd)
        return minInd, maxInd

    def _trajectory(self, mainInd):
        minInd, maxInd = self._trajectoryRange(mainInd)
        return list(range(minInd, mainInd + 1)), list(range(mainInd + 1, maxInd + 1))

    def _prepareT(self, res):
        T = np.array([self._storage[ind]['time'] for ind in res])
        T -= T[0]
        diff = np.diff(T, 1)
        idx = np.nonzero(diff)[0]
        if len(idx) < 1: return None  # All frames have the same time
        if len(diff) == len(idx):
            T = diff
        else:
            return None # Time is not consistent
        T = np.insert(T, 0, 0.0)
        assert len(res) == len(T)
        # Convert to cumulative time if required
        if self._cumulative_time:
            T = np.cumsum(T)
        return T

    def _reshapeSteps(self, values, steps):
        if steps is None:
            return values

        res = []
        for x in values:
            B, *s = x.shape
            newShape = (B // steps, steps, *s)
            res.append(x.reshape(newShape))
            continue
        return tuple(res)

    @property
    def totalSamples(self):
        return len(self._storage)

    def validSamples(self):
        return list(sorted(self._samples))
    
    def _framesFor(self, mainInd, samples, steps, stepsSampling):
        if 'uniform' == stepsSampling:
            samples = random.sample(samples, steps - 1)
        elif 'last' == stepsSampling:
            samples = samples[-(steps - 1):]
        elif isinstance(stepsSampling, dict):
            candidates = list(samples)
            maxFrames = stepsSampling['max frames']
            candidates = candidates[::-1]
            samples = []
            left = steps - 1
            for _ in range(left):
                avl = min((maxFrames, 1 + len(candidates) - left))
                ind = random.randint(0, avl - 1)
                samples.append(candidates[ind])
                candidates = candidates[ind+1:]
                left -= 1
                continue
            pass
        else:
            raise ValueError('Unknown sampling method: ' + str(stepsSampling))

        res = list(sorted(samples + [mainInd]))
        assert len(res) == steps
        return res