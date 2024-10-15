from .CBaseDataSampler import CBaseDataSampler
import Core.CDataSampler_utils as DSUtils

import numpy as np
from functools import lru_cache

'''
This sampler are sample N frames from the dataset, where N is the number of timesteps.
It returns the tuple (X, Y), where X is the input data and Y is the target data.
To X could be applied some augmentations.
X contains the following data:
  - The points of the face.
  - The left eye.
  - The right eye.
  - The time (cumulative or delta).
  - The user ID, place ID, and screen ID.
Y contains the target data.
  - The target point.
'''
class CDataSampler(CBaseDataSampler):
    def __init__(self, storage, batch_size, minFrames, defaults={}, maxT=1.0, cumulative_time=True):
        super().__init__(storage, batch_size, minFrames, defaults, maxT, cumulative_time)

    def _stepsFor(self, mainInd, steps, stepsSampling='uniform', **_):
        if (steps is None) or (1 == steps): return [(mainInd, 0.0)]
        if mainInd < steps: return False

        samples, _ = self._trajectory(mainInd)
        if len(samples) < (steps - 1): return False
        # Try to sample valid frames
        for _ in range(10):
            res = self._framesFor(mainInd, samples, steps, stepsSampling)
            T = self._prepareT(res)
            if T is not None:
                assert len(res) == len(T)
                return [tuple(x) for x in zip(res, T)]
            continue
        return False

    def sample(self, **kwargs):
        kwargs = {**self._defaults, **kwargs}
        timesteps = kwargs.get('timesteps', None)
        N = kwargs.get('N', self._batchSize)
        indexes = []
        for _ in range(N):
            added = False
            while not added:
                idx = self._samples[self._currentSample]
                self._currentSample = (self._currentSample + 1) % len(self._samples)

                sampledSteps = self._stepsFor(idx, steps=timesteps, **kwargs)
                if sampledSteps:
                    # TODO: remove from samples?
                    indexes.extend(sampledSteps)
                    added = True
                continue

        return self._indexes2XY(indexes, kwargs)

    def sampleById(self, idx, **kwargs):
        kwargs = {**self._defaults, **kwargs}
        timesteps = kwargs.get('timesteps', None)
        sampledSteps = self._stepsFor(idx, steps=timesteps, **kwargs)
        if not sampledSteps: return None
        return self._indexes2XY([*sampledSteps], kwargs)

    def checkById(self, idx, **kwargs):
        kwargs = {**self._defaults, **kwargs}
        timesteps = kwargs.get('timesteps', None)
        sampledSteps = self._stepsFor(idx, steps=timesteps, **kwargs)
        if not sampledSteps: return False
        return True

    def sampleByIds(self, ids, **kwargs):
        kwargs = {**self._defaults, **kwargs}
        timesteps = kwargs.get('timesteps', None)
        sampledSteps = []
        rejected = []
        accepted = []
        for idx in ids:
            sample = self._stepsFor(idx, steps=timesteps, **kwargs)
            if sample:
                accepted.append(idx)
                sampledSteps.extend(sample)
            else:
                rejected.append(idx)
                pass
            continue

        res = None
        if 0 < len(sampledSteps):
            res = self._indexes2XY(sampledSteps, kwargs)
        return res, rejected, accepted

    @lru_cache(None)
    def _targetFor(self, ind):
        mainPt = self._storage[ind]['goal']
        keypoints = np.array(mainPt, np.float32)
        return keypoints

    def _indexes2XY(self, indexesAndTime, kwargs):
        timesteps = kwargs.get('timesteps', None)
        samples = [self._storage[i] for i, _ in indexesAndTime]

        Y = ( np.array([ self._targetFor(i)  for i, _ in indexesAndTime], np.float32), )
        Y = self._reshapeSteps(Y, timesteps)
        ##############
        userIds = np.unique([x['userId'] for x in samples])
        assert 1 == len(userIds), 'Only one user is supported. Found: ' + str(userIds)
        placeIds = np.unique([x['placeId'] for x in samples])
        assert 1 == len(placeIds), 'Only one place is supported. Found: ' + str(placeIds)
        screenIds = np.unique([x['screenId'] for x in samples])
        assert 1 == len(screenIds), 'Only one screen is supported. Found: ' + str(screenIds)

        X = DSUtils.toTensor(
            (
                np.array([x['points'] for x in samples], np.float32),
                np.array([x['left eye'] for x in samples]),
                np.array([x['right eye'] for x in samples]),
                np.array([T for _, T in indexesAndTime], np.float32).reshape((-1, 1)),
            ),
            (
                kwargs.get('pointsNoise', 0.0),
                kwargs.get('pointsDropout', 0.0),

                kwargs.get('eyesAdditiveNoise', 0.0),
                kwargs.get('eyesDropout', 0.0),
                kwargs.get('brightnessFactor', 0.0),
                kwargs.get('lightBlobFactor', 0.0),

                timesteps
            ),
            userIds[0], placeIds[0], screenIds[0]
        )
        ###############
        (Y, ) = Y
        return (X, (Y.astype(np.float32), ))

    def merge(self, samples, expected_batch_size):
        Y = np.concatenate([Y for _, (Y, ) in samples], axis=0)
        assert len(Y) == expected_batch_size, 'Invalid batch size: %d != %d' % (len(Y), expected_batch_size)
        # X contains the clean and augmented data
        # each dictionary contains the subkeys: points, left eye, right eye, time, userId, placeId, screenId
        X = {}
        for key in ['clean', 'augmented']:
            for subkey in ['points', 'left eye', 'right eye', 'time', 'userId', 'placeId', 'screenId']:
                data = [x[key][subkey] for x, _ in samples]
                X[key][subkey] = np.concatenate(data, axis=0)
                assert X[key][subkey].shape[0] == expected_batch_size, 'Invalid batch size: %d != %d' % (X[key][subkey].shape[0], expected_batch_size)
                continue
            continue
        return (X, (Y, ))