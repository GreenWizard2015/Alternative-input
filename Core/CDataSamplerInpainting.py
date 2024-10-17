from .CBaseDataSampler import CBaseDataSampler
import Core.CDataSampler_utils as DSUtils
from Core.Utils import FACE_MESH_POINTS

import numpy as np
import tensorflow as tf

'''
This sampler are sample N frames from the dataset, where N is the number of timesteps.
Within the range of the N sampled frames, its samples K frames to be inpainted/reconstructed.
It returns the tuple (X, Y), where X is the input data and Y is the target data.
To X could be applied some augmentations.
X contains the following data:
  - The points of the face.
  - The left eye.
  - The right eye.
  - The time (cumulative or delta).
  - The target point.
  - The user ID, place ID, and screen ID.
Y contains the target data, K frames to be inpainted/reconstructed.
  - The points of the face.
  - The left eye.
  - The right eye.
  - The normalized time.
  - The target point.
'''
class CDataSamplerInpainting(CBaseDataSampler):
    def __init__(self, storage, batch_size, minFrames, keys, defaults={}, maxT=1.0, cumulative_time=True):
        super().__init__(storage, batch_size, minFrames, defaults, maxT, cumulative_time)
        self._keys = keys

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
        N = kwargs.get('N', self._batchSize) // len(self._keys)
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

    def _indexes2XY(self, indexesAndTime, kwargs):
        timesteps = kwargs.get('timesteps', None)
        assert timesteps is not None, 'The number of timesteps must be defined.'
        B = len(indexesAndTime) // timesteps
        samples = [self._storage[i] for i, _ in indexesAndTime]
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
        for k in X.keys():
            # add the target point to the X
            targets = np.array([x['goal'] for x in samples], np.float32).reshape((B, timesteps, 2))
            X[k]['target'] = tf.constant(targets, dtype=tf.float32)

        if 1 == len(self._keys):
            X = X[self._keys[0]]
        else:
            res = {}
            k = self._keys[0]
            subkeys = list(X[k].keys())
            for k in subkeys:
                values = [X[key][k] for key in self._keys]
                res[k] = tf.concat(values, axis=0)
                continue
            X = res
            indexesAndTime = indexesAndTime * len(self._keys)
            B = len(self._keys) * B

        ###############
        # generate the target data
        targets = kwargs.get('targets', {'keypoints': timesteps, 'total': timesteps})
        K = targets.get('keypoints', timesteps)
        assert K <= timesteps, 'The number of keypoints to be inpainted/reconstructed must be less or equal to the number of timesteps.'
        T = targets.get('total', timesteps)
        assert K <= T, 'The total number of frames to be inpainted/reconstructed must be less or equal to the total number of timesteps.'

        samples_indexes = np.array([ i  for i, _ in indexesAndTime], np.int32)
        samples_indexes = samples_indexes.reshape((-1, timesteps))
        assert samples_indexes.shape[0] == B, 'Invalid number of samples: %d != %d' % (samples_indexes.shape[0], B)
        targetsIdx = np.zeros((B, T), np.int32)
        for i in range(B):
            # sample K frames from the X
            sampled = np.random.choice(samples_indexes[i], K, replace=False)
            # repeat the sampled frames to fill the K frames
            targetsIdx[i, :] = np.repeat(sampled, 1 + (T // K))[:T]
            # sample the remaining frames
            if K < T: # if need to sample more frames
                startFrameIdx = samples_indexes[i, 0]
                endFrameIdx = samples_indexes[i, -1]
                allFrames = np.arange(startFrameIdx, endFrameIdx + 1)
                # exclude the frames that are already sampled
                allFrames = np.array([x for x in allFrames if x not in sampled], np.int32)
                # sample the remaining frames
                targetsIdx[i, K:] = np.random.choice(allFrames, T - K, replace=True)
            continue
        # targetsIdx contains the indexes of the frames to be inpainted/reconstructed
        # we need to collect the data for the Y
        # required data: points, left eye, right eye, time, target point
        Y = {
            'points': np.zeros((B, T, FACE_MESH_POINTS, 2), np.float32),
            'left eye': np.zeros((B, T, 32, 32), np.float32),
            'right eye': np.zeros((B, T, 32, 32), np.float32),
            'time': np.zeros((B, T, 1), np.float32),
            'target': np.zeros((B, T, 2), np.float32)
        }
        for i in range(B):
            idxForSample = samples_indexes[i]
            # stricly increasing indexes
            assert np.all(0 < np.diff(idxForSample)), 'Invalid indexes: ' + str(idxForSample)
            startT = self._storage[idxForSample[0]]['time']
            endT = self._storage[idxForSample[-1]]['time']
            duration = endT - startT
            assert 0 < duration, 'Invalid duration: ' + str(duration)
            targets_idx = np.sort(targetsIdx[i])
            for j, idx in enumerate(targets_idx):
                data = self._storage[idx]
                Y['points'][i, j] = data['points']
                # eyes should be cropped to 32x32, so we use the central crop
                p = (data['left eye'].shape[0] - 32) // 2
                Y['left eye'][i, j] = data['left eye'][p:p+32, p:p+32]
                Y['right eye'][i, j] = data['right eye'][p:p+32, p:p+32]
                Y['time'][i, j] = (data['time'] - startT) / duration
                Y['target'][i, j] = data['goal']
        # eyes in 0..255, so we need to normalize them
        Y['left eye'] /= 255.0
        Y['right eye'] /= 255.0
        # check that time is between 0 and 1
        assert np.all((0 <= Y['time']) & (Y['time'] <= 1)), 'Invalid time: ' + str(Y['time'])
        for k, v in X.items():
            assert B == v.shape[0], f'Invalid batch size for X[{k}]: {v.shape[0]} != {B} ({v.shape})'
        for k, v in Y.items():
            assert B == v.shape[0], f'Invalid batch size for Y[{k}]: {v.shape[0]} != {B} ({v.shape})'
        return (X, Y)
    
    def merge(self, samples, expected_batch_size):
        X = {}
        for subkey in ['points', 'left eye', 'right eye', 'time', 'userId', 'placeId', 'screenId', 'target']:
            data = [x[subkey] for x, _ in samples]
            X[subkey] = np.concatenate(data, axis=0)
            assert X[subkey].shape[0] == expected_batch_size, 'Invalid batch size: %d != %d' % (X[subkey].shape[0], expected_batch_size)
            continue
        # 
        Y = {}
        for subkey in ['points', 'left eye', 'right eye', 'time', 'target']:
            data = [y[subkey] for _, y in samples]
            Y[subkey] = np.concatenate(data, axis=0)
            assert Y[subkey].shape[0] == expected_batch_size, 'Invalid batch size: %d != %d' % (Y[subkey].shape[0], expected_batch_size)
            continue
        return (X, Y)