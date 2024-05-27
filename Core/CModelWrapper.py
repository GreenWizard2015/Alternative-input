import os
import numpy as np
import NN.networks as networks
import tensorflow as tf
from tensorflow.keras import layers as L

class CModelWrapper:
  def __init__(self, timesteps, model='simple', user=None, stats=None, **kwargs):
    if user is None:
      user = {
        'userId': 0,
        'placeId': 0,
        'screenId': 0,
      }
    else:
      user = {
        'userId': stats['userId'].index(user['userId']),
        'placeId': stats['placeId'].index(user['placeId']),
        'screenId': stats['screenId'].index(user['screenId']),
      }
    self._user = user

    self._modelID = model
    self._timesteps = timesteps
    embeddings = {
      'userId': len(stats['userId']),
      'placeId': len(stats['placeId']),
      'screenId': len(stats['screenId']),
      'size': 64,
    }
    self._modelRaw = networks.Face2LatentModel(
      steps=timesteps, latentSize=64, embeddings=embeddings
    )
    self._model = self._modelRaw['main']
    self._embeddings = {
      'userId': L.Embedding(len(stats['userId']), embeddings['size']),
      'placeId': L.Embedding(len(stats['placeId']), embeddings['size']),
      'screenId': L.Embedding(len(stats['screenId']), embeddings['size']),
    }
    if 'weights' in kwargs:
      self.load(**kwargs['weights'])
    return
  
  def _replaceByEmbeddings(self, data):
    data = dict(**data) # copy
    for name, emb in self._embeddings.items():
      data[name] = emb(data[name][..., 0])
      continue
    return data
  
  def predict(self, data, **kwargs):
    B = self._timesteps
    userId = kwargs.get('userId', self._user['userId'])
    placeId = kwargs.get('placeId', self._user['placeId'])
    screenId = kwargs.get('screenId', self._user['screenId'])
    # put them as (1, B, ?)
    data['userId'] = np.full((1, B, 1), userId, dtype=np.int32)
    data['placeId'] = np.full((1, B, 1), placeId, dtype=np.int32)
    data['screenId'] = np.full((1, B, 1), screenId, dtype=np.int32)

    data = self._replaceByEmbeddings(data) # replace embeddings
    return self._model(data, training=False)['result'].numpy()
  
  def __call__(self, data, startPos=None):
    predictions = self.predict(data)
    return {
      'coords': predictions[0, -1, :],
    }

  def _modelFilename(self, folder, postfix=''):
    postfix = '-' + postfix if postfix else ''
    return os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'model', postfix))
  
  def save(self, folder=None, postfix=''):
    path = self._modelFilename(folder, postfix)
    self._model.save_weights(path)
    embeddings = {}
    for nm in self._embeddings.keys():
      weights = self._embeddings[nm].get_weights()[0]
      embeddings[nm] = weights
      continue
    np.savez_compressed(path.replace('.h5', '-embeddings.npz'), **embeddings)
    return
    
  def load(self, folder=None, postfix='', embeddings=False):
    path = self._modelFilename(folder, postfix) if not os.path.isfile(folder) else folder
    self._model.load_weights(path)
    if embeddings:
      embeddings = np.load(path.replace('.h5', '-embeddings.npz'))
      for nm in self._embeddings.keys(): # recreate embeddings
        w = embeddings[nm]
        emb = L.Embedding(w.shape[0], w.shape[1])
        emb.build((None, 1))
        emb.set_weights([w])
        self._embeddings[nm] = emb # replace
        continue
    return
  
  def lock(self, isLocked):
    self._model.trainable = not isLocked
    return
  
  @property
  def timesteps(self):
    return self._timesteps
  