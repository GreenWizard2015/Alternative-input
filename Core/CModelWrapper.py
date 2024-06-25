import os
import numpy as np
import NN.networks as networks
import tensorflow as tf
from tensorflow.keras import layers as L

class CModelWrapper:
  def __init__(self, timesteps, model='simple', user=None, stats=None, use_encoders=True, **kwargs):
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
    self._intermediateEncoders = {}
    if use_encoders:
      shapes = self._modelRaw['intermediate shapes']
      for name, shape in shapes.items():
        enc = networks.IntermediatePredictor(name='%s-encoder' % name)
        enc.build(shape)
        self._intermediateEncoders[name] = enc
        continue
   
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
    # save intermediate encoders
    if self._intermediateEncoders:
      encoders = {}
      for nm, encoder in self._intermediateEncoders.items():
        # save each variable separately
        for ww in encoder.trainable_variables:
          encoders['%s-%s' % (nm, ww.name)] = ww.numpy()
        continue
      np.savez_compressed(path.replace('.h5', '-intermediate-encoders.npz'), **encoders)
    return
    
  def load(self, folder=None, postfix='', embeddings=False):
    path = self._modelFilename(folder, postfix) if not os.path.isfile(folder) else folder
    self._model.load_weights(path)
    if embeddings:
      embeddings = np.load(path.replace('.h5', '-embeddings.npz'))
      for nm, emb in self._embeddings.items():
        w = embeddings[nm]
        if not emb.built: emb.build((None, w.shape[0]))
        emb.set_weights([w]) # replace embeddings
        continue
    
    if self._intermediateEncoders:
      encodersName = path.replace('.h5', '-intermediate-encoders.npz')
      if os.path.isfile(encodersName):
        encoders = np.load(encodersName)
        for nm, encoder in self._intermediateEncoders.items():
          for ww in encoder.trainable_variables:
            w = encoders['%s-%s' % (nm, ww.name)]
            ww.assign(w)
          continue
    return
  
  def lock(self, isLocked):
    self._model.trainable = not isLocked
    return
  
  @property
  def timesteps(self):
    return self._timesteps
  
  def trainable_variables(self):
    parts = list(self._embeddings.values()) + [self._model] + list(self._intermediateEncoders.values())
    return sum([p.trainable_variables for p in parts], [])