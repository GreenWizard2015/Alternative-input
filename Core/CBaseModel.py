import os
import numpy as np
from tensorflow.keras import layers as L

class CBaseModel:
  def __init__(self, model, embeddings, submodels):
    self._model = model
    self._embeddings = {
      'userId': L.Embedding(embeddings['userId'], embeddings['size']),
      'placeId': L.Embedding(embeddings['placeId'], embeddings['size']),
      'screenId': L.Embedding(embeddings['screenId'], embeddings['size']),
    }
    self._submodels = submodels
    return  

  def replaceByEmbeddings(self, data):
    data = dict(**data) # copy
    for name, emb in self._embeddings.items():
      data[name] = emb(data[name][..., 0])
      continue
    return data

  def _modelFilename(self, folder, postfix=''):
    postfix = '-' + postfix if postfix else ''
    return os.path.join(folder, '%s%s.h5' % (self._modelID, postfix))
  
  def save(self, folder=None, postfix=''):
    path = self._modelFilename(folder, postfix)
    if 1 < len(self._submodels):
      for i, model in enumerate(self._submodels):
        model.save_weights(path.replace('.h5', '-%d.h5' % i))
    else:
      self._submodels[0].save_weights(path)

    embeddings = {}
    for nm in self._embeddings.keys():
      weights = self._embeddings[nm].get_weights()[0]
      embeddings[nm] = weights
    
    np.savez_compressed(path.replace('.h5', '-embeddings.npz'), **embeddings)
    
  def load(self, folder=None, postfix='', embeddings=False):
    path = self._modelFilename(folder, postfix) if not os.path.isfile(folder) else folder
    if 1 < len(self._submodels):
      for i, model in enumerate(self._submodels):
        model.load_weights(path.replace('.h5', '-%d.h5' % i))
    else:
      self._submodels[0].load_weights(path)
      
    if embeddings:
      embeddings = np.load(path.replace('.h5', '-embeddings.npz'))
      for nm, emb in self._embeddings.items():
        w = embeddings[nm]
        if not emb.built: emb.build((None, w.shape[0]))
        emb.set_weights([w]) # replace embeddings
    
  def trainable_variables(self):
    parts = list(self._embeddings.values()) + self._submodels
    return sum([p.trainable_variables for p in parts], [])
