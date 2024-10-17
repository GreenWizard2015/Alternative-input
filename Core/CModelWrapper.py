import numpy as np
import NN.networks as networks
from Core.CBaseModel import CBaseModel
  
class CModelWrapper:
  def __init__(self, timesteps, model='simple', user=None, stats=None, **kwargs):
    if user is not None:
      user = {
        'userId': stats['userId'].index(user['userId']),
        'placeId': stats['placeId'].index(user['placeId']),
        'screenId': stats['screenId'].index(user['screenId']),
      }
    self._user = user
    self._timesteps = timesteps
    embeddings = {
      'userId': len(stats['userId']),
      'placeId': len(stats['placeId']),
      'screenId': len(stats['screenId']),
      'size': kwargs.get('embeddingSize', 64),
    }
    self._modelRaw = networks.Face2LatentModel(
      steps=timesteps, latentSize=kwargs.get('latentSize', 64), embeddings=embeddings
    )
    NN =  self._network = self._modelRaw['main']
    self._model = CBaseModel(model=model, embeddings=embeddings, submodels=[NN])
    if 'weights' in kwargs:
      self.load(**kwargs['weights'])
    return
  
  def predict(self, data, **kwargs):
    assert self._user is not None, 'User is not set'
    B = self._timesteps
    userId = kwargs.get('userId', self._user['userId'])
    placeId = kwargs.get('placeId', self._user['placeId'])
    screenId = kwargs.get('screenId', self._user['screenId'])
    # put them as (1, B, ?)
    data['userId'] = np.full((1, B, 1), userId, dtype=np.int32)
    data['placeId'] = np.full((1, B, 1), placeId, dtype=np.int32)
    data['screenId'] = np.full((1, B, 1), screenId, dtype=np.int32)

    data = self._replaceByEmbeddings(data) # replace embeddings
    return self._network(data, training=False)['result'].numpy()
  
  def __call__(self, data, startPos=None):
    predictions = self.predict(data)
    return { 'coords': predictions[0, -1, :], }
  
  @property
  def timesteps(self):
    return self._timesteps
  
  def save(self, folder=None, postfix=''):
    self._model.save(folder=folder, postfix=postfix)

  def load(self, folder=None, postfix='', embeddings=False):
    self._model.load(folder=folder, postfix=postfix, embeddings=embeddings)

  def trainable_variables(self):
    return self._model.trainable_variables()