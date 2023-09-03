import os
import NN.networks as networks

class CModelWrapper:
  def __init__(self, timesteps, model='simple', **kwargs):
    self._modelID = model
    self._timesteps = timesteps
    self._modelRaw = networks.Face2LatentModel(steps=timesteps, contexts=None, latentSize=64)
    self._model = self._modelRaw['main']
  
    if 'weights' in kwargs:
      self.load(**kwargs['weights'])
    return
  
  def __call__(self, data, startPos=None):
    predictions = self._model(data, training=False)
    points = predictions['result'][:, -1, :]
    return {
      'coords': points[0].numpy(),
    }

  def _modelFilename(self, folder, postfix=''):
    postfix = '-' + postfix if postfix else ''
    return os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'model', postfix))
  
  def save(self, folder, postfix=''):
    self._model.save_weights(self._modelFilename(folder, postfix))
    return
    
  def load(self, folder, postfix=''):
    self._model.load_weights(self._modelFilename(folder, postfix))
    return
  
  @property
  def timesteps(self):
    return self._timesteps
  