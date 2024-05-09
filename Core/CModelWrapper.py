import os
import NN.networks as networks

class CModelWrapper:
  def __init__(self, timesteps, model='simple', **kwargs):
    self._modelID = model
    self._timesteps = timesteps
    self._modelRaw = networks.Face2LatentModel(steps=timesteps, latentSize=64)
    self._model = self._modelRaw['main']
  
    if 'weights' in kwargs:
      self.load(**kwargs['weights'])
    return
  
  def predict(self, data):
    return self._model(data, training=False)['result'].numpy()
  
  def __call__(self, data, startPos=None):
    predictions = self.predict(data)
    return {
      'coords': predictions[0, -1, :],
    }

  def _modelFilename(self, folder, postfix=''):
    postfix = '-' + postfix if postfix else ''
    return os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'model', postfix))
  
  def save(self, folder=None, postfix='', path=None):
    if path is None: path = self._modelFilename(folder, postfix)
    self._model.save_weights(path)
    return
    
  def load(self, folder=None, postfix='', path=None):
    if path is None: path = self._modelFilename(folder, postfix)
    self._model.load_weights(path)
    return
  
  @property
  def timesteps(self):
    return self._timesteps
  