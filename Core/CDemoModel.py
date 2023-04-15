import tensorflow as tf
import time
import os
import NN.networks as networks

def _InputSpec():
  return {
    'points': tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
    'left eye': tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
    'right eye': tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
    'time': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
  }
  
class CDemoModel:
  def __init__(self, timesteps, model='simple', **kwargs):
    self._modelID = model
    self._timesteps = timesteps
    self._model = networks.Face2LatentModel(steps=timesteps, contexts=None, latentSize=32)['main']
    self._epoch = 0
    self.trainable = kwargs.get('trainable', True)

    if 'weights' in kwargs:
      self.trainable = kwargs.get('trainable', False)
      self.load(**kwargs['weights'])
    else:
      self._compile()
    return
  
  def _compile(self):
    if not self.trainable: return
    self._model.compile(tf.optimizers.Adam(learning_rate=1e-4))
    return
  
  @tf.function(
    input_signature=[(
      {
        'clean': _InputSpec(),
        'augmented': _InputSpec(),
      },
      (
        tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32),
      )
    )]
  )
  def _trainStep(self, Data):
    print('Instantiate _trainStep')
    ###############
    x, (y, ) = Data
    y = y[..., 0, :]
    losses = []
    TV = self._model.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(TV)
      for data in x.values():
        predictions = self._model(data, training=True)
        for pts in predictions['intermediate']:
          tf.assert_equal(tf.shape(pts), tf.shape(y))
          loss = tf.losses.mse(y, pts)
          losses.append(tf.reduce_mean(loss))
          continue
        continue
      loss = sum(losses)
      losses = {'loss': loss}
  
    self._model.optimizer.minimize(loss, TV, tape=tape)
    ###############
    return losses

  def fit(self, data):
    t = time.time()
    losses = {'loss': 0.0}
    if self.trainable:
      losses = self._trainStep(data)
      losses = {k: v.numpy() for k, v in losses.items()}
      self._epoch += 1
    t = time.time() - t
    return {'epoch': self._epoch, 'time': int(t * 1000), 'losses': losses}
  
  def __call__(self, data, startPos=None):
    predictions = self._model(data, training=False)
    intermediate = predictions['intermediate']
    lastPoints = [pts[:, -1, :] for pts in intermediate]
    points = sum(lastPoints) / len(lastPoints)
    return {
      'coords': points[0].numpy(),
    }

  def save(self, folder, postfix=''):
    if postfix: postfix = '-' + postfix
    self._model.save_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'model', postfix)))
    return
    
  def load(self, folder, postfix=''):
    if postfix: postfix = '-' + postfix
    self._model.load_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'model', postfix)))
    self._compile()
    return
  
  def eval(self, data):
    x, (y, ) = data
    raise Exception('Unknown model (%s)' % (self._modelID, ))

  @property
  def timesteps(self):
    return self._timesteps
