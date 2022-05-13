import networks
import tensorflow as tf
import time
import numpy as np

class CFakeModel:
  def __init__(self, model='simple', **kwargs):
    self._modelID = model
    self.useAR = 'autoregressive' == model
      
    self._epoch = 0
    self.trainable = kwargs.get('trainable', True)
    
    if self.useAR:
      self._model = networks.ARModel()
      self._ARDepth = kwargs.get('depth', 5)
      self._trainStepF = self._trainARStep(steps=self._ARDepth)
      
    if 'simple' == self._modelID:
      self._model = networks.simpleModel()
      self._trainStepF = self._trainSimpleStep
    
    if 'weights' in kwargs:
      self._model.load_weights(kwargs['weights'])
      self.trainable = kwargs.get('trainable', False)
      
    if self.trainable:
      self._model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=None
      )
    return

  @tf.function
  def _inferAR(self, data, training, steps, positions, F=lambda x: x):
    history = []
    for _ in range(steps + 1):
      positions = self._model([*data, F(positions)], training=training)
      history.append(positions)
      continue
    return history
  
  def _trainARStep(self, steps):
    def f(x, y):
      pred = self._inferAR(
        x, training=True, steps=steps,
        # train on random points
        F=lambda p: tf.stop_gradient(p + tf.random.normal(shape=tf.shape(p), stddev=0.1)),
        positions=tf.random.uniform(shape=tf.shape(y), minval=-0.1, maxval=1.1)
      )
      loss = 0.0
      for coords in pred:
        loss = loss + self._coordsLoss(y, coords)
        continue
      return loss / len(pred)
    return f

  def _trainSimpleStep(self, x, y):
    pred = self._model(x, training=True)
    return self._coordsLoss(y, pred['coords'])

  @tf.function
  def _trainStep(self, data):
    x, (y, ) = data
    with tf.GradientTape() as tape:
      loss = tf.reduce_mean(self._trainStepF(x, y))

    model = self._model
    model.optimizer.minimize(loss, model.trainable_variables, tape=tape)
    return loss

  def fit(self, data):
    t = time.time()
    loss = 0.0
    extra = {}
    if self.trainable:
      loss = self._trainStep(data).numpy()
      self._epoch += 1
    t = time.time() - t
    return {'loss': loss, 'epoch': self._epoch, 'time': int(t * 1000), **extra}
  
  def __call__(self, data, startPos=None):
    if self.useAR:
      res = self._inferAR(data, training=False, steps=self._ARDepth, positions=startPos)
      return {
        'coords': [x.numpy()[0] for x in res]
      }
    
    if 'simple' == self._modelID:
      return {
        'coords': self._model(data, training=False)['coords'].numpy()
      }
    
    raise Exception('Unknown model (%s)' % (self._modelID, ))
    
  def debug(self, data):
    return
    
  def save(self, filename):
    self._model.save_weights(filename)
    return

  def eval(self, data):
    x, (y, ) = data
    loss = tf.reduce_mean(self._trainStepF(x, y)).numpy()
    return loss

  def _coordsLoss(self, ytrue, ypred):
    diff = ypred - ytrue
    l1 = tf.abs(diff)
    dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
    mse = tf.losses.mse(ytrue, ypred)
    return sum([tf.reduce_mean(v) for v in [l1, dist, mse]])