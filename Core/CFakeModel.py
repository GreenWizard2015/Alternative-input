import networks
import tensorflow as tf
import time
import numpy as np

# TODO: Fix nerf saving/loading
class CFakeModel:
  def __init__(self, model='simple', **kwargs):
    self._modelID = model
    self.useNL = 'NerfLike' == model
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

    if self.useNL:
      self._encoder = networks.NerfLikeEncoder()
      self._decoder = networks.NerfLikeDecoder()
      self._trainStepF = self._trainNerfStep
      
      self._encoder.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=None
      )
      self._decoder.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=None
      )
      if 'weights' in kwargs:
        self._encoder.load_weights('encoder.h5')
        self._decoder.load_weights('decoder.h5')
      return
    
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
  def _inferAR(self, data, training, steps, positions=None, F=lambda x: x):
    if positions is None:
      positions = tf.random.normal(shape=(tf.shape(data[0])[0], 2), mean=0.5, stddev=0.1)
      pass
    
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
        F=tf.stop_gradient
      )
      loss = 0.0
      for coords in pred:
        loss = loss + tf.losses.mse(y, coords)
      return loss
    return f

  @tf.function
  def _trainNerfStep(self, data):
    x, (y, ) = data
    with tf.GradientTape(persistent=True) as tape:
      SAMPLED_POINTS = 32 * 4
      targetY = tf.repeat(y, repeats=SAMPLED_POINTS, axis=0)
      latent = self._encoder(x, training=True)['latent']
      ############
      B = tf.shape(latent)[0]
      targetE = self._decoder([latent, y], training=True)['valueAt']
      rays = tf.linalg.l2_normalize(tf.random.normal((B * SAMPLED_POINTS, 2)), axis=-1)
      
      latentForPoints = tf.repeat(latent, repeats=SAMPLED_POINTS, axis=0)
      steps = 15
      EHistory = tf.TensorArray(tf.float32, 1 + steps, dynamic_size=False, clear_after_read=False)
      EHistory = EHistory.write(0, tf.reduce_mean(targetE))
      LSteps = tf.range(1 + steps, dtype=tf.float32) * 1.1 / (0.0 + steps)
      LSteps = tf.square(LSteps)
      for i in tf.range(steps):
        raysL = tf.random.uniform((B * SAMPLED_POINTS, 2), LSteps[i], LSteps[i + 1])
        points = targetY + rays * raysL
        E = self._decoder([latentForPoints, points], training=True)['valueAt']
        EHistory = EHistory.write(1 + i, tf.reduce_mean(E))
        continue
      ######################
      E = EHistory.stack()
      N = tf.shape(E)[-1]
      mask = tf.range(N)
      mask = tf.cast(tf.reshape(mask, (N, 1)) < tf.repeat(mask[None], N, axis=0), tf.float32)
      EDiff = tf.reshape(E, (N, 1)) - tf.repeat(E[None], N, axis=0)
      EMatrix = tf.math.softplus(EDiff) * mask
      loss = tf.reduce_sum(EMatrix) #/ tf.reduce_sum(mask)

    for model in [self._encoder, self._decoder]:
      grads = tape.gradient(loss, model.trainable_weights)
      model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
      continue
    return loss

  def _trainSimpleStep(self, x, y):
    pred = self._model(x, training=True)
    return tf.losses.mse(y, pred['coords'])

  @tf.function
  def _trainStep(self, data):
    x, (y, ) = data
    with tf.GradientTape() as tape:
      loss = tf.reduce_mean(self._trainStepF(x, y))

    model = self._model
    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

  def fit(self, data):
    t = time.time()
    loss = 0.0
    extra = {}
    if self.trainable:
      if self.useNL:
        loss = self._trainNerfStep(data).numpy()
      else:
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
    
    if self.useNL:
      nerf = self.sampleNerf(data).numpy()[0]
      # nerf = np.log(1.0 + nerf)
      r = nerf.max() - nerf.min()
      if r <= 0.0: r = 1.0
      nerf = (nerf - nerf.min()) / r
      
      return {
        'coords': [[0.5, 0.5]],
        'nerf': nerf 
      }
    raise Exception('Unknown model (%s)' % (self._modelID, ))
    
  def debug(self, data):
    res = self._model(data, training=False)
    print(res['raw coords'].numpy())
    print(res['coords'].numpy())
    return
    
  def save(self, filename):
    if self.useNL:
      self._encoder.save_weights('encoder.h5')
      self._decoder.save_weights('decoder.h5')
      return
    self._model.save_weights(filename)
    return
  
  @tf.function
  def sampleNerf(self, data, NPoints=128, batch_size=1024, training=False):
    rng = tf.linspace(0.0, 1.0, NPoints)
    x, y = tf.meshgrid(rng, rng)
    coords = tf.squeeze(tf.stack([tf.reshape(x, (-1,1)), tf.reshape(y, (-1,1))], axis=-1))
    
    latent = self._encoder(data, training=training)['latent']
    B = tf.shape(latent)[0]
    res = tf.TensorArray(tf.float32, size=tf.shape(coords)[0] // batch_size, dynamic_size=False, clear_after_read=False)
    for i in tf.range(tf.shape(coords)[0] // batch_size):
      points = coords[i*batch_size:i*batch_size+batch_size]
      
      values = self._decoder(
        [
          tf.repeat(latent, tf.shape(points)[0], axis=0),
          tf.tile(points, (B, 1))
        ], training=False
      )['valueAt']
      values = tf.reshape(values, (B, tf.shape(points)[0]))
      res = res.write(i, values)
      continue

    N = res.size()
    res = res.stack()
    res = tf.transpose(res, (1, 0, 2))
    res = tf.reshape(res, (B, -1))
    #######
    points = coords[N*batch_size:N*batch_size+batch_size]
    values = self._decoder(
      [
        tf.repeat(latent, tf.shape(points)[0], axis=0),
        tf.tile(points, (B, 1))
      ], training=training
    )['valueAt']
    values = tf.reshape(values, (B, tf.shape(points)[0]))
    res = tf.concat([res, values], axis=1)
    return tf.reshape(res, (B, NPoints, NPoints))