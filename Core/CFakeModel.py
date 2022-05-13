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

  @tf.function
  def _NerfDecoder_sampleEnergyLossMatrix(self, latent, y, steps, SAMPLED_POINTS):
    B = tf.shape(latent)[0]
    targetY = tf.repeat(y, repeats=SAMPLED_POINTS, axis=0)
    targetE = self._decoder([latent, y], training=True)['valueAt']
    
    EHistory = tf.TensorArray(tf.float32, 1 + steps, dynamic_size=False, clear_after_read=False)
    EHistory = EHistory.write(0, tf.reduce_mean(targetE))
    
    rays = tf.linalg.l2_normalize(tf.random.normal((B * SAMPLED_POINTS, 2)), axis=-1)
    latentForPoints = tf.repeat(latent, repeats=SAMPLED_POINTS, axis=0)
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
    return (tf.math.softplus(EDiff) * mask) / tf.reduce_sum(mask)
  
  @tf.function
  def _trainNerfStep(self, data):
    x, (y, ) = data
    # first pass
    with tf.GradientTape(persistent=True) as tape:
      latent = self._encoder(x, training=True)['latent']
      EMatrix = self._NerfDecoder_sampleEnergyLossMatrix(latent, y, steps=5, SAMPLED_POINTS=128)
      loss = tf.reduce_sum(EMatrix)

    for model in [self._encoder, self._decoder]:
      model.optimizer.minimize(loss, model.trainable_variables, tape=tape)
      continue
    
    totalLoss = loss
    latent = tf.stop_gradient(self._encoder(x, training=True)['latent'])
    # extra passes for decoder
    for _ in tf.range(0):
      with tf.GradientTape(persistent=True) as tape:
        EMatrix = self._NerfDecoder_sampleEnergyLossMatrix(latent, y, steps=5, SAMPLED_POINTS=128)
        loss = tf.reduce_sum(EMatrix)
  
      for model in [self._decoder]:
        model.optimizer.minimize(loss, model.trainable_variables, tape=tape)
        continue
      
      totalLoss += loss
      continue
    return totalLoss

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
    coords = tf.squeeze(tf.stack([tf.reshape(x, (-1,1)), tf.reshape(y, (-1,1))], axis=-1)) # (N, 2)
    
    latent = self._encoder(data, training=training)['latent']
    B = tf.shape(latent)[0]
    res = self.decodeNerfFixedCoords(latent, coords, batch_size, training)
    return tf.reshape(res, (B, NPoints, NPoints))
  
  @tf.function
  def decodeNerfFixedCoords(self, 
    latent, coords, 
    batchSizePerSample=1024, training=False
  ):
    '''
      Coords are SHARED for all samples in batch
    '''
    batch_size = batchSizePerSample
    B = tf.shape(latent)[0]
    NPoints = tf.shape(coords)[0]
    
    res = tf.TensorArray(tf.float32, size=NPoints // batch_size, dynamic_size=False, clear_after_read=False)
    latentForLoop = tf.repeat(latent, batch_size, axis=0)
    for i in tf.range(NPoints // batch_size):
      points = coords[i*batch_size:i*batch_size+batch_size]
      points = tf.tile(points, (B, 1))
      
      values = self._decoder([latentForLoop, points], training=False)['valueAt']
      res = res.write(i, tf.reshape(values, (B, batch_size)))
      continue

    N = res.size()
    res = res.stack()
    res = tf.transpose(res, (1, 0, 2))
    res = tf.reshape(res, (B, -1))
    #######
    # tail
    points = coords[N*batch_size:N*batch_size+batch_size]
    values = self._decoder(
      [
        tf.repeat(latent, tf.shape(points)[0], axis=0),
        tf.tile(points, (B, 1))
      ], training=training
    )['valueAt']
    values = tf.reshape(values, (B, tf.shape(points)[0]))
    return tf.concat([res, values], axis=1)
  
  def eval(self, data):
    if self.useNL: return np.nan
    
    x, (y, ) = data
    loss = tf.reduce_mean(self._trainStepF(x, y)).numpy()
    return loss

  def _coordsLoss(self, ytrue, ypred):
    diff = ypred - ytrue
    l1 = tf.abs(diff)
    dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
    mse = tf.losses.mse(ytrue, ypred)
    return sum([tf.reduce_mean(v) for v in [l1, dist, mse]])