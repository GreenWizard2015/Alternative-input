import networks
import tensorflow as tf
import time

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
  def _inferAR(self, data, training, steps, startPos=0.5):
    # positions = tf.random.uniform(shape=(tf.shape(data[0])[0], 2))
    positions = tf.ones(shape=(tf.shape(data[0])[0], 2), dtype=tf.float32) * startPos
    history = []
    for _ in range(steps + 1):
      positions = self._model([*data, positions], training=training)
      history.append(positions)
      continue
    return history

  def _trainARStep(self, steps):
    def f(x, y):
      pred = self._inferAR(x, training=True, steps=steps)
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
      loss = 0.0
      latent = self._encoder(x, training=True)['latent']
      ############
      rng = tf.linspace(0.0, 1.0, 20)
      xi, yi = tf.meshgrid(rng, rng)
      coords = tf.squeeze(tf.stack([tf.reshape(xi, (-1,1)), tf.reshape(yi, (-1,1))], axis=-1))
      SAMPLED_POINTS = tf.shape(coords)[0]
      
      B = tf.shape(latent)[0]
      targetV = self._decoder([latent, y], training=True)['valueAt']
      targetV = tf.repeat(targetV, repeats=SAMPLED_POINTS, axis=0)
      
      latentForPoints = tf.repeat(latent, repeats=SAMPLED_POINTS, axis=0)
      targetY = tf.repeat(y, repeats=SAMPLED_POINTS, axis=0)
      pointsA = tf.random.uniform((B * SAMPLED_POINTS, 2), 0.0, 1.0)

      for _ in tf.range(3):
        pointsA = tf.clip_by_value(pointsA, clip_value_min=0.0, clip_value_max=1.0)
        
        noiseVec = tf.linalg.l2_normalize(tf.random.normal(tf.shape(pointsA)), axis=-1)
        noiseL = tf.random.uniform((B * SAMPLED_POINTS, 1), 1e-4, 1e-2)
        pointsB = pointsA + noiseVec * noiseL
        
        sampledA = self._decoder([latentForPoints, pointsA], training=True)['valueAt']
        sampledB = self._decoder([latentForPoints, pointsB], training=True)['valueAt']

        # move from B, so if (sampledA - sampledB) == 0 then move to (random) B
        # k = (sampledA - sampledB) / noiseL
        # D = tf.math.divide_no_nan(sampledB, k)
        D = tf.math.divide_no_nan(sampledB * noiseL, sampledA - sampledB)
#         tf.print(D[0], sampledA[0], sampledB[0])
        
        DNoised = 1.0 # tf.random.normal(tf.shape(D), mean=1.0, stddev=0.1)
        pointsA += (pointsB + D * DNoised * noiseVec) - pointsA

        loss += tf.reduce_mean(tf.losses.mse(targetY, pointsB + D * noiseVec))
        loss += tf.reduce_mean(tf.math.softplus(targetV - sampledA)) * 0.1
        loss += tf.reduce_mean(tf.math.softplus(targetV - sampledB)) * 0.1
        continue
      loss = tf.reduce_mean(loss) + tf.reduce_mean(targetV) #* 1e-3 # minimum must be at target

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
    if self.trainable:
      if self.useNL:
        loss = self._trainNerfStep(data).numpy()
      else:
        loss = self._trainStep(data).numpy()
      self._epoch += 1
    t = time.time() - t
    return {'loss': loss, 'epoch': self._epoch, 'time': int(t * 1000)}
  
  def __call__(self, data, startPos=None):
    if self.useAR:
      res = self._inferAR(data, training=False, steps=self._ARDepth, startPos=startPos)
      return {
        'coords': [x.numpy()[0] for x in res] 
      }
    
    if 'simple' == self._modelID:
      return {
        'coords': self._model(data, training=False)['coords'].numpy()
      }
    
    if self.useNL:
      nerf = self.sampleNerf(data).numpy()[0]
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
    self._model.save_weights(filename)
    return
  
  @tf.function
  def sampleNerf(self, data, NPoints=25, batch_size=128, training=False):
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