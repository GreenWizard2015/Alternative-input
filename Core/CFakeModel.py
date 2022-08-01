import networks
import tensorflow as tf
import time
import numpy as np
import os
from NN.MIModel import MIModel
from NN.CGMModel import CGMModel
import NN.Utils as NNU
import tensorflow.python.ops.resource_variable_ops as resource_variable_ops

class CFakeModel:
  def __init__(self, model='simple', **kwargs):
    F2LArgs = kwargs.get('F2LArgs', {})
    modelArgs = kwargs.get('modelArgs', {})
    self._modelID = model
    self._useAR = 'autoregressive' == model
    
    self._face2latent = networks.Face2LatentModel(**F2LArgs)
    FACE_LATENT_SIZE = self._face2latent.output_shape['latent'][-1]
    modelArgs['FACE_LATENT_SIZE'] = FACE_LATENT_SIZE
    
    # self._MI = MIModel()
    self._latent2GMM = CGMModel()
    self._latent2GMM.build((None, FACE_LATENT_SIZE))
    
    self._epoch = 0
    self.trainable = kwargs.get('trainable', True)
    if self._useAR:
      self._model = networks.ARModel(**modelArgs)
      self._ARDepth = kwargs.get('depth', 5)
      self._ARStep = kwargs.get('step', 0.01)
      self._trainStepF = self._trainARStep(steps=self._ARDepth)
      
    if 'simple' == self._modelID:
      self._trainStepF = self._trainSimpleStep
    
    if 'weights' in kwargs:
      self.trainable = kwargs.get('trainable', False)
      self.load(**kwargs['weights'])
    else:
      self._compile()
    return
  
  def _compile(self):
    if not self.trainable: return
    self._optimizer = tf.optimizers.Adam(learning_rate=1e-4)
#     self._face2latent.compile(
#       optimizer=tf.keras.optimizers.Adam(1e-4),
#       loss=None
#     )
#     self._model.compile(
#       optimizer=tf.keras.optimizers.Adam(1e-4),
#       loss=None
#     )
    return

  def _normVec(self, x):
    V, L = tf.linalg.normalize(x, axis=-1)
    V = tf.where(tf.math.is_nan(V), 0.0, V)
    return(V, L)

  def _ARClip(self, a, b, step):
    V, L = self._normVec(b - a)
    L = tf.minimum(L, step)
    return a + V * L

  @tf.function
  def _dynamicAR(self, latent, stepsLimit, positions, returnHistory):
    B = tf.shape(positions)[0]
    allIndices = tf.range(B)[..., None]
    msk = tf.fill((B,), True)
    stepSize = self._ARStep
    ittr = tf.constant(0)
    history = tf.TensorArray(tf.float32, stepsLimit, dynamic_size=False, clear_after_read=True)
    while tf.logical_and(tf.reduce_any(msk), ittr < stepsLimit):
      indices = tf.boolean_mask(allIndices, msk, axis=0)
      activePos = tf.boolean_mask(positions, msk, axis=0)
      activeLatent = tf.boolean_mask(latent, msk, axis=0)
      
      pred = self._model([activeLatent, activePos], training=False)['coords']
      newValues = self._ARClip(activePos, pred, stepSize)

      newPos = tf.tensor_scatter_nd_update(positions, indices, newValues)
      _, dist = self._normVec(newPos - positions)
      msk = (stepSize * 0.01) < dist[:, 0]
      positions = newPos
      history = history.write(ittr, positions)
      ittr += 1
      continue

    if not returnHistory: return positions
    return history.stack()[:ittr]

  @tf.function
  def _inferAR(self, data, steps, positions):
    return self._dynamicAR(
      latent=self._face2latent(data, training=False)['latent'], 
      stepsLimit=steps, 
      positions=positions, returnHistory=True
    )
    
  def _trainARStep(self, steps):
    @tf.function
    def f(x, y):
      y = y[:, 0]
      B = tf.shape(y)[0]
      F2L = self._face2latent(x, training=True)
      latent = F2L['latent']

      NSamples = 16 * 16
      yN = tf.repeat(y, NSamples, axis=0)
      latentN = tf.repeat(latent, NSamples, axis=0)
      
      stddev = tf.linspace(self._ARStep / 3.0, 1.0, NSamples)[:, None]
      stddev = tf.repeat(stddev, B, axis=0)
      
      BN = tf.shape(yN)[0]
      NSteps = 1
      loss = tf.TensorArray(yN.dtype, NSteps, dynamic_size=False, clear_after_read=False)
      for i in tf.range(NSteps):
        maxStepL = self._ARStep
        vec, d = self._normVec(tf.random.normal((BN, 2), stddev=stddev))
        d = tf.maximum(maxStepL * 0.1, d)
        initP = yN + vec * d
        predictions = self._model([latentN, initP], training=True)
        coords = predictions['coords']
        
        _, dist = self._normVec(yN - coords[:, None, :])
        dist = dist[..., 0]
        bestMask = tf.one_hot(
          tf.argmin(dist, axis=-1),
          depth=tf.shape(dist)[-1]
        )
        ybest = tf.reduce_sum(yN * bestMask[..., None], axis=-2)
      
        V, L = self._normVec(predictions['coords'] - initP)
        goalV, goalL = self._normVec(ybest - initP)
  
        losses = [
          1.0 + tf.losses.cosine_similarity(goalV, V),
          tf.losses.mape(
            tf.minimum(goalL, maxStepL),
            tf.minimum(L, maxStepL)
          ) / 100.0,
          #self._coordsLoss(yN, predictions['coords'])[:, 0] * 0.1
        ]
        losses = tf.reshape(sum(losses), (B, NSamples))
        losses = tf.reduce_mean(losses, axis=-1)
        curLoss = tf.reduce_mean(losses)
        loss = loss.write(i, curLoss)
        continue
      loss = loss.stack()
      return tf.reduce_mean(loss)
    return f
    
  def _trainSimpleStep(self, x, y):
    B = tf.shape(y)[0]
    N = tf.shape(y)[1]
    KP = tf.shape(y)[2]
    y = tf.reshape(y, tf.concat(((-1,), tf.shape(y)[-2:]), axis=-1))
    extraLoss = 0.0
    
    latent = self._face2latent(x['augmented'], training=True)['latent']
    extraLoss += sum(map(tf.reduce_mean, self._face2latent.losses))
    latent = tf.reshape(latent, (-1, tf.shape(latent)[-1]))

    _, distr = self._latent2GMM(latent, training=True)
    extraLoss += sum(map(tf.reduce_mean, self._latent2GMM.losses))
    ######################
    def gmmPerPointNLL(points):
      points = tf.reshape(points, (B, N, 2))
      points = tf.transpose(points, (1, 0, 2))
      tf.assert_equal(tf.shape(points), (N, B, 2))
      points = tf.repeat(points, N, axis=1)
      tf.assert_equal(tf.shape(points), (N, B * N, 2))
      loss = distr.log_prob(points)
      tf.assert_equal(tf.shape(loss), (N, B * N, ))
      return loss
    ######################
    GMMRealLoss = gmmPerPointNLL(y[:, 0])

    # y = (B, N, P, 2)
    forecastGT = tf.reshape(y, (B * N, KP, 2))
    forecastGT = forecastGT[:, 1:] # drop y[:, 0]
    forecastGT = tf.transpose(forecastGT, (1, 0, 2))
    tf.assert_equal(tf.shape(forecastGT), (KP - 1, B * N, 2))
    GMMForecastLoss = distr.log_prob(forecastGT)
    tf.assert_equal(tf.shape(GMMForecastLoss), (KP - 1, B * N, ))
    return {
      'GMM real': tf.nn.sigmoid(-tf.reduce_mean(GMMRealLoss)),
      'GMM forecast': tf.nn.sigmoid(-tf.reduce_mean(GMMForecastLoss)),
      'extra': extraLoss,
    }

  @tf.function(
    input_signature=[(
      {
        'clean': (
          tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
          tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
          tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        ),
        'augmented': (
          tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
          tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
          tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        ),
      },
      (
        tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32),
      )
    )]
  )
  def _trainStep(self, data):
    print('Instantiate _trainStep')
    x, (y, ) = data
    with tf.GradientTape(persistent=True) as tape:
      losses = self._trainStepF(x, y)
      losses = {k: tf.reduce_mean(v) for k, v in losses.items()}
      losses['total'] = loss = sum(losses.values(), 0.0)

    models = [self._face2latent, self._latent2GMM]
    self._optimizer.minimize(
      loss,
      sum([x.trainable_variables for x in models], []),
      tape=tape
    )
    ###############
#     startPt = Y[:, 0, 0]
#     startPt = tf.repeat(startPt, N, axis=0)
#     states = tf.concat(self._normVec(startPt - predictions['coords']), axis=-1)
#     with tf.GradientTape(persistent=True) as tape:
#       MI = self._MI(
#         x=tf.stop_gradient(latent),
#         y=tf.stop_gradient(states),
#         mask=mask,
#         training=True,
#       )
#       loss = -tf.reduce_mean(MI)
#       losses['MI'] = loss
#       losses['total'] += loss
# 
#     models = [self._MI]
#     for model in models:
#       if model.trainable and not(model.optimizer is None):
#         model.optimizer.minimize(loss, model.trainable_variables, tape=tape)
#       continue
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

  @tf.function(
    input_signature=[
      (
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
      ),
      resource_variable_ops.VariableSpec(shape=(None, 2), dtype=tf.float32)
    ]
  )
  def _gmmAR(self, data, pos):
    latent = self._face2latent(data, training=False)['latent'][:, -1]
    gmm, _ = self._latent2GMM(latent, training=False)
    distr = self._latent2GMM.distribution(gmm, nograds=True)
    pos.assign(distr.sample())
    start = tf.identity(pos)
    N = 50
    history = tf.TensorArray(tf.float32, N, dynamic_size=False, clear_after_read=True)
    for i in tf.range(N):
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(pos)
        loss = -distr.log_prob(pos) + self._coordsLoss(start, pos)
        pass
      
      (dx,) = tape.gradient(loss, [pos])
      vec, dst = self._normVec(dx)
      dx = vec * 0.005
      pos.assign_sub( dx + tf.random.normal(tf.shape(pos))*1e-5 )
      history = history.write(i, tf.identity(pos))
      continue
    return history.stack()[::5, 0]
  
  def __call__(self, data, startPos=None):
    if self._useAR:
      res = self._inferAR(data, steps=self._ARDepth, positions=startPos)
      return {
        'coords': [x.numpy()[0] for x in res]
      }

    if 'simple' == self._modelID:
      if not(startPos is None):
        return {
          'coords': self._gmmAR(
            data, tf.Variable( startPos, tf.float32),
          ).numpy()
        }
      #######################
      latent = self._face2latent(data, training=False)['latent'][:, -1]
      gmm, distr = self._latent2GMM(latent, training=False)
      points = distr.sample(1500)
      points = tf.reduce_mean(points, 0)
      return {
        'coords': points.numpy(),
        'distribution': distr,
        'gmm': gmm
      }
    
    raise Exception('Unknown model (%s)' % (self._modelID, ))
    
  def save(self, folder, postfix=''):
    if postfix: postfix = '-' + postfix
    self._face2latent.save_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'F2L', postfix)))
    self._latent2GMM.save_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'GMM', postfix)))
    return
    
  def load(self, folder, postfix=''):
    if postfix: postfix = '-' + postfix
    self._face2latent.load_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'F2L', postfix)))
    self._latent2GMM.load_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'GMM', postfix)))
    
    self._compile()
    return

  @tf.function
  def _evalAR(self, data, y, steps=100):
    predicted = self._dynamicAR(
      latent=self._face2latent(data, training=False)['latent'], 
      stepsLimit=steps,
      positions=tf.ones_like(y[:, 0]) * 0.5,
      returnHistory=False
    )
    _, dist = self._normVec(predicted - y[:, 0])
    return dist, predicted
  
  def eval(self, data):
    x, (y, ) = data
    if self._useAR:
      dist, pred = self._evalAR(x, y)
      return dist.numpy(), pred.numpy()

    if 'simple' == self._modelID:
      y = y[:, -1, 0]
      gmm, distr = self._latent2GMM(
        self._face2latent(x, training=False)['latent'][:, -1],
        training=False
      )

      loss = tf.nn.sigmoid(-distr.log_prob(y))
      tf.assert_equal(tf.shape(loss), tf.shape(y)[:1])
      sampled = distr.sample(15)
      sampled = tf.transpose(sampled, (1, 0, 2))
      sampled = tf.reduce_mean(sampled, -2)
      tf.assert_equal(tf.shape(sampled), tf.shape(y))
      ##############
      _, dist = self._normVec(y[:, None] - gmm['mu'])
      dist = tf.reduce_min(dist[..., 0], axis=-1)
      ##############
      return loss.numpy(), sampled.numpy(), dist.numpy()

    raise Exception('Unknown model (%s)' % (self._modelID, ))

  def _coordsLoss(self, ytrue, ypred):
    # tf.assert_equal(tf.shape(ytrue), tf.shape(ypred))
    diff = ypred - ytrue
    _, dist = self._normVec(diff)
    L1Loss = tf.reduce_sum(tf.abs(diff), axis=-1, keepdims=True)
    L2Loss = tf.reduce_mean(tf.square(diff), axis=-1, keepdims=True)
    tf.assert_equal(tf.shape(L1Loss), tf.shape(dist))
    tf.assert_equal(tf.shape(L2Loss), tf.shape(dist))
    return dist + L1Loss + L2Loss

  @property
  def timesteps(self):
    return 5
    s = self._face2latent.inputs[0].shape # face mesh
    if 3 == len(s): return None
    return s[1]