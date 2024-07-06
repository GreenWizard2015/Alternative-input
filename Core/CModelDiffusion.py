import os
import numpy as np
import NN.networks as networks
import tensorflow as tf
import tensorflow_probability as tfp
import NN.Utils as NNU
import time
from tensorflow.keras import layers as L

# TODO: Implement the standard diffusion process (with the prediction of the noise, proper sampling, etc)
class CModelDiffusion:
  '''
  Wrapper for the diffusion model to predict the gaze point
  Diffusion T is equal to the stddev of the gaussian noise
  '''
  def __init__(self, timesteps, model='simple', user=None, stats=None, use_encoders=False, **kwargs):
    if user is None:
      user = {
        'userId': 0,
        'placeId': 0,
        'screenId': 0,
      }
    else:
      user = {
        'userId': stats['userId'].index(user['userId']),
        'placeId': stats['placeId'].index(user['placeId']),
        'screenId': stats['screenId'].index(user['screenId']),
      }
    self._user = user

    self._modelID = model
    self._timesteps = timesteps
    embeddings = {
      'userId': len(stats['userId']),
      'placeId': len(stats['placeId']),
      'screenId': len(stats['screenId']),
      'size': 64,
    }
    self._modelRaw = networks.Face2LatentModel(
      steps=timesteps, latentSize=64, embeddings=embeddings,
      diffusion=True
    )
    self._model = self._modelRaw['main']
    self._embeddings = {
      'userId': L.Embedding(len(stats['userId']), embeddings['size']),
      'placeId': L.Embedding(len(stats['placeId']), embeddings['size']),
      'screenId': L.Embedding(len(stats['screenId']), embeddings['size']),
    }
    self._intermediateEncoders = {}
    if use_encoders:
      shapes = self._modelRaw['intermediate shapes']
      for name, shape in shapes.items():
        enc = networks.IntermediatePredictor(name='%s-encoder' % name)
        enc.build(shape)
        self._intermediateEncoders[name] = enc
        continue
   
    self._maxDiffusionT = 100.0
    if 'weights' in kwargs:
      self.load(**kwargs['weights'])
    self.compile()
    # add signatures to help tensorflow optimize the graph
    specification = self._modelRaw['inputs specification']
    self._trainStep = tf.function(
      self._trainStep,
      input_signature=[
        (
          { 'clean': specification, 'augmented': specification, },
          ( tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32), )
        )
      ]
    )
    self._eval = tf.function(
      self._eval,
      input_signature=[(
        specification,
        ( tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32), )
      )]
    )

    return
  
  def _step2mean(self, step):
    step = tf.cast(step, tf.float32) / self._maxDiffusionT
    step = tf.cast(step, tf.float32) + 1e-6
    # step = tf.pow(step, 2.0) # make it decrease faster
    return tf.clip_by_value(step, 1e-3, 1.0)
  
  def _replaceByEmbeddings(self, data):
    data = dict(**data) # copy
    for name, emb in self._embeddings.items():
      data[name] = emb(data[name][..., 0])
      continue
    return data
  
  def _makeGaussian(self, mean, stddev):
    stddev = tf.concat([stddev, stddev], axis=-1)
    return tfp.distributions.MultivariateNormalDiag(mean, stddev)
  
  @tf.function
  def _infer(self, data, training=False):
    print('Instantiate _infer')
    data = self._replaceByEmbeddings(data)
    shp = tf.shape(data['userId'])
    B, N = shp[0], self.timesteps
    result = tf.zeros((B, N, 2), dtype=tf.float32)
    for step in tf.range(self._maxDiffusionT, -1, -5):
      mean = self._step2mean(
        tf.fill((B, N, 1), step)
      )
      stepData = dict(**data)
      stepData['diffusionT'] = mean
      stepData['diffusionPoints'] = tf.random.normal((B, N, 2), mean=result, stddev=mean)
      result = self._model(stepData, training=training)['result']
    return result
    
  def predict(self, data, **kwargs):
    B = self._timesteps
    userId = kwargs.get('userId', self._user['userId'])
    placeId = kwargs.get('placeId', self._user['placeId'])
    screenId = kwargs.get('screenId', self._user['screenId'])
    # put them as (1, B, ?)
    data['userId'] = np.full((1, B, 1), userId, dtype=np.int32)
    data['placeId'] = np.full((1, B, 1), placeId, dtype=np.int32)
    data['screenId'] = np.full((1, B, 1), screenId, dtype=np.int32)

    data = self._replaceByEmbeddings(data) # replace embeddings
    
    result = self._infer(data)
    return result.numpy()
  
  def __call__(self, data, startPos=None):
    predictions = self.predict(data)
    return {
      'coords': predictions[0, -1, :],
    }
    
  def compile(self):
    self._optimizer = NNU.createOptimizer()
    return

  def _modelFilename(self, folder, postfix=''):
    postfix = '-' + postfix if postfix else ''
    return os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'model', postfix))
  
  def save(self, folder=None, postfix=''):
    path = self._modelFilename(folder, postfix)
    self._model.save_weights(path)
    embeddings = {}
    for nm in self._embeddings.keys():
      weights = self._embeddings[nm].get_weights()[0]
      embeddings[nm] = weights
      continue
    np.savez_compressed(path.replace('.h5', '-embeddings.npz'), **embeddings)
    # save intermediate encoders
    if self._intermediateEncoders:
      encoders = {}
      for nm, encoder in self._intermediateEncoders.items():
        # save each variable separately
        for ww in encoder.trainable_variables:
          encoders['%s-%s' % (nm, ww.name)] = ww.numpy()
        continue
      np.savez_compressed(path.replace('.h5', '-intermediate-encoders.npz'), **encoders)
    return
    
  def load(self, folder=None, postfix='', embeddings=False):
    path = self._modelFilename(folder, postfix) if not os.path.isfile(folder) else folder
    self._model.load_weights(path)
    if embeddings:
      embeddings = np.load(path.replace('.h5', '-embeddings.npz'))
      for nm, emb in self._embeddings.items():
        w = embeddings[nm]
        if not emb.built: emb.build((None, w.shape[0]))
        emb.set_weights([w]) # replace embeddings
        continue
    
    if self._intermediateEncoders:
      encodersName = path.replace('.h5', '-intermediate-encoders.npz')
      if os.path.isfile(encodersName):
        encoders = np.load(encodersName)
        for nm, encoder in self._intermediateEncoders.items():
          for ww in encoder.trainable_variables:
            w = encoders['%s-%s' % (nm, ww.name)]
            ww.assign(w)
          continue
    return
  
  def lock(self, isLocked):
    self._model.trainable = not isLocked
    return
  
  @property
  def timesteps(self):
    return self._timesteps
  
  def trainable_variables(self):
    parts = list(self._embeddings.values()) + [self._model] + list(self._intermediateEncoders.values())
    return sum([p.trainable_variables for p in parts], [])  
  
  def _pointLoss(self, ytrue, ypred):
    # pseudo huber loss
    delta = 0.01
    tf.assert_equal(tf.shape(ytrue), tf.shape(ypred))
    diff = tf.square(ytrue - ypred)
    loss = tf.sqrt(diff + delta ** 2) - delta
    tf.assert_equal(tf.shape(loss), tf.shape(ytrue))
    return tf.reduce_mean(loss, axis=-1)

  def _trainStep(self, Data):
    print('Instantiate _trainStep')
    ###############
    x, (y, ) = Data
    y = y[..., 0, :]
    losses = {}
    with tf.GradientTape() as tape:
      data = x['augmented']
      data = self._replaceByEmbeddings(data)
      # add sampled T
      B = tf.shape(y)[0]
      N = self.timesteps
      maxT = 100
      diffusionT = tf.random.uniform((B, 1), minval=0, maxval=maxT, dtype=tf.int32)
      # (B, 1) -> (B, N, 1)
      diffusionT = tf.tile(diffusionT, (1, N))[..., None]
      diffusionT = self._step2mean(diffusionT)
      tf.assert_equal(tf.shape(diffusionT), (B, N, 1))
      
      # store the diffusion parameters
      data['diffusionT'] = diffusionT
      # sample the points
      data['diffusionPoints'] = tf.random.normal((B, N, 2), mean=y, stddev=diffusionT)
      predictions = self._model(data, training=True)
    #   intermediate = predictions['intermediate']
    #   assert len(intermediate) == 0, 'Intermediate predictions are not supported'
      
      predictedMean = predictions['result']
      gaussian = self._makeGaussian(predictedMean, diffusionT)
      losses['log_prob'] = tf.reduce_mean(
        -gaussian.log_prob(y)
      )
      losses['points'] = self._pointLoss(y, predictedMean)
      loss = sum(losses.values())
      losses['loss'] = loss
  
    self._optimizer.minimize(loss, tape.watched_variables(), tape=tape)
    ###############
    return losses

  def fit(self, data):
    t = time.time()
    losses = self._trainStep(data)
    losses = {k: v.numpy() for k, v in losses.items()}
    return {'time': int((time.time() - t) * 1000), 'losses': losses}
  
  def _eval(self, xy):
    print('Instantiate _eval')
    x, (y,) = xy
    y = y[:, :, 0]
    B, N = tf.shape(y)[0], tf.shape(y)[1]
    
    predictions = self._infer(x)
    
    mean = self._step2mean(tf.fill((B, N, 1), 0))
    gaussian = self._makeGaussian(predictions, mean)
    loss = tf.nn.sigmoid( -gaussian.log_prob(y) )
    points = predictions
    _, dist = NNU.normVec(y - predictions)
    return loss, points, dist

  def eval(self, data):
    loss, sampled, dist = self._eval(data)
    return loss.numpy(), sampled.numpy(), dist.numpy()