import NN.networks as networks
import tensorflow as tf
import time
import os
from NN.CGMModel import CGMModel as CGMModelNet
from NN.CGMModelDiscriminator import CGMModelDiscriminator
import NN.Utils as NNU
import numpy as np
from collections import defaultdict
import tensorflow_probability as tfp

def GMMInputSpec():
  return {
    'points': tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
    'left eye': tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
    'right eye': tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
    'ContextID': tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
    'time': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
  }
  
class CGMModel:
  def __init__(self, useDiscriminator=True, L1Reg=0.0, **kwargs):
    self._L1Reg = L1Reg
    self._useDiscriminator = useDiscriminator

    F2LArgs = kwargs.get('F2LArgs', {})
    self._modelID = 'simple'
    
    F2L = kwargs.get('F2L', None)
    if F2L is None:
      F2L = networks.Face2LatentModel(**F2LArgs)
      pass
    
    self._face2latent = F2L['main']
    self._face2step = F2L['Face2Step']
    self._step2latent = F2L['Step2Latent']
    FACE_LATENT_SIZE = self._face2latent.output_shape['latent'][-1]
    
    self._latent2GMM = CGMModelNet()
    self._latent2GMM.build((None, FACE_LATENT_SIZE))

    if self._useDiscriminator:
      self._discriminator = CGMModelDiscriminator()
      self._discriminator.build((None, FACE_LATENT_SIZE))
    
    self._epoch = 0
    self.trainable = kwargs.get('trainable', True)

    if 'weights' in kwargs:
      self.trainable = kwargs.get('trainable', False)
      self.load(**kwargs['weights'])
    else:
      self._compile()

    self._partialFraction = kwargs.get('partialFraction', 0.1)
    return
  
  def _compile(self):
    if not self.trainable: return
    self._optimizer = tf.optimizers.Adam(learning_rate=1e-4)

    # CDR = tf.keras.optimizers.schedules.CosineDecayRestarts(
    #   initial_learning_rate=1e-4, first_decay_steps=10000, t_mul=2.0, m_mul=0.5, alpha=0.01
    # )
    # self._optimizer = tf.optimizers.Adam(learning_rate=CDR)
    return
  
  @property
  def learning_rate(self):
    return 0.

  @tf.function(
    input_signature=[(
      {
        'clean': GMMInputSpec(),
        'augmented': GMMInputSpec(),
      },
      (
        tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32),
      )
    )]
  )
  def _trainStep(self, data):
    print('Instantiate _trainStep')
    models = [self._face2latent, self._latent2GMM]
    if self._useDiscriminator:
      models.append(self._discriminator)

    TV = sum([x.trainable_variables for x in models], [])

    x, (y, ) = data
    B = tf.shape(y)[0]
    N = tf.shape(y)[1]
    KP = tf.shape(y)[2]
    # (B, N, KP, 2) => (-1, KP, 2) => (B*N, KP, 2)
    y = tf.reshape(y, tf.concat(((-1,), tf.shape(y)[-2:]), axis=-1))
    mainPt = y[:, 0, :]
    tf.assert_equal(tf.shape(mainPt), (B * N, 2))
    #######################
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(TV)

      losses = defaultdict(list)
      extra = 0.0
      for sourceId in ['augmented']: # ['clean', 'augmented']
        latents = self._face2latent(x[sourceId], training=True)
        extra += sum(map(tf.reduce_mean, self._face2latent.losses))
        ###########################
        latent = latents['latent']
        latent = tf.reshape(latent, (B * N, tf.shape(latent)[-1]))
        if 0.0 < self._partialFraction:
          partial = latents['partial']
          partial = tf.reshape(partial, (B * N, tf.shape(partial)[-1]))
          # take with probability self._partialFraction from partial
          latent = tf.where(
            tf.random.uniform((B * N, 1)) < self._partialFraction,
            partial,
            latent
          )
          pass
        tf.assert_equal(tf.shape(latent)[:1], (B * N,))
        ##################################
        distrRaw, distr = self._latent2GMM(
          latent,
          shift=tf.reshape(latents['shift'], (-1, 2)),
          training=True,
        )
        extra += sum(map(tf.reduce_mean, self._latent2GMM.losses))
        distrDetached = self._latent2GMM.distribution(distrRaw, nograds=True)
        ################################## 
        # losses['GMM mean'].append(tf.losses.mae(mainPt, distr.mean()))
        # maximize distr.log_prob(mainPt)
        realL = tf.nn.sigmoid(-distr.log_prob(mainPt))
        tf.debugging.assert_all_finite(realL, 'realL')
        tf.assert_equal(tf.shape(realL), (B * N,))
        realL = tf.reduce_mean(realL)
        losses['GMM real'].append(realL)
        # scale additional losses by realL
        GScale = tf.stop_gradient(realL * (1.0 - realL))
        GScale = tf.clip_by_value(GScale, 0.0, 1.0)
        ################################## 
        if self._useDiscriminator:
          discriminator = self._discriminator(
            latent,
            realPoints=mainPt,
            distributionPredicted=distr,
            distributionPredictedDetached=distrDetached,
            training=True
          )
          extra += sum(map(tf.reduce_mean, self._discriminator.losses))
          for k, v in discriminator['losses'].items():
            losses['D ' + k].append(tf.reduce_mean(v))

          losses['G'].append(tf.reduce_mean(discriminator['generator loss']) * GScale)
          continue

        if 0.0 < self._L1Reg:
          losses['L1'].append(tf.reduce_mean(tf.abs(latent)) * GScale * self._L1Reg)
        continue
      
      reduceLoss = lambda x: sum(x, 0.0) / max((1, len(x)))
      losses = {k: reduceLoss(v) for k, v in losses.items()}
      losses['extra'] = extra
      losses['total'] = loss = sum(losses.values(), 0.0)
      pass
    self._optimizer.minimize(loss, TV, tape=tape)
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
    # input_signature=[GMMInputSpec(),]
  )
  def __call__(self, data=None, latent=None, training=False):
    if latent is None:
      f2l = self._face2latent(data, training=training)
      latent, latent2 = f2l['latent'], f2l['partial']
      # calculate only for the last point
      latent = latent[:, -1, :]
   
    if True:
      gmm, distribution = self._latent2GMM(latent, training=training)
    else:
      B = tf.shape(latent)[0]
      # latent = tf.concat((latent, latent2), axis=1)
      N = tf.shape(latent)[1]
      latent = tf.reshape(latent, (B * N, tf.shape(latent)[-1]))
      
      gmm, distribution = self._latent2GMM(latent[::], training=training)
      G = tf.shape(gmm['mu'])[1]
      # combine GMMs
      weights = tf.ones(N, dtype=tf.float32)
      weights = tf.nn.softmax(weights)
      weights = tf.reshape(weights, (N, 1))
      weights = tf.tile(weights, (B, 1))
      tf.assert_equal(tf.shape(weights), (B * N, 1))

      # first, compute the weights
      alpha = gmm['alpha'] * weights
      alpha = tf.reshape(alpha, (B, N * G))
      # then, combine the mu
      mu = tf.reshape(gmm['mu'], (B, N * G, 2))
      # finally, combine the scale_tril
      scale_tril = tf.reshape(gmm['scale_tril'], (B, N * G, 2, 2))
      # make distribution
      tfd = tfp.distributions
      distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril)
      )
      pass
    # Dmu = self._discriminator(latent, training=False)
    return {
      # 'D mu': Dmu,
      'distribution': distribution,
      'gmm': gmm,
      'latent': latent
    }
    
  def save(self, folder, postfix=''):
    if postfix: postfix = '-' + postfix
    self._face2latent.save_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'F2L', postfix)))
    self._latent2GMM.save_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'GMM', postfix)))
    if self._useDiscriminator:
      self._discriminator.save_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'D', postfix)))
    return
    
  def load(self, folder, postfix=''):
    if postfix: postfix = '-' + postfix
    self._face2latent.load_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'F2L', postfix)))
    self._latent2GMM.load_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'GMM', postfix)))
    if self._useDiscriminator:
      self._discriminator.load_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'D', postfix)))
    
    self._compile()
    return

  @tf.function
  def _eval(self, x, y):
    y = y[:, -1, 0]
    latents = self._face2latent(x, training=False)
    gmm, distr = self._latent2GMM(
      latents['latent'][:, -1],
      shift=tf.reshape(latents['shift'][:, -1], (-1, 2)),
      training=False,
    )

    loss = tf.nn.sigmoid(-distr.log_prob(y))
    tf.assert_equal(tf.shape(loss), tf.shape(y)[:1])
    sampled = distr.sample()
    tf.assert_equal(tf.shape(sampled), tf.shape(y))
    _, dist = NNU.normVec(distr.mean() - y)
    return loss, sampled, dist

  def eval(self, data):
    x, (y, ) = data
    loss, sampled, dist = self._eval(x,y)    
    return loss.numpy(), sampled.numpy(), dist.numpy()

  @property
  def timesteps(self):
    s = self._face2latent.inputs[0].shape # face mesh
    if 3 == len(s): return None
    return s[1]

  def extended(self, X):
    N = X['time'].shape[1]
    indices = np.arange(N)[::-1]
    t = self.timesteps
    perm = [
      indices[::S][:t][::-1]
      for S in range(1, N + 1)
      if N//S >= t
    ]
    # add [m-t, m-t+1, ..., m-1, last]
    for m in range(t, len(indices) - 1):
      seq = [m - i for i in range(t)]
      perm.append(seq[::-1] + [len(indices) - 1])
    
    perm = np.array(perm)
    # for i in range(perm.shape[0]):
    #   print(perm[i])
    # print('total', perm.shape[0])
    perm = np.arange(N).reshape(1, N)
    perm = perm[:1]
    # use perm indices to create new X dict
    XNew = {}
    for k in ['time']:
      XNew[k] = np.array([X[k][0, p] for p in perm])

    # fix 'time' field of each perm, so that it is always sorted from 0. Use np.diff
    XNew['time'] = np.diff(XNew['time'][..., 0], axis=1)
    XNew['time'] = np.concatenate((
      np.zeros((XNew['time'].shape[0], 1)),
      XNew['time']
    ), axis=1)[..., None]
    
    X['time'] = XNew['time']
    return self._extended(X, perm, XNew['time'])

  @tf.function(
    input_signature=[
      GMMInputSpec(),
      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
      tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
    ]
  )
  def _extended(self, X, indices, time):
    B = tf.shape(indices)[0]
    X = {k: v for k, v in X.items() if k != 'time'}
    stepsData = self._face2step(X, training=False)['latent']
    tf.assert_equal(tf.shape(stepsData)[:1], (1,))
    steps = tf.shape(time)[1]
    stepsData = tf.gather(stepsData[0], tf.reshape(indices, (-1,)), axis=0)
    stepsData = tf.reshape(stepsData, (B, steps, -1))

    tf.assert_equal(tf.shape(stepsData)[:2], tf.shape(indices))
    
    f2l = self._step2latent([stepsData, time], training=False)
    latent = f2l['latent'][:, -1:]
    
    B = tf.shape(latent)[0]
    latent = tf.reshape(latent, (B, tf.shape(latent)[-1]))
    
    gmm, _ = self._latent2GMM(latent, training=False)
    G = tf.shape(gmm['mu'])[1]
    # combine GMMs
    weights = tf.ones(B, dtype=tf.float32)
    weights = tf.nn.softmax(weights)
    weights = tf.reshape(weights, (B, 1))
    tf.assert_equal(tf.shape(weights), (B, 1))

    # first, compute the weights
    alpha = gmm['alpha'] * weights
    alpha = tf.reshape(alpha, (1, B * G))
    # then, combine the mu
    mu = tf.reshape(gmm['mu'], (1, B * G, 2))
    # finally, combine the scale_tril
    scale_tril = tf.reshape(gmm['scale_tril'], (1, B * G, 2, 2))
    # make distribution
    tfd = tfp.distributions
    distribution = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(probs=alpha),
      components_distribution=tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril)
    )
      
    # Dmu = self._discriminator(latent, training=False)
    return {
      # 'D mu': Dmu,
      'distribution': distribution,
      'gmm': None,
      'latent': latent
    }
    
  def scorePoints(self, latents, points):
    return self._discriminator.scorePoints(latents, points)

  def fit_embeddings(self, data):
    x, (y, ) = data
    B = tf.shape(y)[0]
    N = tf.shape(y)[1]
    KP = tf.shape(y)[2]
    # (B, N, KP, 2) => (-1, KP, 2) => (B*N, KP, 2)
    y = tf.reshape(y, tf.concat(((-1,), tf.shape(y)[-2:]), axis=-1))
    mainPt = y[:, 0, :]
    tf.assert_equal(tf.shape(mainPt), (B * N, 2))
    #######################
    losses = defaultdict(list)
    extra = 0.0
  
    latents = self._face2latent(x, training=False)
    extra += sum(map(tf.reduce_mean, self._face2latent.losses))
    ###########################
    latent = latents['latent']
    latent = tf.reshape(latent, (B * N, tf.shape(latent)[-1]))
    
    distrRaw, distr = self._latent2GMM(
      latent,
      shift=tf.reshape(latents['shift'], (-1, 2)),
      training=False,
    )
    extra += sum(map(tf.reduce_mean, self._latent2GMM.losses))
    distrDetached = self._latent2GMM.distribution(distrRaw, nograds=True)
    ################################## 
    # losses['GMM mean'].append(tf.losses.mae(mainPt, distr.mean()))
    # maximize distr.log_prob(mainPt)
    realL = tf.nn.sigmoid(-distr.log_prob(mainPt))
    tf.debugging.assert_all_finite(realL, 'realL')
    tf.assert_equal(tf.shape(realL), (B * N,))
    realL = tf.reduce_mean(realL)
    losses['GMM real'].append(realL)
    
    # scale additional losses by realL
    GScale = tf.stop_gradient(realL * (1.0 - realL))
    GScale = tf.clip_by_value(GScale, 0.0, 1.0)
    ################################## 
    if self._useDiscriminator:
      discriminator = self._discriminator(
        latent,
        realPoints=mainPt,
        distributionPredicted=distr,
        distributionPredictedDetached=distrDetached,
        training=False
      )
      extra += sum(map(tf.reduce_mean, self._discriminator.losses))
      for k, v in discriminator['losses'].items():
        losses['D ' + k].append(tf.reduce_mean(v))

      losses['G'].append(tf.reduce_mean(discriminator['generator loss']) * GScale)
      pass

    reduceLoss = lambda x: sum(x, 0.0) / max((1, len(x)))
    losses = {k: reduceLoss(v) for k, v in losses.items()}
    losses['extra'] = extra
    losses['total'] = loss = sum(losses.values(), 0.0)
    return loss