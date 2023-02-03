import tensorflow as tf
import time
import os
from NN.MIModel import MIModel
import NN.Utils as NNU
import NN.networks as networks

GRID_HW = 300
GRID_MIN = -0.0
GRID_MAX = 1.0
def makeGrid():
  import numpy as np
  xy = np.linspace(GRID_MIN, GRID_MAX, GRID_HW)
  XY = np.meshgrid(xy, xy)
  res = np.concatenate([x.reshape(-1, 1) for x in XY], axis=-1)
  return tf.constant(res, tf.float32)
gridTF = makeGrid()

def distr2image(distr):
  lp = distr.log_prob(gridTF)
  lp = tf.floor(lp)
  lp = tf.maximum(0., lp)
  lp = tf.cast(lp * 20, tf.uint8)
  lp = tf.reshape(lp, (GRID_HW, GRID_HW, 1))
  lp = tf.concat([lp]*3, -1)
  return lp // 2 + 200

class CCoreModel:
  def __init__(self, GMM, model='simple', **kwargs):
    self._modelID = model
    
    self._GMM = GMM
    self._MI = MIModel()
    _ = self._MI(
      tf.zeros((1, 64), tf.float32), 
      y=tf.zeros((1, 2), tf.float32),
      #y=[tf.zeros((1, 2), tf.float32), tf.zeros((1, 2), tf.float32)],
    )
    self._sampler = networks.simpleModel(FACE_LATENT_SIZE=64)

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
    self._optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    self._sampler.compile(tf.optimizers.Adam(learning_rate=1e-4))
    return
  
  @tf.function(
    input_signature=[(
      {
        'latent': tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        'pos': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        'goal': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        'prev': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
      },
      {
        'latent': tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        'pos': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        'goal': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        'prev': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
      },
    )]
  )
  def _trainStep(self, Data):
    print('Instantiate _trainStep')
    ###############
    data = Data[0]
    losses = {}
    TV = self._MI.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(TV)
      MI = self._MI(
        x=data['latent'],
        y=data['pos'],
        # y=[data['prev'], data['pos']],
        training=True,
      )
      loss = -tf.reduce_mean(MI)
      losses['MI'] = loss
  
    self._optimizer.minimize(loss, TV, tape=tape)
    ###############
    data = Data[1]
    latent = data['latent']
    distr = self._GMM(latent=latent)['distribution']
    TV = self._sampler.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(TV)
      sampled = self._sampler([latent, distr.mean()], training=True)['coords']
      # main loss
      _, GLoss = NNU.normVec(data['goal'] - sampled)
      losses['GLoss'] = GLoss = tf.reduce_mean(GLoss)
      # extra losses
      MILoss = self._MI(x=latent, y=sampled, training=False)
      extraLosses = {
        # 'MILoss': -MILoss, # tf.exp(-MILoss),
        'distrLoss': tf.nn.sigmoid( distr.log_prob(sampled) ),
        'disc': self._GMM.scorePoints(latent, sampled),
      }
      # scale by GLoss
      scale = tf.stop_gradient(GLoss * (1.0 - GLoss))
      scale = tf.clip_by_value(scale, 0.0, 1.0)
      loss = 0.0
      for k, v in extraLosses.items():
        v = tf.reduce_mean(v) * scale
        losses[k] = v
        loss += v
        continue
      loss = loss + GLoss
      losses['sampler'] = loss
 
    self._sampler.optimizer.minimize(loss, TV, tape=tape)
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
    gmm = self._GMM(data)
    mean = gmm['distribution'].mean()
    latent = gmm['latent']
    points = self._sampler([latent, mean], training=False)['coords']
    return {
      'coords': points[0].numpy(),
      'sampled': points[0].numpy(),
      'density': distr2image(gmm['distribution']),
      **gmm
    }

  def save(self, folder, postfix=''):
    if postfix: postfix = '-' + postfix
#     self._MI.save_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'MI', postfix)))
    self._sampler.save_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'sampler', postfix)))
    return
    
  def load(self, folder, postfix=''):
    if postfix: postfix = '-' + postfix
#     self._MI.load_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'MI', postfix)))
    self._sampler.load_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'sampler', postfix)))
    self._compile()
    return
  
  def eval(self, data):
    x, (y, ) = data
    raise Exception('Unknown model (%s)' % (self._modelID, ))

  @property
  def timesteps(self):
    return self._GMM.timesteps
  
  def extended(self, data):
    gmm = self._GMM.extended(data)
    mean = gmm['distribution'].mean()
    latent = gmm['latent']
    points = mean# self._sampler([latent, mean], training=False)['coords']
    return {
      'coords': points[0].numpy(),
      'sampled': points[0].numpy(),
      'density': distr2image(gmm['distribution']),
      **gmm
    }