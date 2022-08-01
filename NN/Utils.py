import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np

class CDecodeSeries(tf.keras.layers.Layer):
  def __init__(self, base=2.0, powers=None, **kwargs):
    super().__init__(**kwargs)
    self._base = tf.Variable(
      initial_value=float(base),
      trainable=False, dtype="float32",
      name=self.name + '/_base'
    )
    self._powers = None
    if not(powers is None):
      self._powers = tf.Variable(
        initial_value=powers,
        trainable=False, dtype="float32",
        name=self.name + '/_powers'
      )
    return
  
  def call(self, x):
    powers = self._powers
    if powers is None:
      N = tf.shape(x)[-1]
      powers = tf.cast(tf.range(N) - N // 2, tf.float32)
    coefs = tf.pow(self._base, powers)
    return tf.reduce_sum(x * coefs, axis=-1)
############################################
class sMLP(tf.keras.layers.Layer):
  def __init__(self, sizes, activation='linear', dropout=0.05, **kwargs):
    super().__init__(**kwargs)
    layers = []
    for i, sz in enumerate(sizes):
      if 0.0 < dropout:
        layers.append(L.Dropout(dropout, name='%s/dropout-%i' % (self.name, i)))
      layers.append(L.Dense(sz, activation=activation, name='%s/dense-%i' % (self.name, i)))
      continue
    self._F = tf.keras.Sequential(layers, name=self.name + '/F')
    return
  
  def call(self, x, **kwargs):
    return self._F(x, **kwargs)
############################################
class CRolloutTimesteps(tf.keras.layers.Layer):
  def __init__(self, F, **kwargs):
    super().__init__(**kwargs)
    self._F = F
    return
  
  def _reshapeAll(self, x, shapePrefix, axis):
    if tf.is_tensor(x):
      return tf.reshape(x, tf.concat([shapePrefix, x.shape[axis:]], axis=-1))

    if isinstance(x, list):
      return [tf.reshape(v, tf.concat([shapePrefix, v.shape[axis:]], axis=-1)) for v in x]

    if isinstance(x, dict):
      return {k: tf.reshape(v, tf.concat([shapePrefix, v.shape[axis:]], axis=-1)) for k, v in x.items()}
    return x
    
  def call(self, x, **kwargs):
    xi = x if tf.is_tensor(x) else x[0]
    B = tf.shape(xi)[0]
    steps = xi.shape[1]
    if steps is None:
      steps = tf.shape(xi)[1]

    rolloutX = self._reshapeAll(x, (B * steps,), axis=2)
    res = self._F(rolloutX, **kwargs)
    return self._reshapeAll(res, (B, steps), axis=1)
############################################
class CGate(tf.keras.layers.Layer):
  def __init__(self, axis=[], **kwargs):
    super().__init__(**kwargs)
    self._axis = axis if isinstance(axis, list) else [axis]
    return
  
  def build(self, input_shape):
    gateShape = [1 for _ in input_shape]
    for i in self._axis:
      gateShape[i] = input_shape[i]

    self._gate = tf.Variable(
      initial_value=tf.zeros(gateShape),
      trainable=True, dtype="float32",
      name=self.name + '/_gate'
    )
    return
  
  def call(self, x):
    return tf.nn.tanh(self._gate) * x
############################################
class CDecodePoint(tf.keras.layers.Layer):
  def __init__(self, N, activation='linear', **kwargs):
    super().__init__(**kwargs)
    self._dense = L.Dense(2 * N, activation=activation)
    
    P = -1.0 * np.arange(N)
    self._decode = CDecodeSeries(base=2.0, powers=P)
    return
  
  def call(self, x):
    x = self._dense(x)
    x = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [2, -1]], axis=-1))
    return self._decode(x)
############################################
def KLD_approx(p, q, num_draws=1e2):
  s = p.sample(num_draws)
  diff = p.log_prob(s) - q.log_prob(s)
  return tf.reduce_mean(diff, axis=0)
