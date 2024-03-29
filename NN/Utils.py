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
  def __init__(self, sizes, activation='linear', dropout=0.01, **kwargs):
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
  def __init__(self, N, activation='linear', base=2.0, P=None, **kwargs):
    super().__init__(**kwargs)
    self._dense = L.Dense(2 * N, activation=activation)
    
    P = -1.0 * np.arange(N) if P is None else P
    self._decode = CDecodeSeries(base=base, powers=P)
    return
  
  def call(self, x):
    x = self._dense(x)
    x = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [2, -1]], axis=-1))
    return self._decode(x)
############################################
class CConvPE(tf.keras.layers.Layer):
  def __init__(self, channels=1, activation=None, **kwargs):
    super().__init__(**kwargs)
    self._channels = channels
    self._activation = tf.keras.activations.get(activation)
    return
  
  def build(self, s):
    super().build(s)
    self._PE = tf.Variable(
      initial_value=tf.zeros((1, *s[1:-1], self._channels), dtype="float32"),
      trainable=True, dtype="float32",
      name=self.name + '/_PE'
    )
    return
  
  def call(self, x):
    B = tf.shape(x)[0]
    pe = tf.repeat(self._activation(self._PE), B, axis=0)
    return tf.concat([x, pe], axis=-1)
####################################
class CStackApplySplit(tf.keras.layers.Layer):
  def __init__(self, F, **kwargs):
    super().__init__(**kwargs)
    self._F = F
    return

  def call(self, *data):
    B = [tf.shape(x)[0] for x in data]
    x = self._F(tf.concat(data, axis=0))
    return tf.split(x, B, axis=0)
####################################
def normVec(x):
  V, L = tf.linalg.normalize(x, axis=-1)
  V = tf.where(tf.math.is_nan(V), 0.0, V)
  return(V, L)
####################################
class CParallelDense(L.Layer):
  def __init__(self, units, N, stepwise=False, activation=None, **kwargs):
    super().__init__(**kwargs)
    self._size = units
    self._N = N
    self._stepwise = stepwise
    self._activation = tf.keras.activations.get(activation)
    return
  
  def build(self, input_shape):
    M = input_shape[-1]
    P = [self._N]
    N = self._N
    if self._stepwise:
      assert 2 < len(input_shape), "input_shape must be at least 3D for stepwise parallel dense"
      P = [input_shape[-2], self._N]
      N = input_shape[-2] * self._N
      pass
    
    self._kernel = self.add_weight(
      shape=P + [M, self._size],
      initializer='glorot_uniform',
      trainable=True,
      name='kernel'
    )
    # bias
    self._bias = self.add_weight(
      shape=[N, self._size],
      initializer='glorot_uniform',
      trainable=True,
      name='bias'
    )
    return

  def call(self, x):
    xs = tf.shape(x)
    N = self._N
    axes = -1
    if self._stepwise:
      N = xs[-2] * self._N
      axes = -2
      
      x = tf.einsum('...jm,jzms->...jzs', x, self._kernel)
      # ...jzs -> ...(jz)
      x = tf.reshape(x, tf.concat([xs[:-2], [N, self._size]], axis=-1))
    else:
      x = tf.einsum('...m,nms->...ns', x, self._kernel)
    tf.assert_equal(tf.shape(x)[-2:], (N, self._size))
    x = x + tf.reshape(self._bias, tf.shape(x[:1]))
    
    tf.assert_equal(tf.shape(x)[-2:], (N, self._size))
    tf.assert_equal(tf.shape(x)[:-2], xs[:axes])
    return self._activation(x)
####################################
# Custom layer that quantizes the input per each value
class CQuantizeLayer(tf.keras.layers.Layer):
  def __init__(self, minValue=0, maxValue=256, **kwargs):
    super().__init__(**kwargs)
    self._minValue = float(minValue)
    self._maxValue = float(maxValue)
    return

  def call(self, x, training=None):
    quantized = x
    if training:
      quantized = x + tf.random.truncated_normal(tf.shape(x), 0.0, 0.1)
      pass

    quantized = tf.round(quantized)
    quantized = 0.5 + tf.clip_by_value(quantized, self._minValue, self._maxValue)
    return x + tf.stop_gradient(quantized - x)
####################################
# causal self-attention transformer
class CMyTransformerLayer(tf.keras.layers.Layer):
  def __init__(self,
    latentSize, num_heads=8,
    toQuery=None, toKey=None,
    useNormalization=False, 
    **kwargs
  ):
    super().__init__(**kwargs)
    self._layerNorm = tf.keras.layers.LayerNormalization() if useNormalization else lambda x: x
    self._mha = tf.keras.layers.MultiHeadAttention(
      num_heads=num_heads, dropout=0.01, key_dim=latentSize
    )
    self._toQuery = toQuery if toQuery is not None else lambda x: x
    self._toKey = toKey if toKey is not None else lambda x: x
    return

  def call(self, x):
    attention = self._mha(
      query=self._toQuery(x),
      key=self._toKey(x),
      value=x,
      use_causal_mask=True
    )
    return self._layerNorm(x + attention)