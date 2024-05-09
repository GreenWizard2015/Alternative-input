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
from .CCoordsEncodingLayer import CCoordsEncodingLayer
class CConvPE(tf.keras.layers.Layer):
  def __init__(self, channels=32, activation=None, **kwargs):
    super().__init__(**kwargs)
    self._channels = channels
    self._coords2pe = CCoordsEncodingLayer(N=channels, sharedTransformation=True)
    self._activation = tf.keras.activations.get(activation)
    return
  
  def _makeGrid(self, H, W):
    HRange = tf.linspace(-1.0, 1.0, H)
    WRange = tf.linspace(-1.0, 1.0, W)
    coords = tf.meshgrid(HRange, WRange, indexing='ij')
    coords = tf.stack(coords, axis=-1)
    coords = tf.reshape(coords, [1, H * W, 2])
    return coords
  
  def _PEFor(self, H, W):
    coords = self._makeGrid(H, W)
    coords = self._coords2pe(coords)
    coords = tf.reshape(coords, [1, H, W, coords.shape[-1]])
    coords = self._activation(coords)
    return coords
  
  def call(self, x):
    B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    pe = self._PEFor(H, W)
    pe = tf.repeat(pe, B, axis=0)
    tf.assert_equal(tf.shape(pe)[:-1], tf.shape(x)[:-1])
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
      d = 1e-4
      quantized = x + tf.random.truncated_normal(tf.shape(x), -0.5 + d, 0.5 - d)
      pass

    quantized = tf.floor(quantized)
    quantized = 0.5 + tf.clip_by_value(quantized, self._minValue, self._maxValue)
    return x + tf.stop_gradient(quantized - x)
####################################
class CResidualMultiplicativeLayer(tf.keras.layers.Layer):
  def __init__(self, eps=1e-8, headsN=1, **kwargs):
    super().__init__(**kwargs)
    self._eps = eps
    self._scale = tf.Variable(
      initial_value=tf.random.normal((1, ), mean=0.0, stddev=0.1),
      trainable=True, dtype=tf.float32,
      name=self.name + '/_scale'
    )
    self._headsN = headsN
    self._normalization = None
    return
  
  @property
  def scale(self): return tf.nn.sigmoid(self._scale) * (1.0 - 2.0 * self._eps) + self._eps # [eps, 1 - eps]
  
  def _SMNormalization(self, xhat):
    xhat = tf.nn.softmax(xhat, axis=-1)
    xhat = xhat - tf.reduce_mean(xhat, axis=-1, keepdims=True)
    rng = tf.reduce_max(tf.abs(xhat), axis=-1, keepdims=True)
    return 1.0 + tf.math.divide_no_nan(xhat, rng * self.scale) # [1 - scale, 1 + scale]
  
  def _HeadwiseNormalizationNoPadding(self, xhat):
    shape = tf.shape(xhat)
    # reshape [B, ..., N * headsN] -> [B, ..., headsN, N], apply normalization, reshape back
    xhat = tf.reshape(xhat, tf.concat([shape[:-1], [self._headsN, shape[-1] // self._headsN]], axis=-1))
    xhat = self._SMNormalization(xhat)
    xhat = tf.reshape(xhat, shape)
    return xhat
  
  def _HeadwiseNormalizationPadded(self, lastChunk):  
    def F(xhat):
      mainPart = self._HeadwiseNormalizationNoPadding(xhat[..., :-lastChunk])
      tailPart = self._SMNormalization(xhat[..., -lastChunk:])
      return tf.concat([mainPart, tailPart], axis=-1)
    return F
  
  def build(self, input_shapes):
    _, xhatShape = input_shapes
    self._normalization = self._SMNormalization
    if 1 < self._headsN:
      assert 1 < (xhatShape[-1] // self._headsN), "too few channels for headsN"

      lastChunk = xhatShape[-1] % self._headsN
      self._normalization = self._HeadwiseNormalizationPadded(lastChunk) if 0 < lastChunk else self._HeadwiseNormalizationNoPadding
      pass
    return super().build(input_shapes)
  
  def call(self, x):
    x, xhat = x
    # return (tf.nn.relu(x) + self._eps) * (self._normalization(xhat) + self._eps) # more general/stable version
    # with SM normalization, relu and addition are redundant
    return x * self._normalization(xhat)
####################################
class CRMLBlock(tf.keras.Model):
  def __init__(self, mlp=None, RML=None, **kwargs):
    super().__init__(**kwargs)
    if mlp is None: mlp = lambda x: x
    self._mlp = mlp
    if RML is None: RML = CResidualMultiplicativeLayer()
    self._RML = RML
    return
  
  def build(self, input_shapes):
    xShape = input_shapes[0]
    self._lastDense = L.Dense(xShape[-1], activation='relu', name='%s/LastDense' % self.name)
    return super().build(input_shapes)
  
  def call(self, x):
    assert isinstance(x, list), "expected list of inputs"
    xhat = tf.concat(x, axis=-1)
    xhat = self._mlp(xhat)
    xhat = self._lastDense(xhat)
    x0 = x[0]
    return self._RML([x0, xhat])
####################################
# Hacky way to provide same optimizer for all models
def createOptimizer(config=None):
  if config is None:
    config = {
      'learning_rate': 1e-4,
      'weight_decay': 1e-1,
      'exclude_from_weight_decay': [
        'batch_normalization', 'bias',
        'CEL_', # exclude CCoordsEncodingLayer from weight decay
        '_gate', '_PE', '_scale', # exclude some custom layers variables
      ],
    }
    pass

  optimizer = tf.optimizers.AdamW(
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay'],
  )
  var_names = config.get('exclude_from_weight_decay', None)
  if var_names is not None:
    optimizer.exclude_from_weight_decay(var_names=var_names)
  return optimizer