import tensorflow as tf
import math

# Learnable encoding of coordinates
class CCoordsEncodingLayer(tf.keras.layers.Layer):
  def __init__(self, 
    N, raw=True, useShifts=False,
    scaling='pow', # 'pow' or 'linear'
    maxFrequency=1e+4,
    useLowBands=True, useHighBands=True,
    finalDropout=0.0, bandsDropout=False,
    sharedTransformation=False,
    **kwargs
  ):
    super().__init__(**kwargs)
    self._N = N
    self._raw = raw
    self._sharedTransformation = sharedTransformation

    if useShifts:
      self._shifts = tf.Variable(
        initial_value=tf.random_normal_initializer()(shape=(1, 1, 1, N), dtype="float32"),
        trainable=True, dtype="float32",
        name=self.name + '/_shifts'
      )
    else:
      self._shifts = tf.constant(0.0, dtype=tf.float32)

    maxN = 1 + N // 2 if useHighBands and useLowBands else N
    freq = self._createBands(scaling, maxFrequency, maxN)
    bands = []
    if useLowBands: bands.append(1.0 / freq[::-1])
    if useHighBands: bands.append(freq)
    self._baseFreq = tf.concat([(x[1:] + x[:-1])[:maxN] / 2.0 for x in bands], axis=-1)
    self._freqRange = tf.concat([(x[1:] - x[:-1])[:maxN] / 2.0 for x in bands], axis=-1)
    self._freqDeltas = tf.Variable(
      initial_value=tf.random_normal_initializer()(shape=(N,), dtype="float32"),
      trainable=True, dtype="float32",
      name=self.name + '/_freqDeltas'
    )

    self._dropout = lambda x, **_: x
    if 0.0 < finalDropout:
      if bandsDropout:
        self._dropout = self._bandsDropoutMake(finalDropout)
      else:
        self._dropout = tf.keras.layers.Dropout(finalDropout)
    return
  
  def build(self, input_shape):
    M = input_shape[1]
    shapePrefix = [1, ] * (len(input_shape) - 2)
    P = 1 if self._sharedTransformation else self._N
    F = self._transform(tf.zeros([1, M, input_shape[-1]])).shape[-1]
    
    self._fussionW = tf.Variable(
      initial_value=tf.random_normal_initializer()(shape=shapePrefix+[P, F], dtype="float32"),
      trainable=True, dtype="float32",
      name=self.name + '/_fussionW'
    )
    self._fussionB = tf.Variable(
      initial_value=tf.random_normal_initializer()(shape=shapePrefix+[P,], dtype="float32"),
      trainable=True, dtype="float32",
      name=self.name + '/_fussionB'
    )
    
    self._gates = tf.Variable(
      initial_value=tf.zeros(shape=(1, 1, self._N), dtype="float32"),
      trainable=True, dtype="float32",
      name=self.name + '/_gates'
    )
    return

  def _fussion(self, x):
    res = tf.reduce_sum(x * self._fussionW, axis=-1) + self._fussionB
    return res
  
  def _transform(self, x):
    B, M, P, N = tf.shape(x)[0], x.shape[1], x.shape[2], self._N
    data = []
    tX = (x[..., None] * self.coefs) + self.shifts # (B, M, 2, N)
    tX = tX * (2.0 * math.pi)
    tX = tf.transpose(tX, (0, 1, 3, 2))[..., None] # (B, M, N, 2, 1)
    for F in [tf.sin, tf.cos]:
      fX = F(tX)
      data.append(fX)
      
      #fXfX = tf.square(fX)
      #data.append(fXfX)
      continue
    
    res = tf.concat(data, axis=-1)
    tf.assert_equal(tf.shape(res)[:-1], (B, M, N, P))
    return tf.reshape(res, (B, M, N, res.shape[-1] * P))
  
  def call(self, x, training=None):
    # x is (B, M, P)
    # output is (B, M, N)
    B, M, P, N = tf.shape(x)[0], x.shape[1], x.shape[2], self._N

    transformed = self._transform(x) # (B, M, N, P * F)
    res = self._fussion(transformed) * self.gates # (B, M, N)
    tf.assert_equal(tf.shape(res), (B, M, N))
    res = self._dropout(res, training=training)
    
    if self._raw:
      res = tf.concat([x, res], axis=-1)
    return res
  
  @property
  def coefs(self):
    coefs = self._baseFreq + tf.nn.tanh(self._freqDeltas) * self._freqRange
    return coefs[None, None, None]

  @property
  def shifts(self):
    return self._shifts
  
  @property
  def gates(self):
    return tf.nn.tanh(self._gates)
  
  def _bandsDropoutMake(self, maxRate):
    def apply(x):   
      normed = tf.abs(self.gates)
      normed = tf.math.divide_no_nan(normed, tf.reduce_max(normed, axis=-1, keepdims=True))
      normed = (1.0 - normed) * maxRate
      noise = tf.random.uniform(tf.shape(x), minval=0.0, maxval=1.0)
   
      mask = tf.cast(normed < noise, x.dtype) / (1.0 - normed)
      return x * tf.stop_gradient(mask)
    
    def F(x, training):
      training = tf.keras.backend.learning_phase()
      training = tf.cast(training, tf.bool)
      return tf.cond(training, lambda: apply(x), lambda: tf.identity(x))
    return F
  
  def _createBands(self, scaling, maxFrequency, N):
    if 'pow' == scaling:
      if maxFrequency is None:
        maxFrequency = 2.0 ** min((N, 32))
      base = math.pow(maxFrequency, 1.0 / float(N))
      return tf.pow(base, tf.cast(tf.range(N), tf.float32))
    
    if 'linear' == scaling:
      maxFrequency = N if maxFrequency is None else maxFrequency
      return tf.linspace(1.0, float(maxFrequency), N)
    return

