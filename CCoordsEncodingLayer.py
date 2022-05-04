import tensorflow as tf

class CCoordsEncodingLayer(tf.keras.layers.Layer):
  def __init__(self, N, raw=True, cumsum=False, **kwargs):
    super().__init__(**kwargs)
    self._N = N
    self._raw = raw
    self._cumsum = cumsum
    
    self._coefs = tf.Variable(
      initial_value=tf.random_normal_initializer()(shape=(1, 1, 1, N), dtype="float32"),
#       initial_value=tf.linspace(0.0, 10.0, N)[None, None, None],
      trainable=True, dtype="float32",
      name='_coefs'
    )
    
    self._shifts = tf.Variable(
      initial_value=tf.random_normal_initializer()(shape=(1, 1, 1, N), dtype="float32"),
      trainable=True, dtype="float32",
      name='_shifts'
    )
    return
  
  def call(self, x):
    freq = self.coefs
    shifts = self.shifts
    
    tX = (x[..., None] * freq) + shifts
    sinX = tf.sin(tX)
    cosX = tf.cos(tX)
  
    res = sinX - cosX
    res = res[..., 0, :] - res[..., 1, :]
    
    if self._raw:
      res = tf.concat([x, res], axis=-1)
    return res
  
  @property
  def coefs(self):
    if self._cumsum:
      return tf.math.cumsum(1e-8 + tf.nn.relu(self._coefs), axis=-1)
    return tf.nn.relu(self._coefs)
  
  @property
  def shifts(self):
    return self._shifts