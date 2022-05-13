import tensorflow as tf
import tensorflow.keras.layers as L

def decodeSeriesTF(x, base=2.0):
  N = tf.shape(x)[-1]
  coefs = tf.pow(base, tf.cast(tf.range(N), tf.float32))
  return tf.reduce_sum(x / coefs, axis=-1)

class CDecodeSeries(tf.keras.layers.Layer):
  def __init__(self, base=2.0, **kwargs):
    super().__init__(**kwargs)
    self._base = base
    return
  
  def call(self, x):
    return decodeSeriesTF(x, self._base)
############################################
def sMLP(shape, sizes, activation='linear'):
  data = L.Input(shape)
  
  res = data
  for s in sizes:
    res = L.Dense(s, activation=activation)(
      L.Dropout(0.05)(res)
    )
    continue
    
  return tf.keras.Model(
    inputs=[data],
    outputs=[res]
  )