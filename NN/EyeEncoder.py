import tensorflow as tf
import tensorflow.keras.layers as L

import numpy as np
from NN.Utils import sMLP, CConvPE

def eyeEncoderConv(shape, name, latentSize):
  eye = L.Input(shape)
  
  res = eye
  features = []
  for sz in [64, 64, 64, 64]:
    res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)
    # res = CConvPE(channels=3, activation='tanh')(res)
    for _ in range(3):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)

    features.append(
      L.Conv2D(1, 3, padding='same', activation='relu')(
        res
      )
    )
    continue
  
  res = L.Concatenate(-1)([L.Flatten()(x) for x in features])
  res = L.Dense(latentSize, activation='relu')(res)
  model = tf.keras.Model(
    inputs=[eye],
    outputs=[res],
    name=name
  )
  # model.summary()
  return model

class CEyeEncoder(tf.keras.Model):
  def __init__(self, latent_size, **kwargs):
    super().__init__(**kwargs)
    self._latentSize = latent_size
    self.out_mlp = sMLP(sizes=[self._latentSize] * 2, activation='relu', name='%s/out_mlp' % self.name)
    self.ctx_mlp = sMLP(sizes=[64, 64], activation='relu', name='%s/ctx_mlp' % self.name)
    return

  def build(self, input_shape):
    eyeL_shape, eyeR_shape, context_shape = input_shape
    assert np.equal(eyeL_shape, eyeR_shape).all(), 'Left and right eye shapes must be equal'
    # Define the shared encoder
    eyeShp = (None, *eyeL_shape[1:3], 2 * eyeL_shape[3])
    self.encoder = eyeEncoderConv(
      eyeShp[1:],  latentSize=self._latentSize,
      name='%s/eyeEncoderConv' % self.name
    )
    self.encoder.build(eyeShp)
    return super().build(input_shape)

  def call(self, inputs):
    eyeL, eyeR, context = inputs
    # combine the eyes into one tensor (B, H, W, 2C)
    encoded = self.encoder(tf.concat([eyeL, eyeR], -1))
    # Encode the context
    ctx = self.ctx_mlp(context)
    return self.out_mlp(tf.concat([encoded, ctx], -1))

def eyeEncoder(shape=(32, 32, 1), latentSize=64):
  return CEyeEncoder(latent_size=latentSize, name='eyeEncoder')