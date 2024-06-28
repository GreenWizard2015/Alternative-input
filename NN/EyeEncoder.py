import tensorflow as tf
import tensorflow.keras.layers as L

import numpy as np
from NN.Utils import sMLP, CConvPE

def _conv2latent(data, latentSize):
  feats = CConvPE(channels=32, activation='relu')(data)
  # N should be such that N * H * W ~ latentSize
  pixels = np.prod(feats.shape[1:3])
  N = 1 + int(latentSize // pixels)

  feats = L.Conv2D(N, 3, padding='same', activation='relu')(feats)
  feats = L.Conv2D(N, 2, padding='same', activation='relu')(feats)
  feats = L.Conv2D(N, 1, padding='same', activation='relu')(feats)
  
  feats = L.Flatten()(feats)
  feats = sMLP(sizes=[latentSize] * 1, activation='relu')(feats)
  return feats

def eyeEncoderConv(shape, name, latentSize):
  eye = L.Input(shape)
  
  res = eye
  features = []
  for sz in [64, 64, 64, 64]:
    res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)
    for _ in range(2):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)

    features.append(_conv2latent(res, latentSize))
    continue
  
  return tf.keras.Model(inputs=[eye], outputs=features, name=name)

class CEyeEncoder(tf.keras.Model):
  def __init__(self, latent_size, **kwargs):
    super().__init__(**kwargs)
    self._latentSize = latent_size
    return

  def build(self, input_shape):
    eyeL_shape, eyeR_shape = input_shape
    assert np.equal(eyeL_shape, eyeR_shape).all(), 'Left and right eye shapes must be equal'
    # Define the shared encoder
    eyeShp = (None, *eyeL_shape[1:3], 2 * eyeL_shape[3])
    self._encoder = eyeEncoderConv(
      eyeShp[1:],  latentSize=self._latentSize,
      name='%s/eyeEncoderConv' % self.name
    )
    self._encoder.build(eyeShp)
    return super().build(input_shape)

  def call(self, inputs):
    eyeL, eyeR = inputs
    # combine the eyes into one tensor (B, H, W, 2C)
    eyes = tf.concat([eyeL, eyeR], -1)
    return self._encoder(eyes)

def eyeEncoder(latentSize=64):
  return CEyeEncoder(latent_size=latentSize, name='eyeEncoder')