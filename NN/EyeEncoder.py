import tensorflow as tf
import tensorflow.keras.layers as L

import numpy as np
from NN.Utils import sMLP, CConvPE, CStackApplySplit
####################################
def custom_sobel(shape):
  res = []
  for axis in [0, 1]:
    k = np.zeros(shape)
    p = [(j,i) for j in range(shape[0]) 
           for i in range(shape[1]) 
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]

    for j, i in p:
      j_ = int(j - (shape[0] -1)/2.)
      i_ = int(i - (shape[1] -1)/2.)
      k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)

    res.append(k)
    continue
  return res

def createSobelsConv(sizes):
  res = []
  for sz in sizes:
    kernels = custom_sobel((sz, sz))
    res.extend(kernels)
    continue
  
  maxD = max([x.shape[0] for x in res])
  kernels = [np.pad(k, (maxD - k.shape[0]) // 2) for k in res]
  return np.stack(kernels, axis=0)

def _filters2conv(filters):
  ident = np.zeros(filters.shape[1:])
  ident[filters.shape[0] // 2, filters.shape[1] // 2] = 1.0
  filters = np.concatenate((ident[None], filters), axis=0)
  filters = filters.transpose(1, 2, 0)[..., None, :]
  return tf.constant(filters, tf.float32)

####################################
class CEyeEnricher(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._sobelConv = _filters2conv(createSobelsConv([3, 5, 7, 9, 11, 13]))
    return
  
  def call(self, x):
    x = tf.transpose(x, (0, 3, 1, 2))[..., None] # B, N, H, W, 1
    x = tf.nn.conv2d(x, self._sobelConv, strides=1, padding='SAME') # B, N, H, W, C
    
    B = tf.shape(x)[0]
    _, N, H, W, C = x.shape
    return tf.reshape(
      tf.transpose(x, (0, 2, 3, 1, 4)), 
      (B, H, W, N * C)
    )
####################################
def eyeEncoderConv(shape, name):
  eye = L.Input(shape)
  
  res = CEyeEnricher()(eye)
  features = []
  for sz in [8, 16, 32, 64]:
    res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)
    res = CConvPE(channels=3, activation='relu')(res)
    for _ in range(1):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)

    features.append(
      L.Conv2D(1, 3, padding='same', activation='relu')(
        res
      )
    )
    continue
  
  res = L.Concatenate(-1)([L.Flatten()(x) for x in features])
  res = L.Dense(64, activation='relu')(res)
  return tf.keras.Model(
    inputs=[eye],
    outputs=[res],
    name=name
  )

class CEyeEncoder(tf.keras.Model):
  def __init__(self, latent_size, **kwargs):
    super().__init__(**kwargs)
    self._latentSize = latent_size
    self.out_mlp_L = sMLP(sizes=[self._latentSize] * 2, activation='relu', name='%s/out_mlp_L' % self.name)
    self.out_mlp_R = sMLP(sizes=[self._latentSize] * 2, activation='relu', name='%s/out_mlp_R' % self.name)
    self.ctx_mlp = sMLP(sizes=[64, 64], activation='relu', name='%s/ctx_mlp' % self.name)
    return

  def build(self, input_shape):
    eyeL_shape, eyeR_shape, context_shape = input_shape
    # Define the shared encoder
    self.encoder = CStackApplySplit(
      eyeEncoderConv(eyeL_shape[1:], name='%s/eyeEncoderConv' % self.name),
      name='%s/encoder' % self.name
    )
    self.encoder.build(eyeL_shape)
    return super().build(input_shape)

  def call(self, inputs):
    eyeL, eyeR, context = inputs
    # Encode the left and right eye inputs
    encodedL, encodedR = self.encoder(eyeL, eyeR)

    # Encode the context
    ctx = self.ctx_mlp(context)
    return [
      self.out_mlp_L(tf.concat([encodedL, ctx], -1)),
      self.out_mlp_R(tf.concat([encodedR, ctx], -1))
    ]

def eyeEncoder(shape=(32, 32, 1), latentSize=64):
  return CEyeEncoder(latent_size=latentSize, name='eyeEncoder')