import tensorflow as tf
import tensorflow.keras.layers as L

import numpy as np
from NN.Utils import sMLP
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

@tf.function
def _EyeEnricher_processColors(srcImages):
  contrasted = []
  gammaAdjusted = []
  for gamma in [4.0, 1.0, 1.0 / 4.0]:
    images = tf.image.adjust_gamma(srcImages, gamma)
    gammaAdjusted.append(images)
    for contrast in [4.0, 8.0, 16.0]:
      imgs = tf.image.adjust_contrast(images, contrast)
      contrasted.append(imgs)
      continue
    continue
  
  res = [srcImages] + gammaAdjusted + contrasted
  res = tf.concat(res, axis=-1)
  return tf.stop_gradient(res) # prevent some issues with XLA

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
    self._sobelConv = _filters2conv(createSobelsConv([3, 7, 15]))
    return
  
  def call(self, x):
    x = _EyeEnricher_processColors(x) # B, H, W, N
    x = tf.transpose(x, (0, 3, 1, 2))[..., None] # B, N, H, W, 1
    x = tf.nn.conv2d(x, self._sobelConv, strides=1, padding='SAME') # B, N, H, W, C
    
    B = tf.shape(x)[0]
    _, N, H, W, C = x.shape
    return tf.reshape(
      tf.transpose(x, (0, 2, 3, 1, 4)), 
      (B, H, W, N * C)
    ) # B, H, W, N * C
####################################
def eyeEncoderConv(shape):
  eye = L.Input(shape)
  
  res = CEyeEnricher()(eye)
  res = L.Dropout(0.1)(res)
  for sz in [8, 16, 32, 32]:
    res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)
    for _ in range(3):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)
    continue

  return tf.keras.Model(
    inputs=[eye],
    outputs=[res],
    name='eyeEncoderConv'
  )

def eyeEncoder(shape=(32, 32, 1), latentSize=64):
  eyeL = L.Input(shape)
  eyeR = L.Input(shape)
  
  encoder = eyeEncoderConv(shape)
  encodedL = L.Flatten()( encoder(eyeL) )
  encodedR = L.Flatten()( encoder(eyeR) )
  
  resL = sMLP(encodedL.shape[1:], sizes=[128, 96, 80, latentSize])(encodedL)
  resR = sMLP(encodedR.shape[1:], sizes=[128, 96, 80, latentSize])(encodedR)
  return tf.keras.Model(
    inputs=[eyeL, eyeR],
    outputs=[resL, resR],
    name='eyeEncoder'
  )
