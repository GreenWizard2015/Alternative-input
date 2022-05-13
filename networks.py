import Utils
Utils.limitGPUMemory(memory_limit=1024)
import numpy as np
import math

import tensorflow as tf
import tensorflow.keras.layers as L

from CCoordsEncodingLayer import CCoordsEncodingLayer

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
##################
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

def _EyeEnricher_process(filters):
  ident = np.zeros(filters.shape[1:])
  ident[filters.shape[0] // 2, filters.shape[1] // 2] = 1.0
  filters = np.concatenate((ident[None], filters), axis=0)
  filters = filters.transpose(1, 2, 0)[..., None, :]
  filters = tf.constant(filters, tf.float32)

  @tf.function
  def F(srcImages):
    return tf.nn.conv2d(srcImages, filters, strides=1, padding='SAME')
  return F

@tf.function
def _EyeEnricher_unstackResults(images):
  B = tf.shape(images)[0]
  _, N, H, W, C = images.shape
  return tf.reshape(
    tf.transpose(images, (0, 2, 3, 1, 4)), 
    (B, H, W, N * C)
  )
  
def EyeEnricher(shape=(32, 32, 1)):  
  eye = L.Input(shape)
  res = L.Lambda(_EyeEnricher_processColors)(eye) # B, H, W, N
  res = L.Lambda(lambda x: tf.transpose(x, (0, 3, 1, 2))[..., None])(res) # B, N, H, W, 1
  res = L.Lambda(_EyeEnricher_process(createSobelsConv([3, 7, 15])))(res) # B, N, H, W, C
  res = L.Lambda(_EyeEnricher_unstackResults)(res) # B, H, W, N * C
  
  return tf.keras.Model(
    inputs=[eye],
    outputs=[res]
  )

def eyeEncoder(shape):
  eye = L.Input(shape)
  
  res = EyeEnricher(shape)([eye])
  res = L.Dropout(0.1)(res)
  for sz in [8, 16, 32, 32]:
    res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)
    for _ in range(3):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)
    continue
  
  return tf.keras.Model(
    inputs=[eye],
    outputs=[res]
  )

def _decodeSeries(x, base=2.0):
  N = tf.shape(x)[-1]
  coefs = tf.pow(base, tf.cast(tf.range(N), tf.float32))
  return tf.reduce_sum(x / coefs, axis=-1)

def pointsEncoder(pointsN):
  points = L.Input((pointsN, 2))
  
  encodedPoints = CCoordsEncodingLayer(N=30, name='points_encoder/coords')(points)
  encodedPoints = L.Lambda(
    lambda x: x[0] * tf.cast(tf.reduce_all(0.0 <= x[1], axis=-1), tf.float32)[..., None]
  )([encodedPoints, points])
  
  encodedPoints = sMLP(shape=encodedPoints.shape[1:], sizes=[16, 8, 8, 4])(encodedPoints)
  pts = L.Flatten()(encodedPoints)
  res = sMLP(shape=pts.shape[1:], sizes=[256, 256, 128, 128, 96])(pts)
  
  return tf.keras.Model(
    inputs=[points],
    outputs=[res]
  )

def simpleModel(pointsN=468, eyeSize=32):
  points = L.Input((pointsN, 2))
  eyeL = L.Input((eyeSize, eyeSize, 1))
  eyeR = L.Input((eyeSize, eyeSize, 1))
  
  encoder = eyeEncoder(eyeL.shape[1:])
  encodedL = encoder(eyeL)
  encodedR = encoder(eyeR)
  
  encodedP = pointsEncoder(pointsN=points.shape[1])(points)

  combined = L.Concatenate(axis=-1)([
    encodedP,
    L.Flatten()(encodedL),
    L.Flatten()(encodedR),
  ])

  coords = L.Dense(2 * 16, activation='relu')(
    sMLP(shape=combined.shape[1:], sizes=[256, 128, 64, 32])(
      combined
    )
  )
  coords = L.Reshape((2, -1))(coords)
  return tf.keras.Model(
    inputs=[points, eyeL, eyeR],
    outputs={
      'coords': L.Lambda(_decodeSeries)(coords)
    }
  )

def ARModel(pointsN=468, eyeSize=32):
  points = L.Input((pointsN, 2))
  eyeL = L.Input((eyeSize, eyeSize, 1))
  eyeR = L.Input((eyeSize, eyeSize, 1))
  
  position = L.Input((2, ))
  pos = L.Reshape((1, 2))(position)
  
  encoder = eyeEncoder(eyeL.shape[1:])
  encodedL = L.Flatten()(encoder(eyeL))
  encodedR = L.Flatten()(encoder(eyeR))
  
  encodedFace = pointsEncoder(pointsN=points.shape[1])(points)

  combined = L.Concatenate(axis=-1)([
    L.Flatten()(CCoordsEncodingLayer(32, name='posA')(pos)),
    encodedFace, encodedL, encodedR,
  ])
  
  latent = sMLP(shape=combined.shape[1:], sizes=[256, 128, 64, 32])(combined)
  
  combined = L.Concatenate(axis=-1)([
    L.Flatten()(CCoordsEncodingLayer(32, name='posB')(pos)), 
    latent
  ])
  coords = L.Dense(2 * 16, activation='linear')(
    sMLP(shape=combined.shape[1:], sizes=[64, 64, 64])(
      combined
    )
  )
  coords = L.Reshape((2, -1))(coords)
  coords = L.Lambda(_decodeSeries)(coords)
  #############
  return tf.keras.Model(
    inputs=[points, eyeL, eyeR, position],
    outputs=[position + coords]
  )
###########
def NerfLikeEncoder(pointsN=468, eyeSize=32, latentSize=64):
  points = L.Input((pointsN, 2))
  eyeL = L.Input((eyeSize, eyeSize, 1))
  eyeR = L.Input((eyeSize, eyeSize, 1))
  
  encoder = eyeEncoder(eyeL.shape[1:])
  encodedL = encoder(eyeL)
  encodedR = encoder(eyeR)
  
  encodedP = pointsEncoder(pointsN=points.shape[1])(points)

  combined = L.Concatenate(axis=-1)([
    encodedP,
    L.Flatten()(encodedL),
    L.Flatten()(encodedR),
  ])
  
  latent = L.Dense(latentSize, activation='relu')(
    sMLP(shape=combined.shape[1:], sizes=[256, 128, 128, 64])(
      combined
    )
  )
  return tf.keras.Model(
    inputs=[points, eyeL, eyeR],
    outputs={
      'latent': latent
    }
  )

def NerfLikeDecoder(latentSize=64):
  point = L.Input((2,))
  latent = latentX = L.Input((latentSize,))
 
  encodedP = L.Flatten()(
    CCoordsEncodingLayer(32)( L.Reshape((1, 2))(point) )
  )
  initState = L.Concatenate(axis=-1)([encodedP, latent])

  for blockId in range(1):
    combined = L.Concatenate(axis=-1)([initState, latent])
    latent = sMLP(shape=combined.shape[1:], sizes=[128]*5, activation='relu')(
      combined
    )
    continue
  
  series = L.Dense(24, activation='relu')(
    L.Concatenate(axis=-1)([initState, latent])
  )
  return tf.keras.Model(
    inputs=[latentX, point],
    outputs={
      'valueAt': L.Lambda(lambda x: _decodeSeries(x[:, None], base=2.0))(series)
    }
  )

if __name__ == '__main__':
  ARModel().summary()
#   NerfLikeEncoder().summary()
#   NerfLikeDecoder().summary()
  pass