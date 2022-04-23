import Utils
import numpy
import math
Utils.limitGPUMemory(memory_limit=1024)

import tensorflow as tf
import tensorflow.keras.layers as L

def sMLP(shape, sizes):
  data = L.Input(shape)
  
  res = data
  for s in sizes:
    res = L.Dense(s, activation='linear')(
      L.Dropout(0.05)(res)
    )
    continue
    
  return tf.keras.Model(
    inputs=[data],
    outputs=[res]
  )

@tf.function
def _EyeEnricher_process(srcImages):
  contrasted = []
  edges = []
  gammaAdjusted = []
  for gamma in [4.0, 1.0, 1.0 / 4.0]:
    images = tf.image.adjust_gamma(srcImages, gamma)
    gammaAdjusted.append(images)
    for contrast in [4.0, 8.0, 16.0]:
      imgs = tf.image.adjust_contrast(images, contrast)
      contrasted.append(imgs)
      
      sobel = tf.image.sobel_edges(imgs)
      edges.append(sobel[..., 0])
      edges.append(sobel[..., 1])
      continue
    continue
  
  res = [srcImages] + gammaAdjusted + contrasted + edges
  res = tf.concat(res, axis=-1)
  return tf.stop_gradient(res) # prevent some issues with XLA

def EyeEnricher(shape=(32, 32, 1)):  
  eye = L.Input(shape)
  res = L.Lambda(_EyeEnricher_process)(eye)
  return tf.keras.Model(
    inputs=[eye],
    outputs=[res]
  )

def eyeEncoder(shape):
  eye = L.Input(shape)
  
  res = EyeEnricher(shape)([eye]) # 1 channel => 30
  res = L.Dropout(0.5)(res) # 30 channels => ~15
  for sz in [8, 16, 32, 32]:
    res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)
    for _ in range(3):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)
    continue
  
  return tf.keras.Model(
    inputs=[eye],
    outputs=[res]
  )

def _PointsEnricher_process(points, N=5):
  validMask = tf.cast(tf.reduce_all(0.0 <= points, axis=-1), tf.float32)
  
  mask = tf.cast(tf.range(N), tf.float32)
  freqPi = tf.linspace(0.1, 10.0, N)[None, None, None] * (2 * math.pi)
  shifts = mask[None, None, None]
  X = (points[..., None] + shifts) * freqPi
  sinX = tf.sin(X)
  cosX = tf.cos(X)

#   sin_mask = mask % 2
#   cos_mask = 1 - sin_mask
#   res = (sinX * sin_mask) + (cosX * cos_mask)
  res = sinX - cosX
  res = tf.concat([res[..., 0, :], res[..., 1, :]], axis=-1)
  
  return res * validMask[..., None]

def PointsEnricher(shape):
  points = L.Input(shape)
  res = L.Lambda(_PointsEnricher_process)(points)
  return tf.keras.Model(
    inputs=[points],
    outputs=[res]
  )
  
def pointsEncoder(pointsN):
  points = L.Input((pointsN, 2))
  
  pts = L.Flatten()(
    PointsEnricher(points.shape[1:])(points)
  )
  res = sMLP(shape=pts.shape[1:], sizes=[256, 256, 128, 128, 96])(pts)
  
  return tf.keras.Model(
    inputs=[points],
    outputs=[res]
  )

def simpleModel(pointsN=468, eyeSize=32):
  # TODO: Try NeRF-like approach i.e. predict some latent representation and "decode" into 2d probabilities map by sampling
  # f(latent, 0..1, 0..1)
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
  
  coords = L.Dense(2, activation='linear')(
    sMLP(shape=combined.shape[1:], sizes=[256, 128, 64, 32])(
      combined
    )
  )
  return tf.keras.Model(
    inputs=[points, eyeL, eyeR],
    outputs=[coords]
  )

def ARModel(pointsN=468, eyeSize=32):
  position = L.Input((2, ))
  points = L.Input((pointsN, 2))
  eyeL = L.Input((eyeSize, eyeSize, 1))
  eyeR = L.Input((eyeSize, eyeSize, 1))
  
  encoder = eyeEncoder(eyeL.shape[1:])
  encodedL = encoder(eyeL)
  encodedR = encoder(eyeR)
  
  pts = L.Concatenate(axis=1)([ points, L.Reshape((1, 2))(position) ])
  encodedP = pointsEncoder(pointsN=pts.shape[1])(pts)

  combined = L.Concatenate(axis=-1)([
    position,
    encodedP,
    L.Flatten()(encodedL),
    L.Flatten()(encodedR),
  ])
  
  coords = position + L.Dense(2, activation='linear')(
    sMLP(shape=combined.shape[1:], sizes=[256, 128, 64, 32])(
      combined
    )
  )
  return tf.keras.Model(
    inputs=[points, eyeL, eyeR, position],
    outputs=[coords]
  )

if __name__ == '__main__':
  simpleModel().summary()
