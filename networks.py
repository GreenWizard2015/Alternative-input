import Utils
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

def eyeEncoder(shape):
  eye = L.Input(shape)
  
  res = eye
  for sz in [8, 16, 32, 32]:
    res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)
    for _ in range(3):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)
    continue
  
  return tf.keras.Model(
    inputs=[eye],
    outputs=[res]
  )

def simpleModel(pointsN=468, eyeSize=32):
  # TODO: Try autoregressive approach i.e. pass in coordinates and try to predict more accurate position
  # TODO: Try NeRF-like approach i.e. predict some latent representation and "decode" into 2d probabilities map by sampling
  # f(latent, 0..1, 0..1)
  points = L.Input((pointsN, 2))
  eyeL = L.Input((eyeSize, eyeSize, 1))
  eyeR = L.Input((eyeSize, eyeSize, 1))
  
  encoder = eyeEncoder(eyeL.shape[1:])
  encodedL = encoder(eyeL)
  encodedR = encoder(eyeR)
  
  pts = L.Flatten()(points)
  encodedP = sMLP(shape=pts.shape[1:], sizes=[256, 128])(pts)
      
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

if __name__ == '__main__':
  simpleModel().summary()
