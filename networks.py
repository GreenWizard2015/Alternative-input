import Utils
Utils.limitGPUMemory(memory_limit=1024)

import tensorflow as tf
import tensorflow.keras.layers as L

from NN.CCoordsEncodingLayer import CCoordsEncodingLayer
from NN.Utils import CDecodeSeries, sMLP
from NN.EyeEncoder import eyeEncoder
from NN.FaceMeshEncoder import FaceMeshEncoder

def simpleModel(pointsN=468, eyeSize=32):
  points = L.Input((pointsN, 2))
  eyeL = L.Input((eyeSize, eyeSize, 1))
  eyeR = L.Input((eyeSize, eyeSize, 1))
  
  encodedL, encodedR = eyeEncoder()([eyeL, eyeR])
  encodedP = FaceMeshEncoder()(points)

  combined = L.Concatenate(axis=-1)([encodedP, encodedL, encodedR])

  coords = L.Dense(2 * 16, activation='relu')(
    sMLP(shape=combined.shape[1:], sizes=[256, 128, 64, 32])(
      combined
    )
  )
  coords = L.Reshape((2, -1))(coords)
  return tf.keras.Model(
    inputs=[points, eyeL, eyeR],
    outputs={
      'coords': CDecodeSeries()(coords)
    }
  )

def ARModel(pointsN=468, eyeSize=32):
  points = L.Input((pointsN, 2))
  eyeL = L.Input((eyeSize, eyeSize, 1))
  eyeR = L.Input((eyeSize, eyeSize, 1))
  
  position = L.Input((2, ))
  pos = L.Reshape((1, 2))(position)
  
  encodedL, encodedR = eyeEncoder()([eyeL, eyeR])
  
  encodedFace = FaceMeshEncoder()(points)

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
  coords = CDecodeSeries()(coords)
  #############
  return tf.keras.Model(
    inputs=[points, eyeL, eyeR, position],
    outputs=[position + coords]
  )

if __name__ == '__main__':
  ARModel().summary()
  pass