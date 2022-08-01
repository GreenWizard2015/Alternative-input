import Utils
Utils.setupGPU(memory_limit=1024)

import tensorflow as tf
import tensorflow.keras.layers as L

from NN.CCoordsEncodingLayer import CCoordsEncodingLayer
from NN.Utils import CDecodeSeries, sMLP, CRolloutTimesteps, CDecodePoint, CGate
from NN.EyeEncoder import eyeEncoder
from NN.FaceMeshEncoder import FaceMeshEncoder

import numpy as np

def Face2LatentModel(pointsN=468, eyeSize=32, steps=None, residual=True):
  shapePrefix = [steps]
  points = L.Input((*shapePrefix, pointsN, 2))
  eyeL = L.Input((*shapePrefix, eyeSize, eyeSize, 1))
  eyeR = L.Input((*shapePrefix, eyeSize, eyeSize, 1))
  
  encodeEyes = CRolloutTimesteps(eyeEncoder())
  encodeMesh = CRolloutTimesteps(FaceMeshEncoder())

  encodedL, encodedR = encodeEyes([eyeL, eyeR])
  encodedP = encodeMesh(points)

  combined = L.Concatenate(-1)([encodedP, encodedL, encodedR])
  combined = sMLP(sizes=[256, 64])(combined)
  if residual:
    temp = sMLP(sizes=[64, 64])(combined)
    temp = L.LSTM(temp.shape[-1], return_sequences=True)(temp)
    temp = sMLP(sizes=[64, 64])(temp)
    combined = combined + CGate(axis=[-1, -2])(temp)
  else:
    combined = L.LSTM(64, return_sequences=True)(combined)
    pass
  
  latent = sMLP(sizes=[64, 64])(combined)
  return tf.keras.Model(
    inputs=[points, eyeL, eyeR],
    outputs={
      'latent': latent,
    }
  )

def simpleModel(FACE_LATENT_SIZE=None):
  latentFace = L.Input((FACE_LATENT_SIZE, ))

  latent = sMLP(sizes=[256, 256, 256, ], activation='relu')(latentFace)
  return tf.keras.Model(
    inputs=[latentFace],
    outputs={
      'coords': 0.5 + CDecodePoint(16)(latent),
      'submodels': [],
    }
  )

def ARModel(FACE_LATENT_SIZE=None, residualPos=False):
  latentFace = L.Input((FACE_LATENT_SIZE, ))
  position = L.Input((2, ))
  pos = L.Reshape((1, 2))(position)
  POS_FREQ_N = 32
  
  combined = L.Concatenate(axis=-1)([
    L.Flatten()(CCoordsEncodingLayer(POS_FREQ_N, name='posA')(pos)),
    L.Dropout(0.5)(latentFace)
  ])
  
  latent = sMLP(sizes=[64*4, ] * 4)(combined)
  #############
  combined = L.Concatenate(axis=-1)([
    L.Flatten()(CCoordsEncodingLayer(POS_FREQ_N, name='posB')(pos)), 
    latent,
    L.Dropout(0.5)(latentFace)
  ])

  coords = L.Dense(2 * 16, activation='linear')(
    sMLP(sizes=[64*4, ] * 4, activation='relu')(
      combined
    )
  )
  # coords = tf.nn.sigmoid(coords)
  # coords = tf.clip_by_value(coords, 0.0, 1.0)
  coords = L.Reshape((2, -1))(coords)
   
  P = -1.0 * np.arange(coords.shape[-1])
  coords = CDecodeSeries(base=2.0, powers=P)(coords)
  
  if residualPos:
    coords = position + coords
  #############
  return tf.keras.Model(
    inputs=[latentFace, position],
    outputs={
      'coords': coords,
    }
  )

if __name__ == '__main__':
  Face2LatentModel(steps=5).summary()
#   simpleModel(64).summary()
  pass