import Core.Utils as Utils
Utils.setupGPU(memory_limit=1024)

import tensorflow as tf
import tensorflow.keras.layers as L

from NN.CCoordsEncodingLayer import CCoordsEncodingLayer
from NN.Utils import *
from NN.EyeEncoder import eyeEncoder
from NN.FaceMeshEncoder import FaceMeshEncoder

import numpy as np

class CTimeEncoderLayer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._encoder = CRolloutTimesteps(CCoordsEncodingLayer(32))
    return
  
  def call(self, T):
    T = self._encoder(T[..., None])
    return T[..., 0, :]

class IntermediatePredictor(tf.keras.layers.Layer):
  def build(self, input_shape):
    self._mlp = sMLP(
      sizes=[128, 64, 32], activation='relu',
      name='%s/MLP' % self.name
    )
    self._decodePoints = L.Dense(2, name='%s/DecodePoints' % self.name)
    return super().build(input_shape)
  
  def call(self, x):
    B = tf.shape(x)[0]
    N = tf.shape(x)[1]
    x = self._mlp(x)
    x = 0.5 + self._decodePoints(x) # [0, 0] -> [0.5, 0.5]
    tf.assert_equal(tf.shape(x), (B, N, 2))
    return x
################################################################

def Face2StepModel(pointsN, eyeSize, latentSize, contextSize):
  points = L.Input((None, pointsN, 2))
  eyeL = L.Input((None, eyeSize, eyeSize, 1))
  eyeR = L.Input((None, eyeSize, eyeSize, 1))
  context = L.Input((None, contextSize))

  encoded = CRolloutTimesteps(
    eyeEncoder(latentSize=latentSize), name='Eyes'
  )([eyeL, eyeR, context])
  encodedP = CRolloutTimesteps(
    FaceMeshEncoder(latentSize), name='FaceMesh'
  )([points, context])

  combined = L.Concatenate(-1)([encodedP, encoded, context])
  combined = sMLP(sizes=[256, latentSize], activation='relu')(combined)
  
  inputs = {
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
    'context': context,
  }
  
  # IP = IntermediatePredictor() # same IntermediatePredictor for all outputs
  IP = lambda x: IntermediatePredictor()(x) # own IntermediatePredictor for each output
  return tf.keras.Model(
    inputs=inputs,
    outputs={
      'latent': combined,
      'intermediate': []#IP(x) for x in [encodedP, encodedL, encodedR, combined]],
    }
  )

def Step2LatentModel(latentSize, contextSize):
  stepsDataInput = L.Input((None, latentSize))
  context = L.Input((None, contextSize))
  T = L.Input((None, 1))

  stepsData = stepsDataInput
  intermediate = []
  
  encodedT = CTimeEncoderLayer()(T)
  temporal = sMLP(sizes=[latentSize, latentSize], activation='relu')(
    L.Concatenate(-1)([stepsData, encodedT, context])
  )
  intermediate.append(temporal) # aux output per time step
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  useLSTM = not True
  if useLSTM:
    temp = L.LSTM(128, return_sequences=True)(temporal)
    for i in range(2):
      temp = L.LSTM(temp.shape[-1], return_sequences=True)(temp)
    temp = sMLP(sizes=[temp.shape[-1], temporal.shape[-1]], activation='relu')(temp)
    temporal = temporal + CGate(axis=[-1])(temp)
    intermediate.append(temporal)
  else:
    for i in range(3):
      temporal = CMyTransformerLayer(
        latentSize, 32,
        # toQuery=sMLP(sizes=[64, latentSize], activation='relu'),
        # toKey=sMLP(sizes=[64, latentSize], activation='relu'),
        useNormalization=True,
      )(temporal)
      intermediate.append(temporal)
      continue
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  
  latent = temporal
#   IP = IntermediatePredictor() # same IntermediatePredictor for all outputs
  IP = lambda x: IntermediatePredictor()(x) # own IntermediatePredictor for each output
  intermediate = [intermediate[i] for i in [0, -1]]
  intermediate = [IP(x) for x in intermediate]
  result = intermediate[-1] #sum(intermediate) / len(intermediate)
  return tf.keras.Model(
    inputs=[stepsDataInput, T, context],
    outputs={
      'latent': latent,
      'intermediate': intermediate,
      'result': result
    }
  )

def Face2LatentModel(pointsN=468, eyeSize=32, steps=None, latentSize=64, contexts=[1, 1, 1]):
  points = L.Input((steps, pointsN, 2))
  eyeL = L.Input((steps, eyeSize, eyeSize, 1))
  eyeR = L.Input((steps, eyeSize, eyeSize, 1))
  T = L.Input((steps, 1))
  if not(contexts is None):
    ContextID = L.Input((steps, len(contexts)))
    ctx = CContextEncoder(dims=32, contexts=contexts)(ContextID)
  else:
    # make a dummy context ID
    ctx = L.Lambda(
      lambda x: tf.zeros([tf.shape(x)[0], tf.shape(x)[1], 1], tf.float32)
    )(points)
    pass

  ctxSize = ctx.shape[-1]
  Face2Step = Face2StepModel(pointsN, eyeSize, latentSize, ctxSize)
  Step2Latent = Step2LatentModel(latentSize, ctxSize)

  stepsData = Face2Step({
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
    'context': ctx,
  })
  
  res = Step2Latent([stepsData['latent'], T, ctx])
  res['context'] = ctx
  res['intermediate'] = stepsData['intermediate'] + res['intermediate']

  inputs = {
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
    'time': T
  }

  if contexts is None:
    res['shift'] = L.Lambda(
      lambda x: tf.zeros((tf.shape(x)[0], 2), tf.float32)
    )(points)
  else:
    res['shift'] = L.Dense(2)(
      sMLP(sizes=[32,]*4, activation='relu')(ctx)
    )
    inputs['ContextID'] = ContextID
    pass

  main = tf.keras.Model(inputs=inputs, outputs=res)
  return {
    'main': main,
    'Face2Step': Face2Step,
    'Step2Latent': Step2Latent
  }

def simpleModel(FACE_LATENT_SIZE=None):
  latentFace = L.Input((FACE_LATENT_SIZE, ))
  pos = L.Input((2, ))

  latent = sMLP(sizes=[256,]*4, activation='relu')(
    L.Concatenate(-1)([
      latentFace, 
      CCoordsEncodingLayer(32)(pos[:, None])[:, 0]
    ])
  )
  return tf.keras.Model(
    inputs=[latentFace, pos],
    outputs={
      'coords': .5 + CDecodePoint(16)(latent)
    }
  )

def ARModel(FACE_LATENT_SIZE=None, residualPos=False):
  latentFace = L.Input((FACE_LATENT_SIZE, ))
  position = L.Input((2, ))
  pos = L.Reshape((1, 2))(position)
  POS_FREQ_N = 32
  
  combined = L.Concatenate(axis=-1)([
    L.Flatten()(CCoordsEncodingLayer(POS_FREQ_N, name='posA')(pos)),
    L.Dropout(0.05)(latentFace)
  ])
  
  latent = sMLP(sizes=[64*4, ] * 4)(combined)
  #############
  combined = L.Concatenate(axis=-1)([
    L.Flatten()(CCoordsEncodingLayer(POS_FREQ_N, name='posB')(pos)), 
    latent,
    L.Dropout(0.01)(latentFace)
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
  X = Face2LatentModel(steps=5, latentSize=64, contexts=None)
  X['main'].summary(expand_nested=True)
  # X['Face2Step'].summary(expand_nested=False)
  # X['Step2Latent'].summary(expand_nested=False)
  # print(X['main'].outputs)
  pass