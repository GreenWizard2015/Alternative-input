from Core.Utils import setupGPU
setupGPU() # dirty hack to setup GPU memory limit on startup

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
# End of IntermediatePredictor
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

  combined = CResidualMultiplicativeLayer()([
    encodedP,
    sMLP(sizes=[latentSize] * 1, activation='relu')(L.Concatenate(-1)([encodedP, encoded, context]))
  ])
  
  inputs = {
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
    'context': context,
  }
  
  return tf.keras.Model(
    inputs=inputs,
    outputs={
      'latent': combined,
      'intermediate': {
        'F2S/encEyes': encoded,
        'F2S/encFace': encodedP,
        'F2S/combined': combined,
      }
    }
  )

def Step2LatentModel(latentSize, contextSize):
  stepsDataInput = L.Input((None, latentSize))
  context = L.Input((None, contextSize))
  T = L.Input((None, 1))

  stepsData = stepsDataInput
  intermediate = {}
  
  encodedT = CTimeEncoderLayer()(T)
  temporal = sMLP(sizes=[latentSize] * 3, activation='relu')(
    L.Concatenate(-1)([stepsData, encodedT, context])
  )
  temporal = CResidualMultiplicativeLayer()([stepsData, temporal])
  intermediate['S2L/enc0'] = temporal
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  # simple LSTM to capture temporal dependencies
  temp = L.LSTM(128, return_sequences=True)(temporal)
  for i in range(2):
    temp = L.LSTM(temp.shape[-1], return_sequences=True)(temp)
  temp = sMLP(sizes=[latentSize] * 3, activation='relu')(
    L.Concatenate(-1)([temporal, context, encodedT, temp])
  )
  temporal = CResidualMultiplicativeLayer()([temporal, temp])
  intermediate['S2L/ResLSTM'] = temporal
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  latent = sMLP(sizes=[latentSize] * 3, activation='relu')(
    L.Concatenate(-1)([stepsData, temporal, encodedT, context, encodedT])
  )
  latent = CResidualMultiplicativeLayer()([stepsData, latent])
  return tf.keras.Model(
    inputs=[stepsDataInput, T, context],
    outputs={
      'latent': latent,
      'intermediate': intermediate
    }
  )

def _InputSpec():
  return {
    'points': tf.TensorSpec(shape=(None, None, 468, 2), dtype=tf.float32),
    'left eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'right eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'time': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
  }

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
  res['intermediate'] = {
    **stepsData['intermediate'],
    **res['intermediate'],
  }
  # drop all intermediate outputs
  res['intermediate'] = {}

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

  intermediate = res['intermediate']
  IP = lambda x: IntermediatePredictor()(x) # own IntermediatePredictor for each output
  res['intermediate'] = {k: IP(x) for k, x in intermediate.items()}
  res['result'] = IP(res['latent'])

  main = tf.keras.Model(inputs=inputs, outputs=res)
  return {
    'main': main,
    'Face2Step': Face2Step,
    'Step2Latent': Step2Latent,
    'inputs specification': _InputSpec(),
  }
  
if __name__ == '__main__':
  X = Face2LatentModel(steps=5, latentSize=64, contexts=None)
  X['main'].summary(expand_nested=True)
  # X['Face2Step'].summary(expand_nested=False)
  # X['Step2Latent'].summary(expand_nested=False)
  # print(X['main'].outputs)
  pass