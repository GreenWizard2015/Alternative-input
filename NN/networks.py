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
  def __init__(self, shift=0.5, **kwargs):
    super().__init__(**kwargs)
    self._shift = shift
    return
  
  def build(self, input_shape):
    self._mlp = sMLP(
      sizes=[128, 64, 32], activation='relu',
      name='%s/MLP' % self.name
    )
    self._mlp.build(input_shape)
    self._decodePoints = L.Dense(2, name='%s/DecodePoints' % self.name)
    return super().build(input_shape)
  
  def call(self, x):
    B = tf.shape(x)[0]
    N = tf.shape(x)[1]
    x = self._mlp(x)
    x = self._shift + self._decodePoints(x) # [0, 0] -> [0.5, 0.5]
    tf.assert_equal(tf.shape(x), (B, N, 2))
    return x
# End of IntermediatePredictor
################################################################

def Face2StepModel(pointsN, eyeSize, latentSize, embeddingsSize):
  points = L.Input((None, pointsN, 2))
  eyeL = L.Input((None, eyeSize, eyeSize, 1))
  eyeR = L.Input((None, eyeSize, eyeSize, 1))
  embeddings = L.Input((None, embeddingsSize))

  encodedEFList = CRolloutTimesteps(
    eyeEncoder(latentSize=latentSize), name='Eyes'
  )([eyeL, eyeR])
  encodedP = CRolloutTimesteps(
    FaceMeshEncoder(latentSize), name='FaceMesh'
  )(points)

  intermediate = {'F2S/encFace': encodedP,}
  # encodedEFList is a list of encoded eyes features
  # we need to combine them together and with the encodedP
  combined = encodedP # start with the face features
  for i, EFeat in enumerate(encodedEFList):
    combined = CFusingBlock(name='F2S/ResMul-%d' % i)([
      combined,
      sMLP(sizes=[latentSize] * 3, activation='relu', name='F2S/MLP-%d' % i)(
        L.LayerNormalization()(
          L.Concatenate(-1)([combined, encodedP, EFeat, embeddings])
        )
      )
    ])
     # save intermediate output
    intermediate['F2S/encEyes-%d' % i] = EFeat
    intermediate['F2S/combined-%d' % i] = combined
    continue
  
  combined = L.Dense(latentSize, name='F2S/Combine')(combined)
  combined = L.LayerNormalization()(combined)
  # combined = CQuantizeLayer()(combined)
  return tf.keras.Model(
    inputs={
      'points': points,
      'left eye': eyeL,
      'right eye': eyeR,
      'embeddings': embeddings,
    },
    outputs={
      'latent': combined,
      'intermediate': intermediate
    }
  )

def Step2LatentModel(latentSize, embeddingsSize):
  latents = L.Input((None, latentSize))
  embeddingsInput = L.Input((None, embeddingsSize))
  T = L.Input((None, 1))
  embeddings = embeddingsInput

  stepsData = latents
  intermediate = {}
  
  encodedT = CTimeEncoderLayer()(T)
  temporal = sMLP(sizes=[latentSize] * 1, activation='relu')(
    L.Concatenate(-1)([stepsData, encodedT, embeddings])
  )
  temporal = CFusingBlock()([stepsData, temporal])
  intermediate['S2L/enc0'] = temporal
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  for blockId in range(3):
    temp = L.Concatenate(-1)([temporal, encodedT, embeddings])
    for _ in range(3):
      temp = L.LSTM(latentSize, return_sequences=True)(temp)
    temp = sMLP(sizes=[latentSize] * 3, activation='relu')(
      L.Concatenate(-1)([temporal, temp, encodedT, embeddings])
    )
    temporal = CFusingBlock()([temporal, temp])
    intermediate['S2L/ResLSTM-%d' % blockId] = temporal
    continue
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  latent = sMLP(sizes=[latentSize] * 1, activation='relu')(
    L.Concatenate(-1)([stepsData, temporal, encodedT, embeddings])
  )
  latent = CFusingBlock()([stepsData, latent])
  return tf.keras.Model(
    inputs={
      'latent': latents,
      'time': T,
      'embeddings': embeddingsInput,
    },
    outputs={
      'latent': latent,
      'intermediate': intermediate
    }
  )

def _InputSpec():
  return {
    'points': tf.TensorSpec(shape=(None, None, 478, 2), dtype=tf.float32),
    'left eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'right eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'time': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
    'userId': tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
    'placeId': tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
    'screenId': tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
  }

def Face2LatentModel(
  pointsN=478, eyeSize=32, steps=None, latentSize=64,
  embeddings=None,
  diffusion=False # whether to use diffusion model
):
  points = L.Input((steps, pointsN, 2))
  eyeL = L.Input((steps, eyeSize, eyeSize, 1))
  eyeR = L.Input((steps, eyeSize, eyeSize, 1))
  T = L.Input((steps, 1))
  userIdEmb = L.Input((steps, embeddings['size']))
  placeIdEmb = L.Input((steps, embeddings['size']))
  screenIdEmb = L.Input((steps, embeddings['size']))
  
  emb = L.Concatenate(-1)([userIdEmb, placeIdEmb, screenIdEmb])
  if diffusion:
    diffusionT = L.Input((steps, 1))
    diffusionPoints = L.Input((steps, 2))
    encodedDT = CTimeEncoderLayer()(diffusionT)
    # shared transformation for all points
    encodedDP = CCoordsEncodingLayer(32, sharedTransformation=True)(diffusionPoints)
    # add diffusion features to the embeddings
    emb = L.Concatenate(-1)([emb, encodedDT, encodedDP])  
  
  emb = L.LayerNormalization()(emb)
  Face2Step = Face2StepModel(pointsN, eyeSize, latentSize, embeddingsSize=emb.shape[-1])
  Step2Latent = Step2LatentModel(latentSize, embeddingsSize=emb.shape[-1])

  stepsData = Face2Step({
    'embeddings': emb,
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
  })
  
  res = Step2Latent({
    'latent': stepsData['latent'],
    'time': T,
    'embeddings': emb,
  })
  res['intermediate'] = {
    **stepsData['intermediate'],
    **res['intermediate'],
  }

  inputs = {
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
    'time': T,
    'userId': userIdEmb,
    'placeId': placeIdEmb,
    'screenId': screenIdEmb,
  }
  res['result'] = IntermediatePredictor(
    shift=0.0 if diffusion else 0.5 # shift points to the center, if not using diffusion
  )(
    L.Concatenate(-1)([res['latent'], T, emb])
  )
  
  if diffusion:
    inputs['diffusionT'] = diffusionT
    inputs['diffusionPoints'] = diffusionPoints
    # make residuals
    res['result'] = diffusionPoints + res['result']
    
  main = tf.keras.Model(inputs=inputs, outputs=res)
  return {
    'intermediate shapes': {k: v.shape for k, v in res['intermediate'].items()},
    'main': main,
    'Face2Step': Face2Step,
    'Step2Latent': Step2Latent,
    'inputs specification': _InputSpec()
  }
  
if __name__ == '__main__':
  X = Face2LatentModel(steps=5, latentSize=64,
    embeddings={
      'userId': 1, 'placeId': 1, 'screenId': 1, 'size': 64
    }
  )
  X['main'].summary(expand_nested=True)
  X['Face2Step'].summary(expand_nested=False)
  X['Step2Latent'].summary(expand_nested=False)
  print(X['main'].outputs)
  pass