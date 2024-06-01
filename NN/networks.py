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
      sMLP(sizes=[latentSize] * 1, activation='relu', name='F2S/MLP-%d' % i)(
        L.Concatenate(-1)([combined, encodedP, EFeat, embeddings])
      )
    ])
     # save intermediate output
    intermediate['F2S/encEyes-%d' % i] = EFeat
    intermediate['F2S/combined-%d' % i] = combined
    continue
  
  combined = L.Dense(latentSize, name='F2S/Combine')(combined)
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
  embeddings = L.Input((None, embeddingsSize))
  T = L.Input((None, 1))

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
    temp = L.Concatenate(-1)([temporal, encodedT])
    for _ in range(1):
      temp = L.LSTM(latentSize, return_sequences=True)(temp)
    temp = sMLP(sizes=[latentSize] * 1, activation='relu')(
      L.Concatenate(-1)([temporal, temp])
    )
    temporal = CFusingBlock()([temporal, temp])
    intermediate['S2L/ResLSTM-%d' % blockId] = temporal
    continue
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  latent = sMLP(sizes=[latentSize] * 1, activation='relu')(
    L.Concatenate(-1)([stepsData, temporal, encodedT, encodedT])
  )
  latent = CFusingBlock()([stepsData, latent])
  return tf.keras.Model(
    inputs={
      'latent': latents,
      'time': T,
      'embeddings': embeddings,
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
  embeddings=None
):
  points = L.Input((steps, pointsN, 2))
  eyeL = L.Input((steps, eyeSize, eyeSize, 1))
  eyeR = L.Input((steps, eyeSize, eyeSize, 1))
  T = L.Input((steps, 1))
  userIdEmb = L.Input((steps, embeddings['size']))
  placeIdEmb = L.Input((steps, embeddings['size']))
  screenIdEmb = L.Input((steps, embeddings['size']))
  
  emb = L.Concatenate(-1)([userIdEmb, placeIdEmb, screenIdEmb])
  
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
  # drop all intermediate outputs
  res['intermediate'] = {}

  inputs = {
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
    'time': T,
    'userId': userIdEmb,
    'placeId': placeIdEmb,
    'screenId': screenIdEmb,
  }

  intermediate = res['intermediate']
  IP = lambda x: IntermediatePredictor()(x) # own IntermediatePredictor for each output
  res['intermediate'] = {k: IP(x) for k, x in intermediate.items()}
  res['result'] = IP(res['latent'])
  ###################################
  # TODO: figure out is this helpful or not
  # branch for global coordinates transformation
  # predict shift, rotation, scale
  emb = L.Concatenate(-1)([userIdEmb, placeIdEmb, screenIdEmb])
  emb = sMLP(sizes=[64, 64, 64, 64, 32], activation='relu')(emb[:, 0])
  shift = L.Dense(2, name='GlobalShift')(emb)[:, None]
  rotation = L.Dense(1, name='GlobalRotation', activation='sigmoid')(emb)[:, None] * np.pi
  scale = L.Dense(2, name='GlobalScale')(emb)[:, None]

  shifted = res['result'] + shift - 0.5 # [0.5, 0.5] -> [0, 0]
  # Rotation matrix components
  cos_rotation = L.Lambda(lambda x: tf.cos(x))(rotation)
  sin_rotation = L.Lambda(lambda x: tf.sin(x))(rotation)
  rotation_matrix = L.Lambda(lambda x: tf.stack([x[0], x[1]], axis=-1))([cos_rotation, sin_rotation])

  # Apply rotation
  rotated = L.Lambda(
    lambda x: tf.einsum('isj,iomj->isj', x[0], x[1])
  )([shifted, rotation_matrix]) + 0.5 # [0, 0] -> [0.5, 0.5] back

  # Apply scale
  scaled = rotated * scale
  def clipWithGradient(x):
    res = tf.clip_by_value(x, 0.0, 1.0)
    return x + tf.stop_gradient(res - x)
  
  res['result'] = L.Lambda(clipWithGradient)(scaled)
  ###################################

  main = tf.keras.Model(inputs=inputs, outputs=res)
  return {
    'main': main,
    'Face2Step': Face2Step,
    'Step2Latent': Step2Latent,
    'inputs specification': _InputSpec(),
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