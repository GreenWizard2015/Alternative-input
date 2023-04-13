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
  
class CContextEncoder(tf.keras.layers.Layer):
  def __init__(self, dims, contexts, **kwargs):
    super().__init__(**kwargs)
    self._dropoutRate = 0.1
    self._dims = dims
    self._embeddings = []
    self._embeddingsDecoder = []
    for i, N in enumerate(contexts):
      # last 2 are for augmentations id
      isAugmented = (len(contexts) - 2) <= i
      self._embeddings.append(tf.keras.layers.Embedding(
        N, 
        1 if isAugmented else dims,
        name='%s/embeddings-%d' % (self.name, i)
      ))

      decoder = lambda x: x
      if isAugmented: # 1 -> dims
        decoder = tf.keras.Sequential([
          # apply tanh to keep values in [-1, 1]
          tf.keras.layers.Lambda(tf.math.tanh),
          tf.keras.layers.Dense(dims, activation='relu'),
          tf.keras.layers.Dense(dims, activation='relu'),
        ], name='%s/decoder-%d' % (self.name, i))

      self._embeddingsDecoder.append(decoder)
      continue
    
    self._encoder = tf.keras.Sequential([
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(dims, activation='relu'),
      tf.keras.layers.Dense(dims, activation='relu'),
    ], name='%s/encoder' % self.name)
    return

  @property
  def embeddings(self):
    return self._embeddings
  
  def call(self, contextID, training=None):
    B, N, C = [tf.shape(contextID)[i] for i in range(3)]
    tf.assert_equal(C, len(self._embeddings))

    embeddings = []
    for i, emb in enumerate(self._embeddings):
      emb = emb(contextID[..., i])
      emb = self._embeddingsDecoder[i](emb)
      shp = tf.shape(emb)
      tf.assert_equal(shp, (B, N, shp[-1]))
      embeddings.append(emb)
      continue

    # if training: # apply dropout in training mode
    #   for i, emb in enumerate(embeddings):
    #     # mask out each embedding with dropout rate
    #     mask = tf.random.uniform((B, N, 1)) < self._dropoutRate
    #     emb = tf.where(mask, 0.0, emb)
    #     # mask out gradients
    #     # embS = tf.stop_gradient(emb)
    #     # mask = tf.random.uniform((B, N, 1)) < 0.1
    #     # emb = tf.where(mask, embS, emb)
    #     embeddings[i] = emb
    #     continue
    #   pass

    embeddings = tf.concat(embeddings, axis=-1)
    return self._encoder(embeddings)

class IntermediatePredictor(tf.keras.layers.Layer):
  def build(self, input_shape):
    self._mlp = sMLP(
      sizes=[128, 64, 32], activation='relu',
      name='%s/MLP' % self.name
    )
    self._decodePoints = CDecodePoint(
      16, base=1.1,
      name='%s/DecodePoints' % self.name
    )
    return super().build(input_shape)
  
  def call(self, x):
    x = self._mlp(x)
    return 0.5 + self._decodePoints(x)
################################################################

def Face2StepModel(pointsN, eyeSize, latentSize, contextSize):
  points = L.Input((None, pointsN, 2))
  eyeL = L.Input((None, eyeSize, eyeSize, 1))
  eyeR = L.Input((None, eyeSize, eyeSize, 1))
  context = L.Input((None, contextSize))

  encodedL, encodedR = CRolloutTimesteps(eyeEncoder(), name='Eyes')([eyeL, eyeR, context])
  encodedP = CRolloutTimesteps(FaceMeshEncoder(), name='FaceMesh')([points, context])

  combined = L.Concatenate(-1)([encodedP, encodedL, encodedR, context])
  combined = sMLP(sizes=[256, latentSize], activation='relu')(combined)
  
  return tf.keras.Model(
    inputs={
      'points': points,
      'left eye': eyeL,
      'right eye': eyeR,
      'context': context
    },
    outputs={
      'latent': combined,
      'intermediate': [
        IntermediatePredictor()(x)
        for x in [encodedP, encodedL, encodedR, combined]
      ]
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
  intermediate.append(temporal)
  for i in range(3):
    temporal = CMyTransformerLayer(
      latentSize,
      toQuery=sMLP(sizes=[64, ], activation='relu'),
      toKey=sMLP(sizes=[64, ], activation='relu'),
      useNormalization=True
    )(temporal) + temporal
    intermediate.append(temporal)
    continue
  temporal = sMLP(sizes=[latentSize, latentSize], activation='relu')(temporal)
  temporal = stepsData + CGate(axis=[-1])(temporal)
  
  msk = L.Dense(latentSize, activation='tanh', use_bias=False)(context)
  partial, full = CStackApplySplit(
    sMLP(sizes=[latentSize, latentSize], activation='relu')
  )(stepsData * msk, temporal * msk)
  
  return tf.keras.Model(
    inputs=[stepsDataInput, T, context],
    outputs={
      'latent': full,
      'partial': partial,
      'intermediate': [
        IntermediatePredictor()(x)
        for x in [partial, full] + intermediate
      ]
    }
  )

def Face2LatentModel(pointsN=468, eyeSize=32, steps=None, latentSize=64, contexts=[1, 1, 1]):
  points = L.Input((steps, pointsN, 2))
  eyeL = L.Input((steps, eyeSize, eyeSize, 1))
  eyeR = L.Input((steps, eyeSize, eyeSize, 1))
  ContextID = L.Input((steps, len(contexts)))
  T = L.Input((steps, 1))
  
  ctx = CContextEncoder(dims=32, contexts=contexts)(ContextID)
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
  res['shift'] = L.Dense(2)(
    sMLP(sizes=[32,]*4, activation='relu')(ctx)
  )
  res['context'] = ctx
  res['intermediate'] = stepsData['intermediate'] + res['intermediate']
  
  main = tf.keras.Model(
    inputs={
      'points': points,
      'left eye': eyeL,
      'right eye': eyeR,
      'ContextID': ContextID,
      'time': T
    },
    outputs=res
  )

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
      #'coords': .5 + CDecodePoint(16)(latent),
      'coords': .5 + L.Dense(2)(latent),
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
  X = Face2LatentModel(steps=5)
  X['main'].summary(expand_nested=True)
  # print('-----------------')
  # X['Face2Step'].summary()
  # print('-----------------')
  # X['Step2Latent'].summary()
  # print('-----------------')
#   simpleModel(64).summary()
  model = X['main']  
  pass