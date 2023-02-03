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
      # 'latent': combined
      'latent': CQuantizeLayer()(combined)
    }
  )

def Step2LatentModel(latentSize, contextSize):
  stepsDataInput = L.Input((None, latentSize))
  context = L.Input((None, contextSize))
  T = L.Input((None, 1))

#   msk = L.Dense(latentSize, activation='tanh', use_bias=False)(context)
  stepsData = stepsDataInput# * msk
  
  encodedT = CTimeEncoderLayer()(T)
  temporal = sMLP(sizes=[latentSize, latentSize], activation='relu')(
    L.Concatenate(-1)([stepsData, encodedT, context])
  )
  # temporal = L.LSTM(temporal.shape[-1], return_sequences=True)(temporal)
  # use transformer instead of LSTM
  temporal = CMyTransformerLayer(
    latentSize,
    # toQuery=sMLP(sizes=[64, 8], activation='relu'),
    # toKey=sMLP(sizes=[64, 8], activation='relu')
  )(temporal)
  temporal = sMLP(sizes=[latentSize, latentSize], activation='relu')(temporal)
  temporal = stepsData + CGate(axis=[-1])(temporal)
  
  msk = L.Dense(latentSize, activation='tanh', use_bias=False)(context)
  partial, full = CStackApplySplit(
    sMLP(sizes=[latentSize, latentSize], activation='relu')
  )(stepsData + msk, temporal + msk)
  
  return tf.keras.Model(
    inputs=[stepsDataInput, T, context],
    outputs={
      # 'latent': full,
      'latent': CQuantizeLayer()(full),
      'partial': CQuantizeLayer()(partial)
    }
  )

class CContextEncoder(tf.keras.layers.Layer):
  def __init__(self, dims, contexts, **kwargs):
    super().__init__(**kwargs)
    self._dims = dims
    self._embeddings = [
      tf.keras.layers.Embedding(N, dims, name='%s/embeddings-%d' % (self.name, i))
      for i, N in enumerate(contexts)
    ]
    self._encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(dims, activation='relu'),
      tf.keras.layers.Dense(dims, activation='relu'),
    ], name='%s/encoder' % self.name)
    self._dropoutRate = 0.1
    self._dropout = tf.keras.layers.Dropout(0.1, name='%s/dropout' % self.name)
    return

  @property
  def embeddings(self):
    return self._embeddings
  
  def call(self, contextID, training=None):
    B, N, C = [tf.shape(contextID)[i] for i in range(3)]
    tf.assert_equal(C, len(self._embeddings))

    embeddings = []
    for i, emb in enumerate(self._embeddings):
      emb = self._dropout(emb(contextID[..., i]))
      shp = tf.shape(emb)
      tf.assert_equal(shp, (B, N, shp[-1]))
      embeddings.append(emb)
      continue

    if training: # apply dropout in training mode
      for i, emb in enumerate(embeddings):
        # mask out each embedding with dropout rate
        mask = tf.random.uniform((B, N, 1)) < self._dropoutRate
        emb = tf.where(mask, 0.0, emb)
        # mask out gradients
        # embS = tf.stop_gradient(emb)
        # mask = tf.random.uniform((B, N, 1)) < 0.1
        # emb = tf.where(mask, embS, emb)
        embeddings[i] = emb
        continue
      pass

    embeddings = tf.concat(embeddings, axis=-1)
    return self._encoder(embeddings)

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
  })['latent']
  
  res = Step2Latent([stepsData, T, ctx])
  res['shift'] = L.Dense(2)(
    sMLP(sizes=[32,]*4, activation='relu')(ctx)
  )
  res['context'] = ctx
  
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
  # find in model the first layer CContextEncoder
  contextEncoder = model.get_layer('c_context_encoder').embeddings

  test = contextEncoder[0](np.array([[1, 2, 3]])).numpy()
  print(test.sum())

  pass