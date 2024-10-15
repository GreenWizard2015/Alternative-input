from Core.Utils import setupGPU, FACE_MESH_POINTS
setupGPU() # dirty hack to setup GPU memory limit on startup

import tensorflow as tf
import tensorflow.keras.layers as L
from NN.CCoordsEncodingLayer import CCoordsEncodingLayer
from NN.Utils import *
from NN.EyeEncoder import eyeEncoder
from NN.FaceMeshEncoder import FaceMeshEncoder
from NN.LagrangianInterpolation import lagrange_interpolation

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
    'points': tf.TensorSpec(shape=(None, None, FACE_MESH_POINTS, 2), dtype=tf.float32),
    'left eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'right eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'time': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
    'userId': tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
    'placeId': tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
    'screenId': tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
  }

def Face2LatentModel(
  pointsN=FACE_MESH_POINTS, eyeSize=32, steps=None, latentSize=64,
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
  
##########################

def _InpaintingInputSpec():
  return {
    'points': tf.TensorSpec(shape=(None, None, FACE_MESH_POINTS, 2), dtype=tf.float32),
    'left eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'right eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'time': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
    'userId': tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
    'placeId': tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
    'screenId': tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
    'target': tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
  }

def InpaintingEncoderModel(latentSize, embeddings, steps=5, pointsN=FACE_MESH_POINTS, eyeSize=32, KP=5):
  points = L.Input((steps, pointsN, 2))
  eyeL = L.Input((steps, eyeSize, eyeSize, 1))
  eyeR = L.Input((steps, eyeSize, eyeSize, 1))
  T = L.Input((steps, 1)) # accumulative time
  target = L.Input((steps, 2))
  userIdEmb = L.Input((steps, embeddings['size']))
  placeIdEmb = L.Input((steps, embeddings['size']))
  screenIdEmb = L.Input((steps, embeddings['size']))

  emb = L.Concatenate(-1)([userIdEmb, placeIdEmb, screenIdEmb])

  Face2Step = Face2StepModel(pointsN, eyeSize, latentSize, embeddingsSize=emb.shape[-1])
  stepsData = Face2Step({
    'embeddings': emb,
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
  })

  diffT = T[:, 1:] - T[:, :-1]
  diffT = L.Concatenate(-2)([tf.zeros_like(diffT[:, :1]), diffT])
  combinedT = L.Concatenate(-1)([T, diffT])
  encodedT = CRolloutTimesteps(CCoordsEncodingLayer(32), name='Time')(combinedT[..., None, :])[..., 0, :]

  latent = stepsData['latent']
  # add time encoding and target position
  targetEncoded = CRolloutTimesteps(CCoordsEncodingLayer(32), name='Target')(target[..., None, :])[..., 0, :]
  latent = L.Concatenate(-1)([latent, encodedT, targetEncoded])
  # flatten the latent
  latent = L.Reshape((-1,))(latent)

  # compress the latent
  latent_N = latent.shape[-1]
  sizes = []
  for i in range(1, 4):
    for _ in range(i):
       sizes.append(max(latent_N // i, latentSize))
       
  sizes.append(latentSize)
  latent = sMLP(sizes=sizes, activation='relu', name='Compress')(latent)
  keyT = tf.linspace(0.0, 1.0, KP)[None, :]

  # keyT shape: (B, KP, 1)
  def transformKeyT(x):
    t, x = x
    B = tf.shape(x)[0]
    return tf.tile(t, (B, 1))[..., None]
  keyT = L.Lambda(transformKeyT)([keyT, latent])
  # keyT shape: (B, KP, 1)
  maxT = T[:, -1, None]
  keyT = L.Concatenate(-1)([keyT, maxT * keyT]) # fractional time and absolute time
  encodedKeyT = CRolloutTimesteps(CCoordsEncodingLayer(32), name='KeyTime')(keyT[..., None, :])[..., 0, :]

  def combineKeys(x):
    latent, keyT = x
    latent = tf.tile(latent[..., None, :], (1, KP, 1))
    return L.Concatenate(-1)([latent, keyT])
  latent = L.Lambda(combineKeys)([latent, encodedKeyT])

  latent = sMLP(sizes=[latentSize] * 3, activation='relu', name='CombineKeys')(latent)

  main = tf.keras.Model(
    inputs={
      'points': points,
      'left eye': eyeL,
      'right eye': eyeR,
      'time': T,
      'target': target,
      'userId': userIdEmb,
      'placeId': placeIdEmb,
      'screenId': screenIdEmb,
    },
    outputs={
      'latent': latent,
    }
  )
  return main
 
def InpaintingDecoderModel(latentSize, embeddings, pointsN=FACE_MESH_POINTS, eyeSize=32, KP=5):
  latentKeyPoints = L.Input((KP, latentSize))
  T = L.Input((None, 1))
  userIdEmb = L.Input((embeddings['size']))
  placeIdEmb = L.Input((embeddings['size']))
  screenIdEmb = L.Input((embeddings['size']))

  emb = L.Concatenate(-1)([userIdEmb, placeIdEmb, screenIdEmb])[..., None, :]
  # emb shape: (B, 1, 3 * embSize) 
  def interpolateKeys(x):
    latents, T = x
    B = tf.shape(latents)[0]
    keyT = tf.linspace(0.0, 1.0, KP)[None, :]
    keyT = tf.tile(keyT, (B, 1))
    return lagrange_interpolation(x_values=keyT, y_values=latents, x_targets=T[..., 0])
  latents = L.Lambda(interpolateKeys, name='InterpolateKeys')([latentKeyPoints, T])
  # latents shape: (B, N, latentSize)
  def transformLatents(x):
    latents, emb = x
    N = tf.shape(latents)[1]
    emb = tf.tile(emb, (1, N, 1)) # (B, 1, 3 * embSize) -> (B, N, 3 * embSize)
    return L.Concatenate(-1)([latents, emb])
  latents = L.Lambda(transformLatents, name='CombineEmb')([latents, emb])
  # process the latents
  latents = sMLP(sizes=[latentSize] * 3, activation='relu', name='CombineEmb/MLP')(latents)
  # decode the latents to the face points (FACE_MESH_POINTS, 2), two eyes (32, 32, 2) and the target (2) 
  target = IntermediatePredictor(shift=0.5)(latents)
  # two eyes
  eyesN = eyeSize * eyeSize
  eyes = sMLP(sizes=[eyesN] * 2, activation='relu')(latents)
  eyes = L.Dense(eyesN * 2)(eyes)
  eyes = L.Reshape((-1, eyeSize, eyeSize, 2))(eyes)
  # face points
  face = sMLP(sizes=[pointsN] * 2, activation='relu')(latents)
  face = L.Dense(pointsN * 2)(face)
  face = L.Reshape((-1, pointsN, 2))(face)

  model = tf.keras.Model(
    inputs={
      'keyPoints': latentKeyPoints,
      'time': T,
      'userId': userIdEmb,
      'placeId': placeIdEmb,
      'screenId': screenIdEmb,
    },
    outputs={
      'target': target,
      'left eye': eyes[:, :, 0],
      'right eye': eyes[:, :, 1],
      'face': face,
    }
  )
  return model


if __name__ == '__main__':
  # X = InpaintingEncoderModel(latentSize=256, embeddings={
  #   'size': 64
  # })
  X = InpaintingDecoderModel(latentSize=256, embeddings={
    'size': 64
  })
  X.summary(expand_nested=False)

  # X = Face2LatentModel(steps=5, latentSize=64,
  #   embeddings={
  #     'userId': 1, 'placeId': 1, 'screenId': 1, 'size': 64
  #   }
  # )
  # X['main'].summary(expand_nested=True)
  # X['Face2Step'].summary(expand_nested=False)
  # X['Step2Latent'].summary(expand_nested=False)
  # print(X['main'].outputs)
  # pass