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

def Face2StepModel(pointsN, eyeSize, latentSize):
  points = L.Input((None, pointsN, 2))
  eyeL = L.Input((None, eyeSize, eyeSize, 1))
  eyeR = L.Input((None, eyeSize, eyeSize, 1))

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
    combined = CResidualMultiplicativeLayer(name='F2S/ResMul-%d' % i)([
      combined,
      sMLP(sizes=[latentSize] * 1, activation='relu', name='F2S/MLP-%d' % i)(
        L.Concatenate(-1)([combined, encodedP, EFeat])
      )
    ])
     # save intermediate output
    intermediate['F2S/encEyes-%d' % i] = EFeat
    intermediate['F2S/combined-%d' % i] = combined
    continue
  
  return tf.keras.Model(
    inputs={
      'points': points,
      'left eye': eyeL,
      'right eye': eyeR,
    },
    outputs={
      'latent': combined,
      'intermediate': intermediate
    }
  )

def Step2LatentModel(latentSize):
  stepsDataInput = L.Input((None, latentSize))
  T = L.Input((None, 1))

  stepsData = stepsDataInput
  intermediate = {}
  
  encodedT = CTimeEncoderLayer()(T)
  temporal = sMLP(sizes=[latentSize] * 1, activation='relu')(
    L.Concatenate(-1)([stepsData, encodedT])
  )
  temporal = CResidualMultiplicativeLayer()([stepsData, temporal])
  intermediate['S2L/enc0'] = temporal
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  for blockId in range(3):
    temp = L.Concatenate(-1)([temporal, encodedT])
    for _ in range(1):
      temp = L.LSTM(latentSize, return_sequences=True)(temp)
    temp = sMLP(sizes=[latentSize] * 1, activation='relu')(
      L.Concatenate(-1)([temporal, temp])
    )
    temporal = CResidualMultiplicativeLayer()([temporal, temp])
    intermediate['S2L/ResLSTM-%d' % blockId] = temporal
    continue
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  latent = sMLP(sizes=[latentSize] * 1, activation='relu')(
    L.Concatenate(-1)([stepsData, temporal, encodedT, encodedT])
  )
  latent = CResidualMultiplicativeLayer()([stepsData, latent])
  return tf.keras.Model(
    inputs=[stepsDataInput, T],
    outputs={
      'latent': latent,
      'intermediate': intermediate
    }
  )

def Step2FaceModel(latentSize, means):
  '''
  Take encoded latent and reconstruct the face
  '''
  stepsDataInput = L.Input((None, latentSize))

  stepsData = stepsDataInput
  splits = [latentSize // 3, latentSize // 3, latentSize - 2 * (latentSize // 3)]
  pointsData = stepsData[:, :, :splits[0]]
  eyesLData = stepsData[:, :, splits[0]:splits[0] + splits[1]]
  eyesRData = stepsData[:, :, splits[0] + splits[1]:]
  
  sizes = [latentSize * i for i in range(5)] + [latentSize * 4] * 4
  # points branch
  faceCenter = means['points'].mean(axis=0, keepdims=True)
  meanShift = faceCenter - means['points']
  faceCenter = tf.Variable(faceCenter, trainable=False, name='faceCenter')
  meanShift = tf.Variable(meanShift, trainable=False, name='meanShift')

  pointsMean = sMLP(sizes=sizes, activation='relu')(pointsData)
  pointsMean = L.Dense(2)(pointsMean)
  pointsMean = L.Reshape((-1, 1, 2))(pointsMean)

  points = sMLP(sizes=sizes, activation='relu')(
    pointsData
  )
  points = L.Dense(478 * 2)(points)
  points = L.Reshape((-1, 478, 2))(points)
  points = pointsMean + points + meanShift.reshape((1, 1, 478, 2))

  # eyes branch
  eyeL = sMLP(sizes=[latentSize * i for i in range(5)] * 4, activation='relu')(eyesLData)
  eyeL = L.Dense(32 * 32)(eyeL)
  eyeL = L.Reshape((-1, 32, 32))(eyeL)
  eyeLMean = tf.Variable(
    means['left eye'].reshape((1, 1, 32, 32)),
    trainable=False,
    name='leftEyeMean'
  )
  eyeL = means['left eye'].reshape((1, 1, 32, 32)) + eyeL

  eyeR = sMLP(sizes=[latentSize * i for i in range(5)] * 4, activation='relu')(eyesRData)
  eyeR = L.Dense(32 * 32)(eyeR)
  eyeR = L.Reshape((-1, 32, 32))(eyeR)
  eyeRMean = tf.Variable(
    means['right eye'].reshape((1, 1, 32, 32)),
    trainable=False,
    name='rightEyeMean'
  )
  eyeR = eyeRMean + eyeR

  return tf.keras.Model(
    inputs={
      'stepsDataInput': stepsDataInput,
    },
    outputs={
      'points': points,
      'left eye': eyeL,
      'right eye': eyeR,
    }
  )

def FaceAutoencoderModel(pointsN=478, eyeSize=32, steps=None, latentSize=64, means=None):
  # Inputs
  points = L.Input((steps, pointsN, 2), name='input_points')
  eyeL = L.Input((steps, eyeSize, eyeSize, 1), name='input_left_eye')
  eyeR = L.Input((steps, eyeSize, eyeSize, 1), name='input_right_eye')

  # Encoder: Encode face and eyes to a latent space
  face2StepModel = Face2StepModel(pointsN, eyeSize, latentSize)
  encoded_outputs = face2StepModel({
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
  })
  latent = encoded_outputs['latent']

  # Decoder: Decode from latent space back to face and eyes
  step2FaceModel = Step2FaceModel(latentSize=latentSize, means=means)
  decoded = step2FaceModel({
    'stepsDataInput': latent,
  })

  # Define the full model
  autoencoder = tf.keras.Model(
    inputs={
      'points': points,
      'left eye': eyeL,
      'right eye': eyeR,
    },
    outputs=decoded,
    name='face_autoencoder'
  )

  return {
    'encoder': face2StepModel,
    'decoder': step2FaceModel,
    'main': autoencoder,
    'inputs specification': {
      'points': tf.TensorSpec(shape=(None, None, 478, 2), dtype=tf.float32),
      'left eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
      'right eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    }
  }

def _InputSpec():
  return {
    'points': tf.TensorSpec(shape=(None, None, 478, 2), dtype=tf.float32),
    'left eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'right eye': tf.TensorSpec(shape=(None, None, 32, 32), dtype=tf.float32),
    'time': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
  }

def Face2LatentModel(pointsN=478, eyeSize=32, steps=None, latentSize=64):
  points = L.Input((steps, pointsN, 2))
  eyeL = L.Input((steps, eyeSize, eyeSize, 1))
  eyeR = L.Input((steps, eyeSize, eyeSize, 1))
  T = L.Input((steps, 1))

  Face2Step = Face2StepModel(pointsN, eyeSize, latentSize)
  Step2Latent = Step2LatentModel(latentSize)

  stepsData = Face2Step({
    'points': points,
    'left eye': eyeL,
    'right eye': eyeR,
  })
  
  res = Step2Latent([stepsData['latent'], T])
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
  autoencoder = FaceAutoencoderModel(latentSize=64, means={
    'points': np.zeros((478, 2), np.float32),
    'left eye': np.zeros((32, 32), np.float32),
    'right eye': np.zeros((32, 32), np.float32),
  })['main']
  autoencoder.summary(expand_nested=True)

  # X = Face2LatentModel(steps=5, latentSize=64)
  # X['main'].summary(expand_nested=True)
  # X['Face2Step'].summary(expand_nested=False)
  # X['Step2Latent'].summary(expand_nested=False)
  # print(X['main'].outputs)
  pass