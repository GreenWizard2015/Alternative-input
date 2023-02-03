import numpy as np
def gaussian(x, mu, sig):
  return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

import tensorflow as tf
import tensorflow_probability as tfp

def get_gaussian(mu, tril, HW):
  B = tf.shape(mu)[0]
  mvn = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=tril)
  xy = tf.linspace(0., 1., HW)
  # meshgrid as a list of [x,y] coordinates
  coords = tf.reshape(tf.stack(tf.meshgrid(xy, xy),axis=-1),(-1,2))
  
  # prob at mu is biggest over the whole space
  muProb = mvn.prob(mu)
  gauss = mvn.prob(coords[:, None, :])
  tf.assert_equal(tf.shape(gauss), (HW*HW, B))
  gauss = tf.transpose(gauss, (1, 0))
  gauss = tf.reshape(gauss, (B, HW, HW))
  return tf.math.divide_no_nan(
    tf.maximum(gauss, 0.0),
    muProb[:, None, None]
  )

def addLightBlob(imgA, imgB, brightness, shared):
  N = tf.shape(imgA)[0]
  HW = tf.shape(imgA)[1]
  def makeBlobs():
    # randomly sample gaussian mu and scale
    lightMu = tf.random.uniform((N, 2), minval=-0.1, maxval=1.1)
    lightScale = tf.random.uniform((N, 2, 2), minval=0.01, maxval=0.5)
    light = get_gaussian(lightMu, lightScale, HW)
    tf.assert_equal(tf.shape(light), tf.shape(imgA))
    tf.assert_equal(tf.shape(light), tf.shape(imgB))
    lightB = tf.reshape(brightness, (N, 1, 1))
    light = 1.0 + light * lightB
    return light
  
  lightA = makeBlobs()
  lightB = lightA if shared else makeBlobs()
  return(
    tf.clip_by_value(imgA * lightA, 0., 1.),
    tf.clip_by_value(imgB * lightB, 0., 1.),
  )
  
@tf.function(
  input_signature=[
    (
      tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
      tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(7,), dtype=tf.float32),
  ]
)
def toTensor(data, params):
  print('Instantiate CDataSampler_utils.toTensor')
  (
    pointsDropout, pointsNoise, 
    eyesAdditiveNoise, eyesDropout,
    brightnessFactor, lightBlobFactor,
    timesteps
  ) = tf.unstack(params)
  timesteps = tf.cast(timesteps, tf.int32)
  points, imgA, imgB, ContextID, T = data
  N = tf.shape(points)[0]
  imgA = tf.cast(imgA, tf.float32) / 255.
  imgB = tf.cast(imgB, tf.float32) / 255.
  
  reshape = lambda x: tf.reshape(
    x,
    tf.concat([(N // timesteps, timesteps), tf.shape(x)[1:]], axis=-1)
  )
  # assert that all timesteps share the same ContextID
  ctx = reshape(ContextID)
  tf.assert_equal(tf.reduce_min(ctx, axis=-2), tf.reduce_max(ctx, axis=-2))
  clean = {
    'time': reshape(T),
    'points': reshape(points),
    'left eye': reshape(imgA),
    'right eye': reshape(imgB),
    'ContextID': reshape(ContextID),
  }
  ##########################
  def clip(x): return tf.clip_by_value(x, 0., 1.)

  def sampleBrightness(a, b, mid=1.0):
    # first sample from truncated normal
    TN = tf.random.truncated_normal((N,), mean=0.0, stddev=0.5) # range -1 to 1
    # then transform values [-1, 0] to [a, mid] and [0, 1] to [mid, b]
    return tf.where(TN < 0.0, a + (mid - a) * (TN + 1.0), mid + (b - mid) * TN)
    
  # random global brightness
  if 0.0 < brightnessFactor:
    # shared brightness
    brightness = sampleBrightness(1.0 / brightnessFactor, brightnessFactor)[:, None, None]
    aug = lambda x: clip(x * brightness)
    imgA = aug(imgA)
    imgB = aug(imgB)

  if 0.0 < lightBlobFactor:
    LightBlobPower = sampleBrightness(1.0 / lightBlobFactor, lightBlobFactor)
    imgA, imgB = addLightBlob(imgA, imgB, LightBlobPower, shared=False)
  ##########################
  if 0.0 < eyesAdditiveNoise:
    aug = lambda x: clip(x + tf.random.normal(tf.shape(x), stddev=eyesAdditiveNoise))
    imgA = aug(imgA)
    imgB = aug(imgB)
    pass
  ##########################
  if 0.0 < eyesDropout:
    mask = tf.random.uniform((N,)) < eyesDropout
    maskA = 0.5 < tf.random.uniform((N,))
    maskB = tf.logical_not(maskA)
    imgA = tf.where(tf.logical_and(mask, maskA)[:, None, None], 0.0, imgA)
    imgB = tf.where(tf.logical_and(mask, maskB)[:, None, None], 0.0, imgB)
  ##########################
  validPointsMask = tf.reduce_all(-1.0 < points, axis=-1, keepdims=True)
  if 0.0 < pointsDropout:
    mask = tf.random.uniform(tf.shape(points)[:-1])[..., None] < pointsDropout
    points = tf.where(mask, -1.0, points)
  
  if 0.0 < pointsNoise:
    points += tf.random.normal(tf.shape(points), stddev=pointsNoise)

  points = tf.where(validPointsMask, points, -1.0)
  ##########################
  return {
    'augmented': {
      'time': reshape(T),
      'points': reshape(points),
      'left eye': reshape(imgA),
      'right eye': reshape(imgB),
      'ContextID': reshape(ContextID),
    },
    'clean': clean,
  }
