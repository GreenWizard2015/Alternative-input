from Core.Utils import FACE_MESH_INVALID_VALUE

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
      tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(7,), dtype=tf.float32),
    # userId, placeId, screenId
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
  ]
)
def toTensor(data, params, userId, placeId, screenId):
  print('Instantiate CDataSampler_utils.toTensor')
  (
    pointsNoise, pointsDropout,
    eyesAdditiveNoise, eyesDropout,
    brightnessFactor, lightBlobFactor,
    timesteps
  ) = tf.unstack(params)
  timesteps = tf.cast(timesteps, tf.int32)
  points, imgA, imgB, T = data
  N = tf.shape(points)[0]
  imgA = tf.cast(imgA, tf.float32) / 255.
  imgB = tf.cast(imgB, tf.float32) / 255.
  tf.assert_equal(tf.shape(imgA), (N, 48, 48))
  tf.assert_equal(tf.shape(imgA), tf.shape(imgB))
  userId = tf.fill((N, 1), userId)
  placeId = tf.fill((N, 1), placeId)
  screenId = tf.fill((N, 1), screenId)
  
  reshape = lambda x: tf.reshape(
    x,
    tf.concat([(N // timesteps, timesteps), tf.shape(x)[1:]], axis=-1)
  )
  # apply center crop
  fraction = 32.0 / 48.0
  pos = tf.constant(
    [[0.5 - fraction / 2, 0.5 - fraction / 2, 0.5 + fraction / 2, 0.5 + fraction / 2]],
    dtype=tf.float32
  )
  pos = tf.tile(pos, [N, 1])
  withCrop = lambda x: tf.image.crop_and_resize(
    x[..., None],
    boxes=pos,
    box_indices=tf.range(N), crop_size=(32, 32),
  )[..., 0]

  clean = {
    'time': reshape(T),
    'points': reshape(points),
    'left eye': reshape(withCrop(imgA)),
    'right eye': reshape(withCrop(imgB)),
    'userId': reshape(userId),
    'placeId': reshape(placeId),
    'screenId': reshape(screenId),
  }
  ##########################
  # random crop 32x32 eyes
  fraction = 32.0 / 48.0
  pos = tf.random.uniform((N, 2), minval=0.0, maxval=2.0 * fraction)
  boxes = tf.concat([pos, pos + fraction], axis=-1)
  tf.assert_equal(tf.shape(boxes), (N, 4))
  imgA = tf.image.crop_and_resize(
    imgA[..., None],
    boxes=boxes,
    box_indices=tf.range(N), crop_size=(32, 32),
  )[..., 0]
  imgB = tf.image.crop_and_resize(
    imgB[..., None],
    boxes=boxes,
    box_indices=tf.range(N), crop_size=(32, 32),
  )[..., 0]
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
  validPointsMask = tf.reduce_all(FACE_MESH_INVALID_VALUE != points, axis=-1, keepdims=True)
  if 0.0 < pointsNoise:
    points += tf.random.normal(tf.shape(points), stddev=pointsNoise)

  # dropouts
  if 0.0 < pointsDropout:
    mask = tf.random.uniform((N, tf.shape(points)[1])) < pointsDropout
    points = tf.where(mask[:, :, None], FACE_MESH_INVALID_VALUE, points)

  points = tf.where(validPointsMask, points, FACE_MESH_INVALID_VALUE)
  ##########################
  return {
    'augmented': {
      'time': reshape(T),
      'points': reshape(points),
      'left eye': reshape(imgA),
      'right eye': reshape(imgB),
      'userId': reshape(userId),
      'placeId': reshape(placeId),
      'screenId': reshape(screenId),
    },
    'clean': clean,
  }
