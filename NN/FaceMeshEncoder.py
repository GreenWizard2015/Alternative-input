import tensorflow as tf
from NN.Utils import sMLP
from NN.CCoordsEncodingLayer import CCoordsEncodingLayer
from Core.Utils import FACE_MESH_INVALID_VALUE

class FaceMeshEncoder(tf.keras.Model):
  def __init__(self, latentSize, **kwargs):
    super().__init__(**kwargs)
    self._encodedPoints = CCoordsEncodingLayer(
      N=30,
      sharedTransformation=True, # TODO: compare with non-shared transformation
      name='FaceMeshEncoder/coords'
    )
    self._sMLP = sMLP(sizes=[8, 2], activation='relu', name='FaceMeshEncoder/sMLP-1')
    self._dr = tf.keras.layers.SpatialDropout1D(0.1)

    self._contextMLP = sMLP(sizes=[latentSize], activation='relu', name='FaceMeshEncoder/contextMLP')
    self._sMLP2 = sMLP(sizes=[latentSize] * 3, activation='relu', name='FaceMeshEncoder/sMLP-2')
    return

  def call(self, data):
    points, context = data
    B = tf.shape(points)[0]
    N = tf.shape(points)[1]
    tf.assert_equal(tf.shape(points), (B, N, 2))
    tf.assert_equal(tf.shape(context)[0], B)

    validPointsMask = tf.reduce_all(FACE_MESH_INVALID_VALUE != points, axis=-1)[..., None]
    # append to points normalized indices
    indices = tf.range(N, dtype=tf.float32)[None, :, None]
    indices = indices / tf.cast(N, tf.float32)
    indices = tf.tile(indices, (B, 1, 1))
    tf.assert_equal(tf.shape(indices), (B, N, 1))
    points = tf.concat([points, indices], axis=-1)
    tf.assert_equal(tf.shape(points), (B, N, 3))

    encodedPoints = self._encodedPoints(points)
    encodedPoints = self._sMLP(encodedPoints)
    encodedPoints = self._dr(encodedPoints)
    encodedPoints = tf.where(validPointsMask, encodedPoints, 0.0)
    M = encodedPoints.shape[-1]
    tf.assert_equal(tf.shape(encodedPoints), (B, N, M))
    
    encodedPoints = tf.reshape(encodedPoints, (B, N * M))
    pts = tf.concat([encodedPoints, self._contextMLP(context)], axis=-1)
    return self._sMLP2(pts)