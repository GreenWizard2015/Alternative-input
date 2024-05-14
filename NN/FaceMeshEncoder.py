import tensorflow as tf
from NN.Utils import sMLP, CRMLBlock
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
    self._sMLP = sMLP(sizes=[8] * 3, activation='relu', name='FaceMeshEncoder/sMLP-1')
    self._sMLP2 = sMLP(sizes=[latentSize], activation='relu', name='FaceMeshEncoder/sMLP-2')

    self._RML = [
      CRMLBlock(
        mlp=sMLP(
          sizes=[latentSize * 2] * 3,
          activation='relu', name=f'FaceMeshEncoder/RML-{i}/mlp'
        ),
        name=f'FaceMeshEncoder/RML-{i}'
      ) for i in range(5)
    ]
    self._invalidEmbedding = tf.Variable(tf.random.normal((8,)), trainable=True, name='FaceMeshEncoder/invalidEmbedding')
    return

  def call(self, data):
    points = data
    B = tf.shape(points)[0]
    N = tf.shape(points)[1]
    tf.assert_equal(tf.shape(points), (B, N, 2))

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
    encodedPoints = tf.where(validPointsMask, encodedPoints, self._invalidEmbedding)
    M = encodedPoints.shape[-1]
    tf.assert_equal(tf.shape(encodedPoints), (B, N, M))
    
    encodedPoints = tf.reshape(encodedPoints, (B, N * M))
    cond = res = self._sMLP2(encodedPoints)
    for rml in self._RML:
      res = rml([res, cond])
      continue
    return res