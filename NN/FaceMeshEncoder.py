import tensorflow as tf
from NN.Utils import sMLP
from NN.CCoordsEncodingLayer import CCoordsEncodingLayer

class FaceMeshEncoder(tf.keras.Model):
  def __init__(self, latentSize, **kwargs):
    super().__init__(**kwargs)
    self._encodedPoints = CCoordsEncodingLayer(N=30, name='FaceMeshEncoder/coords')
    self._sMLP = sMLP(sizes=[8, 2], activation='relu', name='FaceMeshEncoder/sMLP-1')

    self._contextMLP = sMLP(sizes=[latentSize, latentSize], name='FaceMeshEncoder/contextMLP')
    self._sMLP2 = sMLP(sizes=[latentSize, latentSize], name='FaceMeshEncoder/sMLP-2')

    self._flatten = tf.keras.layers.Flatten(name='FaceMeshEncoder/flatten')
    return

  def call(self, data):
    points, context = data
    B = tf.shape(points)[0]
    tf.assert_equal(tf.shape(context)[0], B)

    validPointsMask = tf.reduce_all(0.0 <= points, axis=-1)[..., None]
    encodedPoints = self._encodedPoints(points)
    encodedPoints = tf.where(validPointsMask, encodedPoints, 0.0)    
    encodedPoints = self._sMLP(encodedPoints)
    encodedPoints = self._flatten(encodedPoints)

    pts = tf.concat([encodedPoints, self._contextMLP(context)], axis=-1)
    return self._sMLP2(pts)