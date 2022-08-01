import tensorflow as tf
import tensorflow.keras.layers as L
from NN.Utils import sMLP
from NN.CCoordsEncodingLayer import CCoordsEncodingLayer

def FaceMeshEncoder(pointsN=468):
  points = L.Input((pointsN, 2))
  
  encodedPoints = CCoordsEncodingLayer(N=30, name='FaceMeshEncoder/coords')(points)
  encodedPoints = L.Lambda(
    lambda x: x[0] * tf.cast(tf.reduce_all(0.0 <= x[1], axis=-1), tf.float32)[..., None]
  )([encodedPoints, points])
  
  encodedPoints = sMLP(sizes=[16, 8, 4, 1], activation='relu')(encodedPoints)
  pts = L.Flatten()(encodedPoints)
  res = sMLP(sizes=[256, ])(pts)
  
  return tf.keras.Model(
    inputs=[points],
    outputs=[res],
    name='FaceMeshEncoder'
  )
