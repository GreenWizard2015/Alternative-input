import tensorflow as tf
import time, os
import NN.Utils as NNU
import NN.networks as networks
from Core.Utils import FACE_MESH_INVALID_VALUE

class CAutoencoderTrainer:
  def __init__(self, model='autoencoder', means=None, **kwargs):
    super().__init__()
    self._modelID = model
    self._modelRaw = networks.FaceAutoencoderModel(steps=None, means=means)
    self._model = self._modelRaw['main']
  
    if 'weights' in kwargs:
      self.load(**kwargs['weights'])
    # Define input signatures for TensorFlow graph optimization
    specification = self._modelRaw['inputs specification']
    self._trainStep = tf.function(
      self._trainStep,
      input_signature=[
        { 'clean': specification, 'augmented': specification, },
      ]
    )
    self._eval = tf.function(
      self._eval,
      input_signature=[specification]
    )
    self._compile()
    return
  
  def _compile(self):
    self._model.compile(
      optimizer=NNU.createOptimizer()
    )
    return
  
  def _pointLoss(self, ytrue, ypred):
    ypred = tf.reshape(ypred, tf.shape(ytrue))
    validPointsMask = tf.reduce_all(ytrue != FACE_MESH_INVALID_VALUE, axis=-1, keepdims=True)
    diff = tf.square(ytrue - ypred)
    diff = tf.where(validPointsMask, diff, 0.0)
    return tf.reduce_mean(diff, axis=-1)
  
  def _trainStep(self, Data):
    print('Instantiate _trainStep')
    x = Data
    B = tf.shape(x['clean']['points'])[0]
    x = {k: v for k, v in x.items()}
    x['clean'] = {k: v for k, v in x['clean'].items()}
    x['augmented'] = {k: v for k, v in x['augmented'].items()}
    with tf.GradientTape() as tape:
      predictions = self._model(x['augmented'], training=True)
      loss = self._pointLoss(x['clean']['points'], predictions['points'])
      for name in ['left eye', 'right eye']:
        loss += tf.reduce_mean(
          tf.keras.losses.MSE(x['clean'][name], predictions[name])
        )
    
    gradients = tape.gradient(loss, self._model.trainable_variables)
    self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
    return {'loss': loss}
  
  def fit(self, data):
    t = time.time()
    losses = self._trainStep(data)
    losses = {k: v.numpy() for k, v in losses.items()}
    return {'time': int((time.time() - t) * 1000), 'losses': losses}
  
  def _modelFilename(self, folder, postfix=''):
    postfix = '-' + postfix if postfix else ''
    return os.path.join(folder, '%s-%s%s.h5' % (self._modelID, 'model', postfix))
  
  def save(self, folder=None, postfix='', path=None):
    for k, v in self._modelRaw.items():
      if isinstance(v, tf.keras.Model):
        v.save_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, k, postfix)))
    return
    
  def load(self, folder=None, postfix='', path=None):
    for k, v in self._modelRaw.items():
      if isinstance(v, tf.keras.Model):
        v.load_weights(os.path.join(folder, '%s-%s%s.h5' % (self._modelID, k, postfix)))
    return
  
  def _eval(self, x):
    print('Instantiate _eval')
    B = tf.shape(x['points'])[0]
    x = {k: v for k, v in x.items()}
    predictions = self._model(x, training=False)
    loss = self._pointLoss(x['points'], predictions['points'])
    for name in ['left eye', 'right eye']:
      A = x[name]
      B = tf.reshape(predictions[name], tf.shape(A))
      tf.assert_equal(tf.shape(A), tf.shape(B))
      loss += tf.reduce_mean(
        tf.keras.losses.MSE(A, B)
      )
    return tf.reduce_mean(loss)

  def eval(self, data):
    loss = self._eval(data)
    return loss.numpy()
