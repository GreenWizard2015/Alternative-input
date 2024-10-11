import tensorflow as tf
import time
import NN.Utils as NNU
from Core.CModelWrapper import CModelWrapper

class CModelTrainer(CModelWrapper):
  def __init__(self, timesteps, model='simple', **kwargs):
    super().__init__(timesteps, model=model, **kwargs)
    self.compile()
    # add signatures to help tensorflow optimize the graph
    specification = self._modelRaw['inputs specification']
    self._trainStep = tf.function(
      self._trainStep,
      input_signature=[
        (
          { 'clean': specification, 'augmented': specification, },
          ( tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32), )
        )
      ]
    )
    self._eval = tf.function(
      self._eval,
      input_signature=[(
        specification,
        ( tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32), )
      )]
    )
    return
  
  def compile(self):
    self._optimizer = NNU.createOptimizer()
    return
  
  def _pointLoss(self, ytrue, ypred):
    # pseudo huber loss
    delta = 0.01
    tf.assert_equal(tf.shape(ytrue), tf.shape(ypred))
    diff = tf.square(ytrue - ypred)
    loss = tf.sqrt(diff + delta ** 2) - delta
    tf.assert_equal(tf.shape(loss), tf.shape(ytrue))
    return tf.reduce_mean(loss, axis=-1)
    
  def _trainOn(self, data, y_list):
      def calculate_loss(predictions):
        # select the smallest loss from the list of suggested points
        losses = []
        for y in y_list:
          loss = self._pointLoss(y, predictions)[..., None]
          losses.append(loss)
          continue
        losses = tf.concat(losses, axis=-1)
        shp = tf.shape(y_list[0])
        tf.assert_equal(tf.shape(losses), tf.concat([shp[:-1], [len(y_list)]], axis=0))
        losses = tf.reduce_min(losses, axis=-1)
        tf.assert_equal(tf.shape(losses), shp[:-1])
        return tf.reduce_mean(losses)
      
      data = self._replaceByEmbeddings(data)
      predictions = self._model(data, training=True)
      intermediate = predictions['intermediate']
      finalPredictions = predictions['result']
      losses = {}
      losses['final'] = calculate_loss(finalPredictions)
      for name, encoder in self._intermediateEncoders.items():
        latent = intermediate[name]
        pts = encoder(latent, training=True)
        loss = calculate_loss(pts)
        losses['loss-%s' % name] = tf.reduce_mean(loss)
        continue
      return losses, tf.stop_gradient(finalPredictions)
    
  def _trainStep(self, Data):
    print('Instantiate _trainStep')
    ###############
    x, (y, ) = Data
    y = y[..., 0, :]
    losses = {}
    with tf.GradientTape() as tape:
      lossesClean, y_clean = self._trainOn(x['clean'], [y])
      # ensure that the augmentations are not affect predictions
      lossesAugmented, _ = self._trainOn(x['augmented'], [y, y_clean])
      assert lossesClean.keys() == lossesAugmented.keys(), 'Losses keys mismatch'
      # combine losses
      losses = {k: lossesClean[k] + lossesAugmented[k] for k in lossesClean.keys()}
      # calculate total loss and final loss
      losses['total-clean'] = sum(lossesClean.values())
      losses['total-augmented'] = sum(lossesAugmented.values())
      losses['loss'] = loss = sum([losses['total-clean'], losses['total-augmented']])
  
    self._optimizer.minimize(loss, tape.watched_variables(), tape=tape)
    ###############
    return losses

  def fit(self, data):
    t = time.time()
    losses = self._trainStep(data)
    losses = {k: v.numpy() for k, v in losses.items()}
    return {'time': int((time.time() - t) * 1000), 'losses': losses}
  
  def _eval(self, xy):
    print('Instantiate _eval')
    x, (y,) = xy
    x = self._replaceByEmbeddings(x)
    y = y[:, :, 0]
    predictions = self._model(x, training=False)
    points = predictions['result'][:, :, :]
    tf.assert_equal(tf.shape(points), tf.shape(y))

    loss = self._pointLoss(y, points)
    tf.assert_equal(tf.shape(loss), tf.shape(y)[:2])
    _, dist = NNU.normVec(points - y)
    return loss, points, dist

  def eval(self, data):
    loss, sampled, dist = self._eval(data)
    return loss.numpy(), sampled.numpy(), dist.numpy()