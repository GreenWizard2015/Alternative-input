import tensorflow as tf
import time
import NN.Utils as NNU
from Core.CModelWrapper import CModelWrapper

class CModelTrainer(CModelWrapper):
  def __init__(self, timesteps, model='simple', **kwargs):
    super().__init__(timesteps, model=model, **kwargs)
    self._compile()
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
  
  def _compile(self):
    self._model.compile(optimizer=NNU.createOptimizer())
    return
  
  def _pointLoss(self, ytrue, ypred):
    # pseudo huber loss
    delta = 0.01
    tf.assert_equal(tf.shape(ytrue), tf.shape(ypred))
    diff = tf.square(ytrue - ypred)
    loss = tf.sqrt(diff + delta ** 2) - delta
    tf.assert_equal(tf.shape(loss), tf.shape(ytrue))
    return tf.reduce_mean(loss, axis=-1)
    
  def _trainStep(self, Data):
    print('Instantiate _trainStep')
    ###############
    x, (y, ) = Data
    y = y[..., 0, :]
    losses = {}
    parts = list(self._embeddings.values()) + [self._model]
    TV = sum([p.trainable_variables for p in parts], [])
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(TV)
      data = x['augmented']
      data = self._replaceByEmbeddings(data)
      predictions = self._model(data, training=True)
      predictions = dict(**predictions['intermediate'], final=predictions['result'])
      for name, pts in predictions.items():
        loss = self._pointLoss(y, pts)
        losses['loss-%s' % name] = tf.reduce_mean(loss)
        continue
      loss = sum(losses.values())
      losses['loss'] = loss
  
    self._model.optimizer.minimize(loss, TV, tape=tape)
    ###############
    return losses

  def fit(self, data):
    t = time.time()
    losses = self._trainStep(data)
    losses = {k: v.numpy() for k, v in losses.items()}
    return {'time': int((time.time() - t) * 1000), 'losses': losses}
  
  def load(self, folder=None, postfix='', encoder=None):
    if encoder is not None:
      F2S = self._modelRaw['Face2Step']
      F2S.trainable = False
      F2S.load_weights(encoder)
    
    if folder is not None: super().load(folder, postfix)
    self._compile()
    return
  
  def _eval(self, xy):
    print('Instantiate _eval')
    x, (y,) = xy
    x = self._replaceByEmbeddings(x)
    y = y[:, :, 0]
    predictions = self._model(x, training=False)
    points = predictions['result'][:, :, :]
    tf.assert_equal(tf.shape(points), tf.shape(y))

    loss = self._pointLoss(y, points)
    tf.assert_equal(tf.shape(loss), tf.shape(y)[:1])
    _, dist = NNU.normVec(points - y)
    return loss, points, dist

  def eval(self, data):
    loss, sampled, dist = self._eval(data)
    return loss.numpy(), sampled.numpy(), dist.numpy()