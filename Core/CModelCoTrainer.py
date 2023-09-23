import tensorflow as tf
import time
import NN.Utils as NNU
from Core.CModelWrapper import CModelWrapper

'''
  Based on ideas from https://arxiv.org/abs/1804.06872
  But highly modified to fit the given task
'''
class CModelCoTrainer(CModelWrapper):
  def __init__(self, timesteps, model='simple', useEMA=False, **kwargs):
    super().__init__(timesteps, model, **kwargs)
    self._useEMA = useEMA
    if self._useEMA: self._eta = kwargs.get('eta', 1e-3)

    self._modelBW = CModelWrapper(timesteps, model, **kwargs)
    self._modelB = self._modelBW._model
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
    print('Compile model')
    self._model.compile(optimizer=NNU.createOptimizer())
    if not self._useEMA:
      self._modelB.compile(optimizer=NNU.createOptimizer())
    else:
      # hard copy the weights of the model to the modelB
      for v, vB in zip(self._model.variables, self._modelB.variables): vB.assign(v)
    return
  
  def _lossWeights(self, 
    model, x, y,
    minSigma=2.0, maxSigma=6.0, 
    minWeight=0.01, maxWeight=1.0,
    fixTarget=True
  ):
    B, N = tf.shape(y)[0], tf.shape(y)[1]
    predictions = model(x, training=False)['result']
    tf.assert_equal(tf.shape(predictions), tf.shape(y))
    _, dist = NNU.normVec(predictions - y)
    tf.assert_equal(tf.shape(dist), (B, N, 1))
    # normalize the distances
    w = tf.math.divide_no_nan(dist - tf.reduce_mean(dist), tf.math.reduce_std(dist))
    normedW = (w - minSigma) / (maxSigma - minSigma) # [0, 1]
    w = 1 - normedW # [1, 0]
    w = tf.clip_by_value(w, clip_value_min=minWeight, clip_value_max=maxWeight)
    w = tf.stop_gradient( tf.square(w) )
    tf.assert_equal(tf.shape(w), (B, N, 1))
    if not fixTarget: return w, y

    targets = tf.where(0.0 < normedW, tf.stop_gradient(predictions), y)
    return w, targets
  
  def _trainStep(self, Data):
    print('Instantiate _trainStep')
    ###############
    x, (y, ) = Data
    y = y[..., 0, :]
    ###############
    # update weights of the modelB using EMA
    if self._useEMA:
      eta = self._eta
      for v, emaV in zip(self._model.variables, self._modelB.variables):
        emaV.assign((emaV * (1.0 - eta)) + (v * eta))
        continue
      pass

    targetForA = y
    lossesB = {}
    weightsA, targetForB = self._lossWeights(self._modelB, x['clean'], y)
    if not self._useEMA:
      weightsB, targetForA = self._lossWeights(self._model, x['clean'], y)
      lossesB = self._optimize(self._modelB, x, targetForB, weightsB)
    ###############
    lossesA = self._optimize(self._model, x, targetForA, weightsA)
    # combine losses, but prefix them with A/B
    losses = {}
    for k, v in lossesA.items(): losses['A-%s' % k] = v
    for k, v in lossesB.items(): losses['B-%s' % k] = v
    return losses

  def _optimize(self, model, x, y, w):
    losses = {}
    TV = model.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(TV)
      data = x['augmented']
      predictions = model(data, training=True)
      predictions = dict(**predictions['intermediate'], final=predictions['result'])
      for name, pts in predictions.items():
        tf.assert_equal(tf.shape(pts), tf.shape(y))
        loss = tf.losses.mse(y * w, pts * w)
        losses['loss-%s' % name] = tf.reduce_mean(loss)
        continue
      losses['loss'] = loss = sum(losses.values())
      if len(losses) <= 2:
        losses = dict(loss=loss)
  
    model.optimizer.minimize(loss, TV, tape=tape)
    ###############
    return losses

  def fit(self, data):
    t = time.time()
    losses = self._trainStep(data)
    losses = {k: v.numpy() for k, v in losses.items()}
    return {'time': int((time.time() - t) * 1000), 'losses': losses}
  
  def load(self, folder, postfix=''):
    super().load(folder, postfix)
    self._compile()
    return
  
  def _eval(self, xy):
    print('Instantiate _eval')
    x, (y,) = xy
    w, y = self._lossWeights(self._modelB, x, y[..., 0, :])
    y = y[:, -1]
    
    predictions = self._model(x, training=False)
    result = predictions['result']
    points = result[:, -1, :]
    tf.assert_equal(tf.shape(points), tf.shape(y))

    loss = tf.losses.mse(y, points)
    tf.assert_equal(tf.shape(loss), tf.shape(y)[:1])
    _, dist = NNU.normVec(points - y)
    return loss, points, dist

  def eval(self, data):
    loss, sampled, dist = self._eval(data)
    return loss.numpy(), sampled.numpy(), dist.numpy()