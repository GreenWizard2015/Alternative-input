import networks
import tensorflow as tf
import time

class CFakeModel:
  def __init__(self):
    self.useAR = False
    if self.useAR:
      self._model = model = networks.ARModel()
      self._ARDepth = 5
      self._trainStepF = self._trainARStep(steps=self._ARDepth)
    else:
      self._model = model = networks.simpleModel()
      self._trainStepF = self._trainSimpleStep

    model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-4),
      loss=None
    )
    self._epoch = 0
    return

  @tf.function
  def _inferAR(self, data, training, steps):
    # positions = tf.random.uniform(shape=(tf.shape(data[0])[0], 2))
    positions = tf.ones(shape=(tf.shape(data[0])[0], 2), dtype=tf.float32) * 0.5
    history = []
    for _ in range(steps + 1):
      positions = self._model([*data, positions], training=training)
      history.append(positions)
      continue
    return history

  def _trainARStep(self, steps):
    def f(x, y):
      pred = self._inferAR(x, training=True, steps=steps)
      loss = 0.0
      for coords in pred:
        loss = loss + tf.losses.mse(y, coords)
      return loss
    return f

  def _trainSimpleStep(self, x, y):
    pred = self._model(x, training=True)
    # lossA = tf.losses.mse(y[:, None], pred['raw coords'])
    # lossB = tf.losses.mse(y, pred['coords'])
    return tf.losses.mse(y, pred['coords'])

  @tf.function
  def _trainStep(self, data):
    x, (y, ) = data
    with tf.GradientTape() as tape:
      loss = tf.reduce_mean(self._trainStepF(x, y))

    model = self._model
    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

  def fit(self, data):
    t = time.time()
    loss = self._trainStep(data).numpy()

    self._epoch += 1
    t = time.time() - t
    return {'loss': loss, 'epoch': self._epoch, 'time': int(t * 1000)}
  
  def __call__(self, data):
    if self.useAR:
      res = self._inferAR(data, training=False, steps=self._ARDepth)
      return [x.numpy()[0] for x in res]
    else:
      return self._model(data, training=False)['coords'].numpy()
    
  def debug(self, data):
    res = self._model(data, training=False)
    print(res['raw coords'].numpy())
    print(res['coords'].numpy())