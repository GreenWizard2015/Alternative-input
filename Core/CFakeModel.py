import networks
import tensorflow as tf
import time

class CFakeModel:
  def __init__(self):
    self.useAR = not False
    if self.useAR:
      self._model = model = networks.ARModel()
    else:
      self._model = model = networks.simpleModel()
      
    model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-4),
      loss='mse'
    )
    self._epoch = 0
    return

  @tf.function
  def _inferAR(self, data, training, steps):
    positions = tf.random.uniform(shape=(tf.shape(data[0])[0], 2))
    history = []
    for _ in range(steps + 1):
      positions = self._model([*data, positions], training=training)
      history.append(positions)
      continue
    return history

  @tf.function
  def _trainAR(self, data, steps):
    x, (y, ) = data
    with tf.GradientTape() as tape:
      pred = self._inferAR(x, training=True, steps=steps)
      loss = 0.0
      for coords in pred:
        loss = loss + tf.losses.mse(y, coords)

    model = self._model
    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return tf.reduce_mean(loss)

  def fit(self, data):
    t = time.time()
    if self.useAR:
      loss = self._trainAR(data, steps=5).numpy()
    else:
      x, y = data
      # TODO: Write custom training step (it's faster)
      loss = self._model.fit(x, y, batch_size=len(y), verbose=2).history['loss'][0]

    self._epoch += 1
    t = time.time() - t
    return {'loss': loss, 'epoch': self._epoch, 'time': int(t * 1000)}
  
  def __call__(self, data):
    if self.useAR:
      res = self._inferAR(data, training=False, steps=5)
      return [x.numpy()[0] for x in res]
    else:
      return self._model(data, training=False).numpy()