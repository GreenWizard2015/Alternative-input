import networks
import tensorflow

class CFakeModel:
  def __init__(self):
    self._model = model = networks.simpleModel()
    model.compile(
      optimizer=tensorflow.keras.optimizers.Adam(1e-4),
      loss='mse'
    )
    return
  
  def fit(self, data):
    x, y = data
    # augmentations?
    return self._model.fit(x, y, batch_size=len(y), verbose=2)
  
  def __call__(self, data):
    res = self._model(data, training=False).numpy()
    return res