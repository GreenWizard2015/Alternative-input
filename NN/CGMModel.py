import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as L
from NN.Utils import sMLP, CParallelDense
import numpy as np

def gumbel_softmax(logits, temperature=1.0):
  '''
    Sample from the Gumbel-Softmax distribution.
    Args:
      logits: [..., n_class] unnormalized log-probs
      temperature: non-negative scalar
    Returns:
      tuple of (sample, hard sample)
      sample: [..., n_class] sample from the Gumbel-Softmax distribution.
      hard sample: [..., n_class] one-hot sample from the Gumbel-Softmax distribution.
  '''
  eps = 1e-20
  U = tf.random.uniform(tf.shape(logits), minval=0.0, maxval=1.0)
  U = -tf.math.log(-tf.math.log(U + eps) + eps)

  y = tf.nn.softmax((logits + U) / temperature, axis=-1)

  k = tf.shape(logits)[-1]
  y_hard = tf.cast(tf.one_hot(tf.argmax(y, axis=-1), k), y.dtype)
  return y, tf.stop_gradient(y_hard - y) + y

# define layer CTreeDecoder that takes latent and outputs mu, scale_tril, weights
class CTreeDecoder(L.Layer):
  def __init__(self, NodesPerLayer, TreeLevels, temperature=1.0, hard=False, reg=1e-2, NGaussians=None, **kwargs):
    super().__init__(**kwargs)
    TreeLevels += 2
    self._temperature = temperature
    self._hard = hard
    self._reg = reg
    self._NGaussians = NGaussians if NGaussians is not None else NodesPerLayer

    self._NodesPerLayer = NodesPerLayer
    self._levelSizes = [NodesPerLayer ** i for i in range(1, TreeLevels - 1)]
    
    self._rawG = CParallelDense(2 + 3 + 1, self._levelSizes[-1] * self._NGaussians)
    totalWeights = sum(self._levelSizes)
    self._w = L.Dense(totalWeights, activation='relu')
    print('total weights:', totalWeights)
    print('Level sizes:', ', '.join([str(s) for s in self._levelSizes]))
    return

  def _regularize(self, w):
    entropy = -tf.reduce_sum(w * tf.math.log(w + 1e-8), axis=-1)
    # penalize only if entropy is low and push it to 1
    entropy = tf.clip_by_value(entropy, 0.0, 1.0)
    self.add_loss(self._reg * (1.0 - tf.reduce_mean(entropy)))
    return

  def _softmax(self, x, training=None):
    relaxed, hard = gumbel_softmax(x, temperature=self._temperature)
    if training:
      self._regularize(relaxed) # add regularization loss NOT to the hard/OHE version
      return hard if self._hard else relaxed

    return hard

  def _weights(self, latent, training=None):
    B = tf.shape(latent)[0]
    # predict and split weights
    chunkedWeights = tf.split(self._w(latent, training=training), self._levelSizes, axis=-1)
    # prepare weights for multiplication
    chunkedWeights = [tf.reshape(w, (B, -1, self._NodesPerLayer)) for w in chunkedWeights]
    chunkedWeights = [
      tf.reshape(
        self._softmax(w, training=training),
        (B, self._NodesPerLayer ** i, self._NodesPerLayer)
      )
      for i, w in enumerate(chunkedWeights)
    ]
    
    # multiply weights
    weights = chunkedWeights[0]
    weights = tf.reshape(weights, (B, self._NodesPerLayer, 1))
    for i, wPart in enumerate(chunkedWeights[1:]):
      tf.assert_equal(tf.shape(wPart), (B, self._NodesPerLayer ** (i + 1), self._NodesPerLayer))
      tf.assert_equal(tf.shape(weights), (B, self._NodesPerLayer ** (i + 1), 1))
      weights = tf.reshape(weights * wPart, (B, -1, 1))
      tf.assert_equal(tf.shape(weights), (B, self._NodesPerLayer ** (i + 2), 1))
      continue
    weights = weights[..., 0]

    # for debugging, find the max weight and print unique values
    if True:
      maxWeight = tf.reduce_max(weights, axis=-1)
      maxInd = tf.argmax(weights, axis=-1)
      uniqueInd, _ = tf.unique(maxInd)
      tf.print(tf.reduce_mean(maxWeight), uniqueInd)

    return weights

  def call(self, latent, training=None):
    B = tf.shape(latent)[0]
    raw = self._rawG(latent, training=training)
    raw = tf.reshape(raw, (B, -1, self._NGaussians, tf.shape(raw)[-1]))
    
    weights = self._weights(latent, training=training)
    tf.assert_equal(tf.shape(weights), tf.shape(raw)[:2])
    weights = tf.reshape(weights, tf.shape(raw[..., :1, :1]))
    raw = raw * weights

    raw = tf.reduce_sum(raw, axis=1)
    tf.assert_equal(tf.shape(raw), (B, self._NGaussians, 2 + 3 + 1))
    ###################################
    mu, scaleRaw, weightsG = tf.split(raw, [2, 3, 1], axis=-1)
    weightsG = tf.nn.leaky_relu(weightsG[..., 0], alpha=0.2)
    return(
      0.5 + mu,
      tfp.math.fill_triangular(tf.exp(scaleRaw) + 1e-8),
      tf.nn.softmax(weightsG, axis=-1)
    )

def _GMModel(FACE_LATENT_SIZE):
  latentFace = L.Input((FACE_LATENT_SIZE, ))

  latent = sMLP(sizes=[256, ] * 5, activation='relu')(latentFace)

  NodesPerLayer = 3
  TreeLevels = 4
  mu, scale_tril, weights = CTreeDecoder(NodesPerLayer, TreeLevels, hard=False)(latent)
  
  return tf.keras.Model(
    inputs=[latentFace],
    outputs={
      'mu': mu,
      'alpha': weights,
      'scale_tril': scale_tril
    }
  )

class CGMModel(tf.keras.Model):
  def build(self, input_shape):
    self._latent2gmm = _GMModel(FACE_LATENT_SIZE=input_shape[-1])
    return super().build(input_shape)
  
  def call(self, latent, shift=None):
    predictions = self._latent2gmm(latent)
    if not(shift is None):
      predictions['mu'] = predictions['mu'] + shift[:, None, :]
    return(predictions, self.distribution(predictions))
    
  def distribution(self, predictions, nograds=False):
    F = tf.stop_gradient if nograds else lambda x: x 
    tfd = tfp.distributions
    mixture_distribution = tfd.Categorical(probs=F(predictions['alpha']))
    components_distribution = tfd.MultivariateNormalTriL(
      loc=F(predictions['mu']),
      scale_tril=F(predictions['scale_tril'])
    )
    return tfd.MixtureSameFamily(
      mixture_distribution=mixture_distribution,
      components_distribution=components_distribution
    )

##############
if __name__ == '__main__':
  import numpy as np
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)]
  )
  model = CGMModel()
  model.build((None, 64))
  model.summary()
  # call model with random input and print output shapes
  latent = tf.random.normal((7, 64))
  predictions, dist = model(latent)
  for k, v in predictions.items():
    print(k, v.shape)
  exit(0)
  ####################
  tfd = tfp.distributions
  scale_tril = tf.constant([
    [0.1, 0.0], 
    [0.0, 0.1],
  ])
  locA = tf.constant([
    [0.0, 0.0],
    [0.3, 0.3],
  ])
  locB = tf.constant([
    [-0.5, -0.5],
    [-0.03, -0.],
  ])

  combined = tf.concat([locA, locB], axis=0)
  tf.assert_equal(tf.shape(combined), [4, 2])
  tril = tf.repeat(scale_tril[tf.newaxis, ...], 4, axis=0)
  tf.assert_equal(tf.shape(tril), [4, 2, 2])
  C = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[1./4.] * 4),
    components_distribution=tfp.distributions.MultivariateNormalTriL(
      loc=combined,
      scale_tril=tril
    )
  )
  distribution = C
  print(distribution.sample())
  def makeGrid():
    xy = np.linspace(-1., 1., 100)
    XY = np.meshgrid(xy, xy)
    res = np.concatenate([x.reshape(-1, 1) for x in XY], axis=-1)
    return XY, tf.constant(res, tf.float32)

  grid, gridTF = makeGrid()
  import matplotlib.pyplot as plt
  plt.figure(figsize=(8, 8))
  cmap = 'jet'
  ###############
  lp = distribution.log_prob(gridTF)
  lp = tf.nn.sigmoid(lp)
  lp = tf.reshape(lp, (100, 100, )).numpy()
  msh = plt.pcolormesh(
      grid[0], grid[1], lp, 
      cmap=cmap, vmin=0, vmax=1
  )
  plt.show()
  pass
