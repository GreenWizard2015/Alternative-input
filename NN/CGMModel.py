import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as L
from NN.Utils import sMLP

def _GMModel(FACE_LATENT_SIZE, NGaussians=8):
  latentFace = L.Input((FACE_LATENT_SIZE, ))

  latent = sMLP(sizes=[256, ] * 3, activation='relu')(latentFace)
  scale_tril = lambda x: tfp.math.fill_triangular(
    tf.exp(x) + 1e-8 # 1e-8 <= x
  )
  return tf.keras.Model(
    inputs=[latentFace],
    outputs={
      'mu': 0.5 + L.Reshape((-1, 2))( L.Dense(2 * NGaussians)(latent) ),
      'alpha': L.Dense(NGaussians, 'softmax')(latent),
      'scale_tril': L.Lambda(scale_tril)(
        L.Reshape((NGaussians, 3))(
          L.Dense(3 * NGaussians)(latent)
        )
      ),
    }
  )

class CGMModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
#     self.optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    return
  
  def build(self, input_shape):
    self._latent2gmm = _GMModel(FACE_LATENT_SIZE=input_shape[-1])
    return super().build(input_shape)
  
  def call(self, latent):
    predictions = self._latent2gmm(latent)
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
  _GMModel(8, 8).summary()
  ####################
  tfd = tfp.distributions
  probs = tf.constant([ [1., 1., 1.], ])
  probs = tf.nn.softmax(probs)
  print('probs: ', probs.shape)
  loc = tf.constant([[
    [-.3, .8],
    [-.3, .1],
    [.7, .1],
  ]])
  print('loc: ', loc.shape)
  scale_tril = tfp.bijectors.fill_scale_tril.FillScaleTriL(
    diag_shift=1e-5,
  )
  scale_tril=scale_tril(tf.random.normal((1, 3, 3)) *1)
  print(scale_tril)
  distribution = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=probs),
    components_distribution=tfd.MultivariateNormalTriL(
      loc=loc, 
      scale_tril=scale_tril
    )
  )
  
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

