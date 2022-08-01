import tensorflow as tf
from NN.Utils import sMLP

def reduce_mean_masked(x, mask):
  N = tf.reduce_sum(mask, axis=-1, keepdims=True)
  summed = tf.reduce_sum(x * mask, axis=-1, keepdims=True)
  tf.assert_equal(tf.shape(summed), tf.shape(N))
  return summed / N

class SeparableCritic(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._g = sMLP([128, 64, 1], name='%s/g' % (self.name,))
    self._h = sMLP([128, 64, 1], name='%s/h' % (self.name,))
    return

  def call(self, x, y):
    B = tf.shape(x)[0]
    scores = tf.matmul(self._h(x), self._g(y), transpose_b=True)
    tf.assert_equal(tf.shape(scores), (B, B))
    return scores
    
class ConcatCritic(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._g = sMLP([128, 64, 1], 'linear', name='%s/g' % (self.name,))
    return

  def call(self, x, y):
    batch_size = tf.shape(x)[0]
    # Tile all possible combinations of x and y
    x_tiled = tf.tile(x[None, :],  (batch_size, 1, 1))
    y_tiled = tf.tile(y[:, None],  (1, batch_size, 1))
    # xy is [batch_size * batch_size, x_dim + y_dim]
    xy_pairs = tf.reshape(tf.concat((x_tiled, y_tiled), axis=2), [batch_size * batch_size, -1])
    # Compute scores for each x_i, y_j pair.
    scores = self._g(xy_pairs) 
    return tf.transpose(tf.reshape(scores, [batch_size, batch_size]))
 
class EnsembledCritic(tf.keras.Model):
  def __init__(self, NCritics, critic=SeparableCritic, **kwargs):
    super().__init__(**kwargs)
    self._critics = [
      critic(name='%s/critic-%d' % (self.name, i)) 
      for i in range(NCritics)
    ]
    self._combine = tf.reduce_mean # tf.reduce_logsumexp
    return
 
  def call(self, x, y):
    B = tf.shape(x)[0]
    scores = tf.stack([critic(x, y) for critic in self._critics], axis=-1)
    tf.assert_equal(tf.shape(scores), (B, B, len(self._critics)))
    scores = self._combine(scores, axis=-1)
    tf.assert_equal(tf.shape(scores), (B, B))
    return scores

class MIModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._scores = SeparableCritic()
    self._alpha = sMLP([168, 64, 1], 'relu')
    self.optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    return
  
  def call(self, x, y, mask=None):
    B = tf.shape(x)[0]
    scores = self._scores(x, y)
    alpha = self._alpha(y)
    
    joint = tf.linalg.diag_part(scores)[..., None]
    tf.assert_equal(tf.shape(joint), (B, 1))
    
    if mask is None:
      mask = tf.linalg.tensor_diag(tf.ones((B, ), dtype=x.dtype))
    tf.assert_equal(tf.shape(scores), tf.shape(mask))
    
    marginal = reduce_mean_masked(scores, 1.0 - mask)
    tf.assert_equal(tf.shape(joint), tf.shape(marginal))
    tf.assert_equal(tf.shape(alpha), tf.shape(marginal))
    
    mi = 1.0 + joint - (tf.exp(marginal - alpha) + alpha)
    return mi