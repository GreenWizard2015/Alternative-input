import tensorflow as tf
from NN.Utils import sMLP
import itertools

def reduce_mean_masked(x, mask):
  N = tf.reduce_sum(mask, axis=-1, keepdims=True)
  summed = tf.reduce_sum(x * mask, axis=-1, keepdims=True)
  tf.assert_equal(tf.shape(summed), tf.shape(N))
  return summed / N

class SeparableCritic(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._g = sMLP([128, 64, 1], 'relu', name='%s/g' % (self.name,))
    self._h = sMLP([128, 64, 1], 'relu', name='%s/h' % (self.name,))
    return

  def call(self, x, y):
    B = tf.shape(x)[0]
    scores = tf.matmul(self._h(x), self._g(y), transpose_b=True)
    tf.assert_equal(tf.shape(scores), (B, B))
    return scores
    
class ConcatCritic(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._g = sMLP([128, 128, 64, 64, 64, 1], 'relu', name='%s/g' % (self.name,))
    return

  def call(self, x, y):
    B = tf.shape(x)[0]
    # Tile all possible combinations of x and y
    x_tiled = tf.tile(x[None, :],  (B, 1, 1))
    y_tiled = tf.tile(y[:, None],  (1, B, 1))
    # xy is [B * B, x_dim + y_dim]
    xy_pairs = tf.reshape(
      tf.concat((x_tiled, y_tiled), axis=2),
      [B * B, -1]
    )
    # Compute scores for each x_i, y_j pair.
    scores = self._g(xy_pairs)
    scores = tf.reshape(scores, [B, B])
    scores = tf.transpose(scores)
    return scores
 
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
#     self._scores = SeparableCritic()
    self._scores = ConcatCritic()
    self._alpha = sMLP([168, 64, 1], 'relu')
    return
  
  def call(self, x, y, mask=None):
    B = tf.shape(x)[0]
    if tf.is_tensor(y):
      scores = self._scores(x, y)
    else:
      permutations = [
        tf.concat(perm, axis=-1)
        for perm in itertools.permutations(y)
      ]

      scores = [
        self._scores(x, tf.concat(perm, axis=-1))
        for perm in permutations
      ]
      scores = tf.concat(scores, axis=-1)
      y = permutations[0] # same as tf.concat(y, axis=-1)
      pass

    tf.assert_equal(tf.shape(scores)[:1], (B, ))
    
    log_alpha = self._alpha(y)
    scores = scores - log_alpha
    
    joint = tf.linalg.diag_part(scores)[..., None]
    tf.assert_equal(tf.shape(joint), (B, 1))
    
    if mask is None:
      mask = tf.ones((tf.shape(scores)[-1], ), dtype=x.dtype)
      mask = tf.linalg.tensor_diag(mask)[:B]
    tf.assert_equal(tf.shape(scores), tf.shape(mask))
     
    marginal = reduce_mean_masked(scores, 1.0 - mask)
    tf.assert_equal(tf.shape(joint), tf.shape(marginal))

    mi = 1.0 + joint - tf.exp(marginal)
    return mi