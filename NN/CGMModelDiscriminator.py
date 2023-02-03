import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as L
from NN.Utils import sMLP, CParallelDense
import NN.Utils as NNU

def _createModel(FACE_LATENT_SIZE, NGaussians=128):
  latentFace = L.Input((FACE_LATENT_SIZE, ))

  latent = sMLP(sizes=[256, ] * 5, activation='relu')(latentFace)
  mu = 0.5 + CParallelDense(2, N=NGaussians)(latent)
  return tf.keras.Model(
    inputs=[latentFace],
    outputs={
      'mu': mu
    }
  )

class CGMModelDiscriminator(tf.keras.Model):
  def __init__(self, deviation=(0.001, 0.01), **kwargs):
    super().__init__(**kwargs)
    self._deviation = deviation
    return

  def build(self, input_shape):
    self._model = _createModel(input_shape[-1])
    return super().build(input_shape)

  def _scaleDiag(self, B, N):
    deviation = self._deviation
    if isinstance(deviation, tuple):
      min_, max_ = deviation
      deviation = tf.linspace(min_, max_, N)[:, None] # (N, 1)
      # make it (B * N, 2) by tiling it
      scaleDiag = tf.tile(deviation, (B, 2)) # (B * N, 2)
      tf.assert_equal(tf.shape(scaleDiag), (B * N, 2))
    else:
      scaleDiag = tf.ones((1, 2), dtype=tf.float32) * deviation
    return scaleDiag

  def fromLatents(self, latents, stop_gradient=False, **kwargs):
    predictions = self._model(latents, **kwargs)
    mu = predictions['mu']
    if stop_gradient: mu = tf.stop_gradient(mu)

    B = tf.shape(latents)[0]
    N = tf.shape(mu)[1]
    tf.assert_equal(tf.shape(mu), (B, N, 2))
    # make from mu a multivariate normal distribution with scaleDiag
    distributions = tfp.distributions.MultivariateNormalDiag(
      loc=tf.reshape(mu, (B * N, 2)), 
      scale_diag=self._scaleDiag(B, N)
    )
    return distributions, mu, (B, N)

  def call(self, latents, **kwargs):
    distributions, mu, (B, N) = self.fromLatents(latents)
    if not ('realPoints' in kwargs): return mu

    # take args from kwargs
    realPoints = kwargs['realPoints']
    tf.assert_equal(tf.shape(realPoints), (B, 2))
    distributionPredicted = kwargs['distributionPredicted']
    distributionPredictedDetached = kwargs['distributionPredictedDetached']
    samplesPerDistribution = kwargs.get('samplesPerDistribution', 2)
    ################################################################
    def realPointsLoss():
      points = tf.repeat(realPoints, N, axis=0)
      tf.assert_equal(tf.shape(points), (B * N, 2))
      # minimize D.log_prob(points)
      loss = distributions.log_prob(points)
      tf.assert_equal(tf.shape(loss), (B * N, ))
      tf.debugging.assert_all_finite(loss, 'realPointsLoss')

      loss = tf.reshape(loss, (B, N))
      loss = tf.nn.sigmoid(loss)
      tf.assert_equal(tf.shape(loss), (B, N))
      return loss
    realPointsLoss = realPointsLoss()
    ################################################################
    # sample from the distribution
    sampled = distributions.sample(samplesPerDistribution)
    tf.assert_equal(tf.shape(sampled), (samplesPerDistribution, B * N, 2))
    tf.debugging.assert_all_finite(sampled, 'sampled')
    
    sampled = tf.reshape(sampled, (samplesPerDistribution, B, N, 2))
    # first, transpose to (samplesPerDistribution, N, B, 2)
    sampled = tf.transpose(sampled, (0, 2, 1, 3))
    # then, reshape to (samplesPerDistribution * N, B, 2)
    sampled = tf.reshape(sampled, (samplesPerDistribution * N, B, 2))
    tf.assert_equal(tf.shape(sampled), (samplesPerDistribution * N, B, 2))
    
    def discriminatorLoss():
      # maximize log_prob(sampled) from detached distribution
      loss = distributionPredictedDetached.log_prob(sampled)
      tf.debugging.assert_all_finite(loss, 'discriminatorLoss')
      tf.assert_equal(tf.shape(loss), (samplesPerDistribution * N, B))
      
      loss = tf.reshape(loss, (samplesPerDistribution, N, B))
      loss = tf.transpose(loss, (2, 1, 0))
      tf.assert_equal(tf.shape(loss), (B, N, samplesPerDistribution))

      lossA = tf.nn.sigmoid(-loss)
      # MSLE penalizes overestimation more than underestimation
      # so, if 0 < log_prob then it is penalized more than if log_prob < 0
      lossA = tf.losses.msle(tf.nn.sigmoid(-0.0), lossA)
      ################################################################
      # minimize intersections of distributions
      # find log_prob(sampled) across all distributions
      tf.assert_equal(tf.shape(sampled), (samplesPerDistribution * N, B, 2))
      pts = tf.repeat(sampled, N, axis=-2)
      tf.assert_equal(tf.shape(pts), (samplesPerDistribution * N, B * N, 2))

      lossB = distributions.log_prob(pts)
      tf.debugging.assert_all_finite(lossB, 'discriminatorLoss B')
      tf.assert_equal(tf.shape(lossB), (samplesPerDistribution * N, B * N))
      lossB = tf.reshape(lossB, (samplesPerDistribution, N, B, N))
      # transpose to (B, N, samplesPerDistribution, N)
      lossB = tf.transpose(lossB, (2, 1, 0, 3))
      tf.assert_equal(tf.shape(lossB), (B, N, samplesPerDistribution, N))
      lossB = tf.nn.softmax(lossB, axis=-1)
      
      # take lossB[:, x, :, x] by using mask multiplication (more explicit than tf.gather)
      mask = tf.eye(N, dtype=tf.float32)[None, :, None, :]
      tf.assert_equal(tf.shape(mask), (1, N, 1, N))
      mask = tf.tile(mask, (B, 1, samplesPerDistribution, 1))
      tf.assert_equal(tf.shape(mask), (B, N, samplesPerDistribution, N))
      tf.assert_equal(tf.shape(mask), tf.shape(lossB))
      lossB = tf.reduce_sum(lossB * mask, axis=-1)
      tf.assert_equal(tf.shape(lossB), (B, N, samplesPerDistribution))
      # take lossB[:, x, :, x] by using tf.gather
      # indx = tf.reshape(tf.range(N), (1, N, 1, 1))
      # indx = tf.tile(indx, (B, 1, samplesPerDistribution, 1))
      # tf.assert_equal(tf.shape(indx), (B, N, samplesPerDistribution, 1))
      # lossB = tf.gather(lossB, indx, batch_dims=3, axis=-1)
      # tf.assert_equal(tf.shape(lossB), (B, N, samplesPerDistribution, 1))
      # smooth the lossB
      lossB = tf.square(1.0 - lossB)
      ################################################################
      return {
        'prob': lossA,
        'intersection': lossB,
      }

    def generatorLoss():
      # sampled: (samplesPerDistribution * N, B, 2)
      # minimize log_prob(sampled) from predicted distribution
      loss = distributionPredicted.log_prob(
        tf.stop_gradient(sampled)
      )
      tf.debugging.assert_all_finite(loss, 'generatorLoss')
      tf.assert_equal(tf.shape(loss), (samplesPerDistribution * N, B))
      loss = tf.nn.sigmoid(loss)
      ################################
      # split samplesPerDistribution * N into samplesPerDistribution and N
      loss = tf.reshape(loss, (samplesPerDistribution, N, B))
      # transpose to (B, N, samplesPerDistribution)
      loss = tf.transpose(loss, (2, 1, 0))
      # calculate the mean over samplesPerDistribution
      loss = tf.reduce_mean(loss, axis=-1)
      tf.assert_equal(tf.shape(loss), (B, N))
      tf.assert_equal(tf.shape(loss), tf.shape(realPointsLoss))
      # scale the loss by the inverse of the realPointsLoss
      scale = tf.stop_gradient(1.0 - realPointsLoss)
      return loss * scale

    discriminatorLoss = discriminatorLoss()
    return {
      'sampled': sampled,
      'generator loss': generatorLoss(),
      'losses': {
        'real': realPointsLoss,
        **discriminatorLoss
      }
    }

  def scorePoints(self, latents, points):
    distribution, _, (B, N) = self.fromLatents(latents, stop_gradient=True, training=False)
    tf.assert_equal(tf.shape(points), (B, 2))
    points = tf.repeat(points, N, axis=0)
    tf.assert_equal(tf.shape(points), (B * N, 2))
    probs = distribution.log_prob(points)
    tf.debugging.assert_all_finite(probs, 'scorePoints')
    tf.assert_equal(tf.shape(probs), (B * N,))
    probs = tf.reshape(probs, (B, N))
    # apply softmax and calc entropy
    probs = tf.nn.softmax(probs, axis=-1)
    probs = tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
    tf.assert_equal(tf.shape(probs), (B,))
    return probs