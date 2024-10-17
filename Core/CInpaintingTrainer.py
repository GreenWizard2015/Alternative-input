import tensorflow as tf
import time
import NN.Utils as NNU
import NN.networks as networks
from Core.CBaseModel import CBaseModel

class CInpaintingTrainer:
  def __init__(self, timesteps, model='simple', KP=5, **kwargs):
    stats = kwargs.get('stats', None)
    embeddingsSize = kwargs.get('embeddingsSize', 64)
    latentSize = kwargs.get('latentSize', 64)
    embeddings = {
      'userId': len(stats['userId']),
      'placeId': len(stats['placeId']),
      'screenId': len(stats['screenId']),
      'size': embeddingsSize,
    }

    self._encoder = networks.InpaintingEncoderModel(
      steps=timesteps, latentSize=latentSize,
      embeddingsSize=embeddingsSize,
      KP=KP,
    )
    self._decoder = networks.InpaintingDecoderModel(
      latentSize=latentSize,
      embeddingsSize=embeddingsSize,
      KP=KP,
    )
    self._model = CBaseModel(
       model=model, embeddings=embeddings, submodels=[self._encoder, self._decoder]
    )
    self.compile()
    # add signatures to help tensorflow optimize the graph
    specification = networks.InpaintingInputSpec()
    self._trainStep = tf.function(
      self._trainStep,
      input_signature=[specification]
    )
    self._eval = tf.function(
      self._eval,
      input_signature=[specification]
    )
    return
  
  def compile(self):
    self._optimizer = NNU.createOptimizer()

  def _trainStep(self, Data):
    print('Instantiate _trainStep')
    ###############
    x, y = Data
    losses = {}
    with tf.GradientTape() as tape:
      x = self._model.replaceByEmbeddings(x)
      latents = self._encoder(x, training=True)['latent']
      decoderArgs = {
        'keyPoints': latents,
        'time': y['time'],
        'userId': x['userId'],
        'placeId': x['placeId'],
        'screenId': x['screenId'],
      }
      predictions = self._decoder(decoderArgs, training=True)
      losses = {}
      for k in predictions.keys():
        pred = predictions[k]
        gt = y[k]
        tf.assert_equal(tf.shape(pred), tf.shape(gt))
        loss = tf.losses.mse(gt, pred)
        losses[f"loss-{k}"] = tf.reduce_mean(loss)
        
      # calculate total loss and final loss
      losses['loss'] = loss = sum(losses.values())
  
    self._optimizer.minimize(loss, tape.watched_variables(), tape=tape)
    ###############
    return losses

  def fit(self, data):
    t = time.time()
    losses = self._trainStep(data)
    losses = {k: v.numpy() for k, v in losses.items()}
    return {'time': int((time.time() - t) * 1000), 'losses': losses}
  
  def _eval(self, xy):
    print('Instantiate _eval')
    x, (y,) = xy
    x = self._replaceByEmbeddings(x)
    y = y[:, :, 0]
    predictions = self._model(x, training=False)
    points = predictions['result'][:, :, :]
    tf.assert_equal(tf.shape(points), tf.shape(y))

    loss = self._pointLoss(y, points)
    tf.assert_equal(tf.shape(loss), tf.shape(y)[:2])
    _, dist = NNU.normVec(points - y)
    return loss, points, dist

  def eval(self, data):
    loss, sampled, dist = self._eval(data)
    return loss.numpy(), sampled.numpy(), dist.numpy()