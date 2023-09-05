#!/usr/bin/env python
# -*- coding: utf-8 -*-.
'''
  This script takes a dataset along with a model and visualizes the results
'''
import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

import Core.Utils as Utils
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
from Core.CModelWrapper import CModelWrapper
import tqdm
import imageio
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def samplesStream(ds, take, batchSize, models, indices, augmentedN):
  augmentations = dict(
    pointsDropout=0.0, pointsNoise=0.01,
    eyesDropout=0.0, eyesAdditiveNoise=0.01, brightnessFactor=2.0, lightBlobFactor=2.0,
  )
  for i in range(0, len(indices), batchSize):
    batch, rejected, accepted = ds.sampleByIds(indices[i:i+batchSize])
    if batch is None: continue

    # main batch
    x, (y, ) = batch
    x = x[take]
    predictions = [model.predict(x) for model in models]
    for _ in range(augmentedN):
      augmBatch, _, augmAccepted = ds.sampleByIds(accepted, **augmentations)
      # augmAccepted should be the same as accepted
      if np.all(np.equal(accepted, augmAccepted)):
        augmX, _ = augmBatch
        augmX = augmX['augmented']
        augmPredictions = [model.predict(augmX) for model in models]
        predictions.extend(augmPredictions)
      continue
    # assert len(predictions) == len(y), 'The number of predictions must be equal to the number of samples'
    for idx in range(len(y)):
      res = {k: v[idx] for k, v in x.items()}
      yield res, y[idx], [pred[idx] for pred in predictions]
    continue
  return

def visualizer():
  plt.figure(figsize=(10, 10))
  def visualize(x, y, predictions):
    y = y[-1, 0] # take the last timestep
    predictions = np.array(predictions, dtype=np.float32)[:, -1]

    plt.clf()
    # plot the y as a green dot
    plt.plot(y[0], y[1], 'go', markersize=2)
    # plot the predicted as a red dot
    for predicted in predictions:
      plt.plot(predicted[0], predicted[1], 'ro', markersize=2)
      continue
    if 1 < len(predictions):
      # plot the mean of the predictions as a blue dot
      mean = np.mean(predictions, axis=0)
      plt.plot(mean[0], mean[1], 'bo', markersize=2)
      pass
    # plot the boundaries
    for i in range(5):
        d = i * 0.1
        plt.gca().add_patch(
          patches.Rectangle(
            (d,d), 1-2*d, 1-2*d,
            linewidth=1,edgecolor='r',facecolor='none'
          )
        )
    # plt.axis('equal')
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    canvas = plt.gcf().canvas
    canvas.draw()
    ncols, nrows = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(nrows, ncols, 3)
    # imageio suggests to pad the image such that the width and height are divisible by 16
    M = 16
    pad1 = M - (image.shape[0] % M)
    pad2 = M - (image.shape[1] % M)
    if (pad1 != M) or (pad2 != M):
      padA = padB = (0, 0)
      if pad1 != 0: padA = (math.floor(pad1 / 2), math.ceil(pad1 / 2))
      if pad2 != 0: padB = (math.floor(pad2 / 2), math.ceil(pad2 / 2))
      image = np.pad(image, (padA, padB, (0, 0)), 'constant', constant_values=255)
    return image
  return visualize

def main(args):
  # load the dataset
  ds = CDataSampler(
    CSamplesStorage(),
    defaults=dict(timesteps=args.steps, stepsSampling='last'),
    batch_size=args.batch_size, minFrames=args.steps
  )
  ds.addBlock( Utils.datasetFrom(args.dataset) )
  models = [
    CModelWrapper(timesteps=args.steps, weights=dict(path=model))
    for model in args.model
  ]
  
  validSamples = ds.validSamples()
  visualize = visualizer()
  with imageio.get_writer(args.output, mode='I', fps=args.fps) as writer:
    with tqdm.tqdm(total=len(validSamples)) as pbar:
      for x, y, predicted in samplesStream(ds, take='clean', batchSize=args.batch_size, models=models, indices=validSamples):
        try:
          frame = visualize(x, y, predicted)
          writer.append_data(frame)
        except Exception as e:
          print(e)
          pass
        pbar.update(1)
        continue
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Visualize the results of a model on a dataset')
  parser.add_argument(
    '--dataset', type=str, help='Path to the dataset (folder or file)',
    default=os.path.join(ROOT_FOLDER, 'Data', 'test.npz')
  )
  # model is a list of models
  parser.add_argument('--model', type=str, help='Path to the model (folder or file)', action='append', default=[])
  parser.add_argument('--steps', type=int, help='Number of timesteps', default=5)
  parser.add_argument('--batch-size', type=int, help='Batch size', default=128)
  parser.add_argument('--output', type=str, help='Output file name', default='visualize.avi')
  parser.add_argument('--fps', type=int, help='Frames per second', default=3)
  parser.add_argument('--augmented', type=int, help='Number of augmented batches', default=0)

  args = parser.parse_args()
  if not args.model:
    args.model = [os.path.join(ROOT_FOLDER, 'Data', 'simple-model-best.h5')]
  main(args)
  pass
