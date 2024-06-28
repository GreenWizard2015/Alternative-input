#!/usr/bin/env python
# -*- coding: utf-8 -*-.
'''
  This script takes a dataset along with a model and visualizes the results
'''
import argparse, os, sys
# add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)
import numpy as np

import Core.Utils as Utils
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
import tqdm
import imageio
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def samplesStream(ds, batchSize, take='clean', indices=None):
  for i in range(0, len(indices), batchSize):
    batch, rejected, accepted = ds.sampleByIds(indices[i:i + batchSize])
    if batch is None: continue

    # main batch
    x, (y, ) = batch
    x = x[take]
    # assert len(predictions) == len(y), 'The number of predictions must be equal to the number of samples'
    for idx in range(len(y)):
      res = {k: v[idx] for k, v in x.items()}
      yield (res, y[idx])
    continue
  return

def visualizer():
  from matplotlib.gridspec import GridSpec
  fig = plt.figure(figsize=(10, 10))
  def visualize(x, y):
    y = y[-1, 0] # take the last timestep

    plt.clf()
    gs = GridSpec(3, 2, figure=fig)
    
    # Main plot
    ax_main = fig.add_subplot(gs[0:2, :])
    ax_main.plot(y[0], y[1], 'go', markersize=2)
    for i in range(5):
        d = i * 0.1
        ax_main.add_patch(
            patches.Rectangle(
                (d,d), 1-2*d, 1-2*d,
                linewidth=1, edgecolor='r', facecolor='none'
            )
        )
    ax_main.set_xlim([-0.1, 1.1])
    ax_main.set_ylim([-0.1, 1.1])
    ax_main.set_title('Main Plot')

    # Left eye plot
    ax_left_eye = fig.add_subplot(gs[2, 0])
    left_eye = x['left eye'].numpy()[0]
    ax_left_eye.imshow(left_eye, cmap='gray')
    ax_left_eye.set_title('Left Eye')

    # Right eye plot
    ax_right_eye = fig.add_subplot(gs[2, 1])
    right_eye = x['right eye'].numpy()[0]
    ax_right_eye.imshow(right_eye, cmap='gray')
    ax_right_eye.set_title('Right Eye')
    
    fig.subplots_adjust(hspace=0.5)
    

    #############
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
  filename = args.dataset
  # filename is "{placeId}/{userId}/{screenId}/train.npz"
  # extract the placeId, userId, and screenId
  parts = os.path.split(filename)[0].split(os.path.sep)
  placeId, userId, screenId = parts[-3], parts[-2], parts[-1]
  # use the stats to get the numeric values of the placeId, userId, and screenId  
  ds = CDataSampler(
    CSamplesStorage(
      placeId=-1, # don't care about the ids
      userId=-1,
      screenId=-1
    ),
    defaults=dict(timesteps=args.steps, stepsSampling='last'),
    batch_size=args.batch_size, minFrames=args.steps
  )
  ds.addBlock(Utils.datasetFrom(filename))
  validSamples = ds.validSamples()
  print(len(validSamples))

  streamSettings = dict(
    take='clean', batchSize=args.batch_size, indices=validSamples, 
  )
  Limit = args.limit if args.limit is not None else len(validSamples)
  
  visualize = visualizer()
  with imageio.get_writer(args.output, mode='I', fps=args.fps) as writer:
    with tqdm.tqdm(total=len(validSamples)) as pbar:
      idx = 0
      for x, y in samplesStream(ds, **streamSettings):
        try:
          frame = visualize(x, y)
          writer.append_data(frame)
          idx += 1
          if idx >= Limit: break
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
  parser.add_argument('--limit', type=int, help='Limit the number of frames', default=None)
  parser.add_argument('--steps', type=int, help='Number of timesteps', default=1)
  parser.add_argument('--batch-size', type=int, help='Batch size', default=128)
  parser.add_argument('--output', type=str, help='Output file name', default='visualize.avi')
  parser.add_argument('--fps', type=int, help='Frames per second', default=3)

  args = parser.parse_args()
  main(args)
  pass
