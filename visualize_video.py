#!/usr/bin/env python
# -*- coding: utf-8 -*-.
# TODO: update this script (currently it is not working in any way)
from collections import defaultdict
import numpy as np
from Core.CCoreModel import CCoreModel
import os, imageio
import matplotlib.pyplot as plt
import Core.Utils as Utils
import tensorflow as tf
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
import matplotlib.patches as patches

import tensorflow_probability as tfp
from Core.CGMModel import CGMModel
tfd = tfp.distributions

###################
GRID_HW = 500
GRID_MIN = -0.1
GRID_MAX = 1.1
def makeGrid():
  import numpy as np
  xy = np.linspace(GRID_MIN, GRID_MAX, GRID_HW)
  XY = np.meshgrid(xy, xy)
  res = np.concatenate([x.reshape(-1, 1) for x in XY], axis=-1)
  return tf.constant(res, tf.float32)
gridTF = makeGrid()

def distr2image(distr):
  lp = distr.log_prob(gridTF)
  lp = tf.maximum(0., lp)
  lp = tf.reshape(lp, (-1,))
  return lp
###################

np.set_printoptions(precision=4, threshold=7777, suppress=True, linewidth=120)
folder = os.path.join(os.path.dirname(__file__), 'Data')

gmm = CGMModel(
  F2LArgs=dict(steps=5, contexts=[11, 20, 99, 514]),
  weights={'folder': folder},
  trainable=False,
  useDiscriminator=False,
)
model = CCoreModel(gmm)

ds = CDataSampler( 
  CSamplesStorage(),
  defaults={
    'timesteps': 5,
    'maxT': 1.0,
    'pointsDropout': 0.0,
    'eyesDropout': 0.0,
    'contextShift': [9, 18, 96, 512],
    'shiftsN': -1,
  }
)
ds.addBlock(Utils.datasetFrom(os.path.join(folder, 'test.npz')))

def batched(BATCH = 128):
  batchIds = []
  for ind in range(len(ds)):
    if not ds.checkById(ind, stepsSampling='last'): continue
    batchIds.append(ind)
    if len(batchIds) < BATCH: continue
    yield ds.sampleByIds(batchIds, stepsSampling='last')[0]
    batchIds = []
    continue
  if 0 < len(batchIds):
    yield ds.sampleByIds(batchIds, stepsSampling='last')[0]
  return

grid = gridTF.numpy()
frame = 0
plt.figure(figsize=(8, 8))
started = False
with imageio.get_writer('movie.avi', mode='I') as writer:
  for B in batched():
    print('......')
    X, (Y,) = B
    gtY = Y[:, -1, 0]

    if not started:
      started = np.any(
        np.logical_or(gtY[:, 0] < .025, 0.975 < gtY[:, 0])
      )
    if not started: continue
    
    pred = model(X['clean'])
    for i in range(len(gtY)):
        print(frame)
        try:
          ind = i
          for j in range(Y.shape[1]):
            pos = Y[ind, j, 0]
            plt.plot([pos[0]], [pos[1]], 'o', color='black', markersize=2)

          pos = gtY[ind]
          plt.plot([pos[0]], [pos[1]], 'o', color='red', markersize=2)
          
          pos = pred['coords'][ind]
          plt.plot([pos[0]], [pos[1]], 'o', color='green', markersize=2)
          ###############
          gmm = pred['distribution'][ind]
          v = distr2image(gmm).numpy()
          msk = np.where(0 < v)
          plt.scatter(
              grid[msk, 0], grid[msk, 1],
              c=v[msk],
              # vmax=5., vmin=0.,
              cmap='jet'
          )
  
          mu = pred['gmm']['mu'][ind].numpy()
          alpha = pred['gmm']['alpha'][ind].numpy()
          scale_tril = pred['gmm']['scale_tril'][ind].numpy()
          for m, a in zip(mu, alpha):
            plt.plot(
              m[0], m[1], 
              'o', markersize=1, color='blue', alpha=a
            )
          ###############
          if 'D mu' in pred:
            mu = pred['D mu'][ind].numpy()
            for m in mu:
              plt.plot(
                m[0], m[1], 
                'o', markersize=3, color='magenta'
              )
          ###############
          for i in range(5):
              d = i * 0.1
              plt.gca().add_patch(
                patches.Rectangle(
                  (d,d), 1-2*d, 1-2*d,
                  linewidth=1,edgecolor='r',facecolor='none'
                )
              )
          
        #   plt.subplots_adjust(bottom=0.25, top=.95)
        #   ax = plt.gca().inset_axes([0, -.3, .3, 0.3])
        #   ax.axis('off')
        #   ax.imshow(X[1][ind, -1], 'gray', interpolation='nearest')
  
        #   ax = plt.gca().inset_axes([0.7, -.3, .3, 0.3])
        #   ax.axis('off')
        #   ax.imshow(X[2][ind, -1], 'gray', interpolation='nearest')
  
        #   ax = plt.gca().inset_axes([0.33, -.3, .3, 0.3])
        #   ax.axis('off')
        #   pts = X[0][ind, -1]
        #   ax.plot(pts[None, :, 0], 1 - pts[None, :, 1], 'o', markersize=1)
        #   ax.set_xlim([0, 1])
        #   ax.set_ylim([0, 1])
          
          # plt.axis('equal')
          plt.axis([-0.1, 1.1, -0.1, 1.1])
          plt.gcf().canvas.draw()
          ncols, nrows = plt.gcf().canvas.get_width_height()
          image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
          writer.append_data(image.reshape(nrows, ncols, 3))
        except Exception as e:
          raise(e)
        plt.clf()
        frame += 1
        continue
    if 3000 < frame: break
    continue
plt.close()