#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
from Core.CFakeModel import CFakeModel
import os, imageio
import matplotlib.pyplot as plt
import Utils
import tensorflow as tf
from Core.CSamplesStorage import CSamplesStorage
from Core.CDataSampler import CDataSampler
import matplotlib.patches as patches

import tensorflow_probability as tfp
tfd = tfp.distributions

import seaborn as sns

np.set_printoptions(precision=4, threshold=7777, suppress=True, linewidth=120)
folder = os.path.dirname(__file__)

model = CFakeModel(
  model='simple', depth=15,
  F2LArgs={'steps': 5},
  weights={'folder': folder}, 
  trainable=False
)

folder = os.path.dirname(__file__)
ds = CDataSampler( 
  CSamplesStorage(),
  defaults={
    'timesteps': model.timesteps,
    'maxT': 1.0,
    'pointsDropout': 0.0,
    'eyesDropout': 0.0,
  }
)
ds.addBlock(Utils.datasetFromFolder(os.path.join(folder, 'Dataset')))

def batched(BATCH = 128):
    batch = (([], [], []), ([], ))
    for ind in range(51111, len(ds)):
        data = ds.sampleById(ind, stepsSampling='last')
        if data is None: continue
        X, (Y,) = data
        for B, v in zip(batch[0], X):
            B.append(v[0])
        batch[1][0].append(Y[0])

        if len(batch[0][0]) < BATCH: continue
        yield(
          [np.array(x, np.float32) for x in batch[0]],
          [np.array(batch[1][0], np.float32)]
        )
        batch = (([], [], []), ([], ))
        continue
    if 0 < len(batch[0]):
        yield(
          [np.array(x, np.float32) for x in batch[0]],
          [np.array(batch[1][0], np.float32)]
        )
    return

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
    
    pred = model(X)
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
          try:
            sampled = gmm.sample(1e4).numpy()
            sns.kdeplot(
                x=sampled[:, 0], y=sampled[:, 1], 
                levels=5, fill=True, gridsize=30, cut=0, thresh=0.1
            )
          except Exception as e:
            print(e)
  
          mu = pred['gmm']['mu'][ind].numpy()
          alpha = pred['gmm']['alpha'][ind].numpy()
          scale_tril = pred['gmm']['scale_tril'][ind].numpy()
          for m, a in zip(mu, alpha):
            plt.plot(
              m[0], m[1], 
              'o', markersize=1, color='blue', alpha=a
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
          
          plt.subplots_adjust(bottom=0.25, top=.95)
          ax = plt.gca().inset_axes([0, -.3, .3, 0.3])
          ax.axis('off')
          ax.imshow(X[1][ind, -1], 'gray', interpolation='nearest')
  
          ax = plt.gca().inset_axes([0.7, -.3, .3, 0.3])
          ax.axis('off')
          ax.imshow(X[2][ind, -1], 'gray', interpolation='nearest')
  
          ax = plt.gca().inset_axes([0.33, -.3, .3, 0.3])
          ax.axis('off')
          pts = X[0][ind, -1]
          ax.plot(pts[None, :, 0], 1 - pts[None, :, 1], 'o', markersize=1)
          ax.set_xlim([0, 1])
          ax.set_ylim([0, 1])
          
          # plt.axis('equal')
          plt.axis([-0.1, 1.1, -0.1, 1.1])
          plt.gcf().canvas.draw()
          ncols, nrows = plt.gcf().canvas.get_width_height()
          image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
          writer.append_data(image.reshape(nrows, ncols, 3))
        except Exception as e:
          print(e)
        plt.clf()
        frame += 1
        continue
    if 3000 < frame: break
    continue
plt.close()