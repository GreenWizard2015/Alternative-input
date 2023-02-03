#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
import os
import Core.Utils as Utils
Utils.setupGPU(memory_limit=1024)

MAIN_FOLDER = os.path.join(os.path.dirname(__file__), 'Data')

def process(src, dest):
  dataset = Utils.datasetFrom(src)
  np.savez_compressed(dest, **dataset)
  return

process(
  os.path.join(MAIN_FOLDER, 'Dataset'),
  os.path.join(MAIN_FOLDER, 'train.npz'),
)

process(
  os.path.join(MAIN_FOLDER, 'Dataset-test'),
  os.path.join(MAIN_FOLDER, 'test.npz'),
)