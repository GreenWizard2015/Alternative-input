import math
import numpy as np
from collections import defaultdict
import cv2
import random
import os
import glob

def limitGPUMemory(memory_limit):
  import tensorflow as tf
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
  )
  return 

FACE_PARTS_CONNECTIONS = {
  'lips': [
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
  ],
  'left eye': [
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
  ],
  'left eyebrow': [
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
  ],
  'right eye': [
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
  ],
  'right eyebrow': [
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
  ],
  'face oval': [
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)
  ]
}

COLORS = {
  'lips': (255, 255, 255),
  'left eye': (255, 255, 0),
  'right eye': (0, 255, 255),
  'face oval': (0, 255, 0),
  'right eyebrow': (255, 0, 0),
  'left eyebrow': (255, 0, 0),
}
###################################
INDEX_TO_PART = {}
PART_TO_INDECES = defaultdict(set)
for k, pairs in FACE_PARTS_CONNECTIONS.items():
  for p in pairs:
    for i in p:
      INDEX_TO_PART[i] = k
      PART_TO_INDECES[k].add(i)
###################################
def decodeLandmarks(landmarks, HW, VISIBILITY_THRESHOLD, PRESENCE_THRESHOLD):
  H, W = HW
  points = {}
  for idx, mark in enumerate(landmarks.landmark):
    if (
      (mark.HasField('visibility') and (mark.visibility < VISIBILITY_THRESHOLD)) or
      (mark.HasField('presence') and (mark.presence < PRESENCE_THRESHOLD))
    ):
      continue
    
    x_px = min(math.floor(mark.x * W), W - 1)
    y_px = min(math.floor(mark.y * H), H - 1)
    points[idx] = (x_px, y_px)
    continue
  return points

def tracked2sample(data):
  points = np.full((468, 2), fill_value=-1, dtype=np.float32)
  for idx, (x, y) in data['face points'].items():
    points[idx, 0] = x
    points[idx, 1] = y
    continue
    
  return {
    'points': points,
    'left eye': cv2.cvtColor(data['left eye'], cv2.COLOR_BGR2GRAY).astype(np.uint8),
    'right eye': cv2.cvtColor(data['right eye'], cv2.COLOR_BGR2GRAY).astype(np.uint8),
  }

def samples2inputs(samples, dropout=0.0):
  processPoints = lambda x: x
  if 0 < dropout:
    def F(x):
      x = x.copy()
      index = np.where(np.all(-1 < x, axis=-1))[0]
      np.random.shuffle(index)
      index = index[:random.randint(0, int(len(index) * dropout)) + 1]
      if 0 < len(index):
        x[index] = -1
      return x
    processPoints = F
    
  return (
    np.array([processPoints(x['points']) for x in samples], np.float32),
    np.array([x['left eye'] for x in samples], np.float32) / 255.0,
    np.array([x['right eye'] for x in samples], np.float32) / 255.0,
  )

def emptyInputs():
  points = np.full((1, 468, 2), fill_value=-1, dtype=np.float32)
  eye = np.zeros((1, 32, 32, 1), dtype=np.float32)
  return (points, eye, eye)

def dataFromFolder(folder):
  for fn in glob.iglob(os.path.join(folder, '*.npz')):
    with np.load(fn) as data:
      yield data
    continue
  return

def datasetFromFolder(folder):
  dataset = defaultdict(list)
  for data in dataFromFolder(folder):
    for k, v in data.items():
      dataset[k].append(v)
    continue
  return {k: np.concatenate(v, axis=0) for k, v in dataset.items()}