import numpy as np
from collections import defaultdict
import os
import glob

def isColab():
  try:
    import google.colab
    return True
  except:
    pass
  return False

def setGPUMemoryLimit(limit):
  import tensorflow as tf
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_virtual_device_configuration(
      gpu,
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)]
    )
    continue
  print('GPU memory limit set to %d MB' % limit)
  return

def setupGPU():
  memory_limit = os.environ.get('TF_MEMORY_ALLOCATION_IN_MB', None)
  if memory_limit is not None: setGPUMemoryLimit(int(memory_limit))
  
  # https://github.com/tensorflow/tensorflow/issues/51818#issuecomment-923274891
  try:
    import keras
    keras.layers.recurrent_v2._use_new_code = lambda: True
  except:
    pass
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
FACE_MESH_INVALID_VALUE = -10.0
FACE_MESH_POINTS = 478
def decodeLandmarks(landmarks, VISIBILITY_THRESHOLD, PRESENCE_THRESHOLD):
  points = np.full((FACE_MESH_POINTS, 2), fill_value=FACE_MESH_INVALID_VALUE, dtype=np.float32)
  for idx, mark in enumerate(landmarks.landmark):
    if (
      (mark.HasField('visibility') and (mark.visibility < VISIBILITY_THRESHOLD)) 
      # or (mark.HasField('presence') and (mark.presence < PRESENCE_THRESHOLD))
    ):
      continue
    
    points[idx, 0] = mark.x
    points[idx, 1] = mark.y
    continue
  # clip to 0..1 where not -1
  # msk = points != -1
  # points[msk] = np.clip(points[msk], 0.0, 1.0)
  return points

def tracked2sample(data):
  points = data['face points']
  return {
    'time': data['time'],
    'points': points,
    'left eye': data['left eye'],
    'right eye': data['right eye'],
  }

def samples2inputs(samples):
  return {
    'points': np.array([x['points'] for x in samples], np.float32),
    'left eye': np.array([x['left eye'] for x in samples], np.float32) / 255.0,
    'right eye': np.array([x['right eye'] for x in samples], np.float32) / 255.0,
    'time': np.array([x['time'] for x in samples], np.float32),
  }

def dataFromFolder(folder):
  for fn in glob.iglob(os.path.join(folder, '*.npz')):
    with np.load(fn) as data:
      yield data
    continue
  return

def datasetFrom(folder):
  # if folder is a file, then load it
  res = None
  if os.path.isfile(folder):
    with np.load(folder) as data:
      res = {k: v for k, v in data.items()}
    pass
  else:
    dataset = defaultdict(list)
    for data in dataFromFolder(folder):
      for k, v in data.items():
        dataset[k].append(v)
      continue
    
    res = {k: np.concatenate(v, axis=0) for k, v in dataset.items()}
    pass

  # if empty dict, then return
  if not res: return None
  
  byTime = np.argsort(res['time'])
  # if byTime not equal to np.arange(len(byTime)), then rearrange
  if not all(x == i for i, x in enumerate(byTime)):
    res = {k: v[byTime] for k, v in res.items()}
    pass
  return res

def extractSessions(dataset, TDelta):
  '''
    Args:
      dataset: dataset
      TDelta: time delta in seconds
      
    Returns:
      list of (start, end+1) tuples where start and end are indices of the dataset
  '''
  res = []
  T = 0
  prevSession = 0
  for i, t in enumerate(dataset['time']):
    if TDelta < (t - T):
      if 1 < (i - prevSession):
        res.append((prevSession, i))
      prevSession = i
      pass
    T = t
    continue
  # if last session is not empty, then append it
  if prevSession < len(dataset['time']):
    res.append((prevSession, len(dataset['time'])))
    pass

  # check that end of one session is equal or less than start of the next
  for i in range(1, len(res)):
    assert res[i-1][1] <= res[i][0]
    continue
  return res

def countSamplesIn(folder):
  res = 0
  for fn in glob.iglob(os.path.join(folder, '*.npz')):
    with np.load(fn) as data:
      res += len(data['time'])
    continue
  return res