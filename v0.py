import cv2
import mediapipe as mp
import numpy as np
import math
import pywinauto
import win32api
from _collections import defaultdict
from pywinauto import win32functions, win32defines
import time
import keyboard
import win32gui

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

INDEX_TO_PART = {}
PART_TO_INDECES = defaultdict(set)
for k, pairs in FACE_PARTS_CONNECTIONS.items():
  for p in pairs:
    for i in p:
      INDEX_TO_PART[i] = k
      PART_TO_INDECES[k].add(i)

pywinauto.timings.Timings.after_setcursorpos_wait = 0.0
pywinauto.timings.Timings.after_clickinput_wait = 0.0

SCREEN_W = win32functions.GetSystemMetrics(win32defines.SM_CXSCREEN)
SCREEN_H = win32functions.GetSystemMetrics(win32defines.SM_CYSCREEN)

print(SCREEN_W, SCREEN_H)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.holistic

PRESENCE_THRESHOLD = VISIBILITY_THRESHOLD = 0.5
cap = cv2.VideoCapture(0)
##
class CCalibration:
  def __init__(self):
    self.reset()
    return
  
  def isDone(self):
    return 6.0 < (time.time() - self._startT)
  
  def reset(self):
    self._startT = time.time()
    return
  
  def calibrate(self, landmarks, points):
    
    return
##
oldHandPos = np.array([0.0, 0.0])
lastVisibleT = 0
acceleration = 1.0
mouthWasOpen = False
calibration = CCalibration()

def mouseMode(points):
  global oldHandPos, acceleration, mouthWasOpen
  LEPoints = [points[i] for i in PART_TO_INDECES['left eye'] if i in points]
  REPoints = [points[i] for i in PART_TO_INDECES['right eye'] if i in points]
  
  mouthH = np.linalg.norm(np.subtract(points[0], points[17]))
  mouthOpen = 30 < mouthH
  LEMean = np.mean(LEPoints, axis=0).astype(np.int)
  REMean = np.mean(REPoints, axis=0).astype(np.int)
  
  mean = np.mean([LEMean, REMean], axis=0).astype(np.int)
  cv2.circle(image, tuple(mean), 5, (0, 255, 0))
  
  diff = mean - oldHandPos
  D = np.linalg.norm(diff)
  if D < 15:
    acceleration *= 1.0 - math.sqrt(D) * .1
  else:
    acceleration *= 1.0 + math.sqrt(D - 14) * .1
  acceleration = np.clip(acceleration, .1, 1.)
  
  newPos = oldHandPos + .25 * diff
  cv2.circle(image, tuple(newPos.astype(np.int)), 5, (255, 0, 0))
  oldHandPos = newPos
  
  xy = np.array(win32api.GetCursorPos(), np.float)
  
  if mouthOpen and not mouthWasOpen:
    pywinauto.mouse.click('left', tuple(xy.astype(np.int)))

  if not mouthOpen:
    try:
      l, t, r, b = win32gui.GetWindowRect(win32gui.GetForegroundWindow())
      w = abs(r - l)
      h = abs(t - b)
      xy = xy + 5. * diff * np.array([-w / 600, h / 200]) * acceleration
      xy = np.array([np.clip(xy[0], l, r), np.clip(xy[1], t, b)])
      pywinauto.mouse.move(tuple(xy.astype(np.int)))
    except:
      pass
  mouthWasOpen = mouthOpen
  return

KEYBOARD = {'cursor': np.array([0, 0]), 'key': ''}

def onHit(X):
  if 'up' == X.event_type:
    print(KEYBOARD['key'])
  return

keyboard.hook_key('Insert', onHit, suppress=True)

def keyboardMode(landmarks, points):
  global oldHandPos, acceleration, mouthWasOpen
  Letters = {
    'EN': [
      'qwertyuiop[]', 'asdfghjkl;\'\\', 'zxcvbnm,./'
    ]
  }
  img = np.full((220, 750, 3), 255, np.uint8)
  
  LEPoints = [points[i] for i in PART_TO_INDECES['left eye'] if i in points]
  REPoints = [points[i] for i in PART_TO_INDECES['right eye'] if i in points]
  
  mouthH = np.linalg.norm(np.subtract(points[0], points[17]))
  mouthOpen = 30 < mouthH
  LEMean = np.mean(LEPoints, axis=0).astype(np.int)
  REMean = np.mean(REPoints, axis=0).astype(np.int)
  
  mean = np.mean([LEMean, REMean], axis=0).astype(np.int)
  
  diff = mean - oldHandPos
  oldHandPos = oldHandPos + .25 * diff
  
  D = np.linalg.norm(diff)
  if D < 15:
    acceleration *= 1.0 - math.sqrt(D) * .1
  else:
    acceleration *= 1.0 + math.sqrt(D - 14) * .1
  acceleration = np.clip(acceleration, .5, 1.5)
  
  xy = KEYBOARD['cursor']
  xy = xy + 2. * diff * np.array([-2., 2.]) * acceleration
  H, W, _ = img.shape
  xy = np.array([ np.clip(xy[0], 20, W), np.clip(xy[1], 20, H) ]).astype(np.int)
  
  nearestLetter = np.array([0., 0.])
  nearestLetterK = ''
  nearestD = float('inf')
  width = 60
  height = 60
  th = 3
  for row, keys in enumerate(Letters['EN']):
    for ind, k in enumerate(keys):
      x = 25 + (ind * width)
      y = 25 + (row * height)
      # Keys
      cv2.rectangle(img, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)
      
      cx = x + width / 2 + th
      cy = y + height / 2 + th
      D = np.linalg.norm(np.subtract((cx, cy), xy))
      if D < nearestD:
        nearestD = D
        nearestLetter = (cx, cy)
        nearestLetterK = k
#       if (row == KEYBOARD['Y']) and (ind == KEYBOARD['X']):
#         cv2.rectangle(img, (x + th, y + th), (x + width - th, y + height - th), (200, 200, 200), -1)
      
      # Text settings
      font_letter = cv2.FONT_HERSHEY_PLAIN
      font_scale = 3
      font_th = 2
      text_size = cv2.getTextSize(k, font_letter, font_scale, font_th)[0]
      width_text, height_text = text_size[0], text_size[1]
      text_x = int((width - width_text) / 2) + x
      text_y = int((height + height_text) / 2) + y
      cv2.putText(img, k, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)
  
  xy = xy + .2 * np.subtract(nearestLetter, xy)
  xy = np.array([ np.clip(xy[0], 20, W), np.clip(xy[1], 20, H) ]).astype(np.int) 
  KEYBOARD['cursor'] = xy
  KEYBOARD['key'] = nearestLetterK
  
#   mouthOpen = win32api.GetAsyncKeyState(ord(' ')) == 1
#   if mouthOpen and not mouthWasOpen:
#     # keyboard.write(nearestLetterK)
#     pywinauto.keyboard.send_keys(nearestLetterK, pause=0.0)
#     print(nearestLetterK)
#   mouthWasOpen = mouthOpen
  
  cv2.circle(img, tuple(xy), 5, (255, 0, 255), -1)
  cv2.imshow('kb', img)
  return 

with mp_pose.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    ret, frame = cap.read()
    
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     V = hsv[:, :, 2].astype(np.float)
#     V = np.clip(V * 2., 0.0, a_max=255.)
#     hsv[:, :, 2] = V.astype(np.uint)
#     
#     frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Make detection
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = frame
    H, W, _ = image.shape
    # Render detections
    landmarks = results.face_landmarks
    if landmarks:
      points = {}
      for idx, mark in enumerate(landmarks.landmark):
        if ((mark.HasField('visibility') and
             mark.visibility < VISIBILITY_THRESHOLD) or
            (mark.HasField('presence') and
             mark.presence < PRESENCE_THRESHOLD)):
          continue
        
        x_px = min(math.floor(mark.x * W), W - 1)
        y_px = min(math.floor(mark.y * H), H - 1)
        points[idx] = pt = (x_px, y_px)
        
        clr = (128, 128, 128)
        if idx in INDEX_TO_PART:
          clr = COLORS[INDEX_TO_PART[idx]]
        cv2.circle(image, pt, 2, clr)

      if not calibration.isDone():
        calibration.calibrate(landmarks, points)
        
      mouseMode(points)
      #keyboardMode(landmarks, points)
      pass

    cv2.imshow('img', image)
    if cv2.waitKey(10) & 0xFF == 27:
      break

  cap.release()
  cv2.destroyAllWindows()
