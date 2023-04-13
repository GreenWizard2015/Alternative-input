import cv2
import mediapipe
import Core.Utils as Utils
import numpy as np
import time

class CEyeTracker:
  def __init__(self):
    self._PRESENCE_THRESHOLD = self._VISIBILITY_THRESHOLD = 0.5
    return

  def __enter__(self):
    cap = self._capture = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -5)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    
    self._pose = mediapipe.solutions.holistic.Holistic(
      min_detection_confidence=self._PRESENCE_THRESHOLD,
      min_tracking_confidence=self._VISIBILITY_THRESHOLD
    )
    return self

  def __exit__(self, type, value, traceback):
    self._capture.release()
    self._pose.close()
    return

  def track(self):
    ret, frame = self._capture.read()
    # Make detection in BGR space
    results = self._pose.process(frame)
    image = frame
    facePoints, LE, RE, lipsDistancePx = self._processFace(results, image)
    
    REVisible = 5 < len(RE)
    LEVisible = 5 < len(LE)
    # if eyes are invisible, try to find RGB
    if not(REVisible or LEVisible):
      results = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      facePoints, LE, RE, lipsDistancePx = self._processFace(results, image)
      pass

    res = {
      # main data
      'time': time.time(),
      'face points': facePoints,
      'right eye': self._extract(image, RE),
      'left eye': self._extract(image, LE),
      # misc
      'lips distance': lipsDistancePx,
      'raw': frame,
    }
    return res
  
  def _extract(self, image, pts):
    sz = (32, 32)
    padding = 5
    if len(pts) < 5:
      return np.zeros((*sz, image.shape[-1]), image.dtype)
    
    XY = np.array(pts)

    A = (XY.min(axis=0) - padding).clip(min=0)
    B = XY.max(axis=0) + padding
    B = np.minimum(B, image.shape[:2][::-1])
    
    crop = image[ A[1]:B[1], A[0]:B[0], ]
    if np.min(crop.shape[:2]) < 8:
      return np.zeros((*sz, image.shape[-1]), image.dtype)
    return cv2.resize(crop, sz)
  
  def _processFace(self, pose, image):
    facePoints = {}
    LE = []
    RE = []
    lipsDistancePx = 0

    landmarks = pose.face_landmarks
    if landmarks:
      H, W = image.shape[:2]
      face_points_scaled = Utils.decodeLandmarks(landmarks, image.shape[:2], self._VISIBILITY_THRESHOLD, self._PRESENCE_THRESHOLD)
      facePoints = {
        idx: (x / W, y / H)
        for idx, (x, y) in face_points_scaled.items()
      }
      
      for idx, pt in face_points_scaled.items():
        if 'right eye' == Utils.INDEX_TO_PART.get(idx, ''):
          RE.append(pt)
        if 'left eye' == Utils.INDEX_TO_PART.get(idx, ''):
          LE.append(pt)
        continue

      lipsDistancePx = np.linalg.norm(np.subtract(face_points_scaled[17], face_points_scaled[0]))
      pass
    return(facePoints, LE, RE, lipsDistancePx)