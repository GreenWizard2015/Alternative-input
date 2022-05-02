import cv2
import mediapipe
import Utils
import numpy as np

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
    
    self._pose = mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return self

  def __exit__(self, type, value, traceback):
    self._capture.release()
    self._pose.close()
    return

  def track(self):
    ret, frame = self._capture.read()
    # Make detection
    results = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = frame
    facePoints, LE, RE = self._processFace(results, image)
    res = {
      'raw': image,
      'size': image.shape[:2],
      'face points': facePoints,
      'right eye': self._extract(image, RE),
      'right eye visible': 5 < len(RE),
      'left eye': self._extract(image, LE),
      'left eye visible': 5 < len(LE),
    }
    return res
  
  def _extract(self, image, pts):
    sz = (32, 32)
    if len(pts) < 5:
      return np.zeros((*sz, image.shape[-1]), image.dtype)
    
    XY = np.array(pts)
    crop = image[
      XY[:, 1].min()-5:XY[:, 1].max()+5,
      XY[:, 0].min()-5:XY[:, 0].max()+5,
    ]
    if np.min(crop.shape[:2]) < 8:
      return np.zeros((*sz, image.shape[-1]), image.dtype)
    return cv2.resize(crop, sz)
  
  def _processFace(self, pose, image):
    facePoints = {}
    LE = []
    RE = []

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
    return(facePoints, LE, RE)