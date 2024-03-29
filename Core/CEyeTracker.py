import cv2
import mediapipe
import Core.Utils as Utils
import numpy as np
import time

class CEyeTracker:
  def __init__(self):
    self._PRESENCE_THRESHOLD = self._VISIBILITY_THRESHOLD = 0.5

    self._leftEyeIdx = np.array(list(Utils.PART_TO_INDECES['left eye']), np.int32)
    self._rightEyeIdx = np.array(list(Utils.PART_TO_INDECES['right eye']), np.int32)
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
    BGR = True
    results = self._pose.process(frame)
    image = frame
    facePoints, LE, RE, lipsDistancePx = self._processFace(results, image)
    
    REVisible = 5 < len(RE)
    LEVisible = 5 < len(LE)
    # if eyes are invisible, try to find RGB
    if not(REVisible or LEVisible):
      BGR = False
      results = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      facePoints, LE, RE, lipsDistancePx = self._processFace(results, image)
      pass

    # if eyes are still invisible, return None
    if not(REVisible or LEVisible):
      return None

    return {
      # main data
      'time': time.time(),
      'face points': facePoints,
      'left eye': self._extract(image, LE, BGR),
      'right eye': self._extract(image, RE, BGR),
      # misc
      'lips distance': lipsDistancePx,
      'raw': frame,
    }
  
  def _extract(self, image, pts, isBGR):
    sz = (32, 32)
    padding = 5
    if len(pts) < 5:
      return np.zeros(sz, np.uint8)

    A = (pts.min(axis=0) - padding).clip(min=0)
    B = pts.max(axis=0) + padding
    B = np.minimum(B, image.shape[:2])
    
    crop = image[ A[1]:B[1], A[0]:B[0], ]
    if np.min(crop.shape[:2]) < 8:
      return np.zeros(sz, np.uint8)
    
    crop = cv2.resize(crop, sz)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY if isBGR else cv2.COLOR_RGB2GRAY)
    return crop.astype(np.uint8)
  
  def _processFace(self, pose, image):
    facePoints = {}
    LE = []
    RE = []
    lipsDistancePx = 0

    landmarks = pose.face_landmarks
    if landmarks:
      dims = np.array(image.shape[:2])[::-1]
      facePoints = Utils.decodeLandmarks(landmarks, self._VISIBILITY_THRESHOLD, self._PRESENCE_THRESHOLD)
      
      LE = facePoints[self._leftEyeIdx]
      RE = facePoints[self._rightEyeIdx]
      # remove invisible points
      LE = LE[LE[:, 0] != -1]
      RE = RE[RE[:, 0] != -1]
      # convert to pixels
      LE = np.multiply(LE, dims[None]).astype(np.int32)
      RE = np.multiply(RE, dims[None]).astype(np.int32)

      # measure distance between lips
      lipsA = np.array(facePoints[17])
      lipsB = np.array(facePoints[0])
      lipsDistancePx = np.linalg.norm(np.multiply(lipsA - lipsB, dims))
      pass
    return(facePoints, LE, RE, lipsDistancePx)