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
    
    self._pose = mediapipe.solutions.face_mesh.FaceMesh(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
      max_num_faces=1,
      refine_landmarks=True,
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

    leftEye, leftEyeArea = self._extract(image, LE, BGR)
    rightEye, rightEyeArea = self._extract(image, RE, BGR)
    return {
      # main data
      'time': time.time(),
      'face points': facePoints,
      'left eye': leftEye,
      'right eye': rightEye,
      # misc
      'lips distance': lipsDistancePx,
      'left eye area': leftEyeArea,
      'right eye area': rightEyeArea,
      'raw': frame,
    }
  
  def _circleROI(self, pts, padding):
    # find center  
    center = pts.mean(axis=0).astype(np.int32)[None]
    assert center.shape == (1, 2)
    # find radius
    diffs = pts - center
    dist = np.sum(diffs**2, axis=1)
    radius = np.sqrt(np.max(dist))
    if radius < 5: return None
    radius = int(radius * padding)
    A = center - radius
    B = center + radius
    res = np.concatenate([A, B], axis=0)
    assert res.shape == (2, 2)
    return res

  def _extract(self, image, pts, isBGR):
    sz = (32, 32)
    EMPTY = np.zeros(sz, np.uint8), None
    if len(pts) < 1: return EMPTY

    HW = np.array(image.shape[:2][::-1])
    roi = self._circleROI(pts, padding=1.25)
    if roi is None: return EMPTY
    A, B = roi
    A = A.clip(min=0, max=HW)
    B = B.clip(min=0, max=HW)
    
    rect = np.array([A, B], np.float32) / HW
    crop = image[ A[1]:B[1], A[0]:B[0], ]
    if np.min(crop.shape[:2]) < 8:
      return np.zeros(sz, np.uint8), rect
    
    crop = cv2.resize(crop, sz)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY if isBGR else cv2.COLOR_RGB2GRAY)
    return crop.astype(np.uint8), rect
  
  def _processFace(self, pose, image):
    facePoints = {}
    LE = []
    RE = []
    lipsDistancePx = 0

    if pose.multi_face_landmarks is None: return (facePoints, LE, RE, lipsDistancePx)
    landmarks = pose.multi_face_landmarks[0]
    if landmarks:
      dims = np.array(image.shape[:2])[::-1][None]
      facePoints = Utils.decodeLandmarks(landmarks, self._VISIBILITY_THRESHOLD, self._PRESENCE_THRESHOLD)
      
      LE = np.multiply(facePoints[self._leftEyeIdx], dims).astype(np.int32)
      RE = np.multiply(facePoints[self._rightEyeIdx], dims).astype(np.int32)

      # measure distance between lips
      lipsA = np.array(facePoints[17, :2])
      lipsB = np.array(facePoints[0, :2])
      lipsDistancePx = np.linalg.norm(np.multiply(lipsA - lipsB, dims))
      pass
    return(facePoints, LE, RE, lipsDistancePx)