"""Eye tracker using MediaPipe face mesh.

Provides real-time face mesh tracking and eye region extraction from webcam
or video input using MediaPipe's FaceMesh solution.
"""

from typing import Optional, Tuple, Any, Dict
import cv2
import mediapipe
from mediapipe.python.solution_base import SolutionOutputs
import numpy as np
import time
import Core.Utils as Utils
from Core.logging_config import get_logger
from Core.Constants import (
    FACEMESH_VISIBILITY_THRESHOLD,
    FACEMESH_PRESENCE_THRESHOLD,
    FACEMESH_DETECTION_CONFIDENCE,
    FACEMESH_TRACKING_CONFIDENCE,
    FACEMESH_MAX_FACES,
    FACEMESH_REFINE_LANDMARKS,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_AUTO_EXPOSURE,
    CAMERA_AUTOFOCUS,
    CAMERA_AUTO_WB,
    EYE_TRACKER_MIN_LANDMARK_COUNT,
    EYE_TRACKER_MIN_ROI_RADIUS,
    EYE_TRACKER_ROI_PADDING,
    EYE_TRACKER_IMAGE_SIZE,
    EYE_TRACKER_INTERMEDIATE_SIZE,
    EYE_REGION_MIN_SIZE,
)

logger = get_logger(__name__)


class EyeTracker:
    """Real-time face mesh and eye tracker using MediaPipe.

    Captures video from webcam, detects face mesh landmarks, and extracts
    eye region images. Uses MediaPipe FaceMesh for landmark detection and
    OpenCV for image processing.

    Attributes:
        _webcam: Webcam device index
        _capture: OpenCV VideoCapture object
        _pose: MediaPipe FaceMesh solution instance
        _VISIBILITY_THRESHOLD: Minimum landmark visibility to include
        _PRESENCE_THRESHOLD: Minimum landmark presence score
        _left_eye_idx: Indices of left eye landmarks
        _right_eye_idx: Indices of right eye landmarks
    """

    def __init__(self, webcam: int = 0) -> None:
        """Initialize eye tracker.

        Args:
            webcam: Webcam device index (default: 0 for primary camera)

        Raises:
            ValueError: If webcam index is negative
        """
        if webcam < 0:
            raise ValueError(f"webcam index must be non-negative, got {webcam}")
        self._webcam = webcam
        self._PRESENCE_THRESHOLD = FACEMESH_PRESENCE_THRESHOLD
        self._VISIBILITY_THRESHOLD = FACEMESH_VISIBILITY_THRESHOLD

        self._left_eye_idx = np.array(
            list(Utils.PART_TO_INDICES["left eye"]), dtype=np.int32
        )
        self._right_eye_idx = np.array(
            list(Utils.PART_TO_INDICES["right eye"]), dtype=np.int32
        )

    def __enter__(self) -> "EyeTracker":
        """Enter context manager: initialize video capture and face mesh.

        Configures webcam settings for optimal tracking and initializes
        MediaPipe FaceMesh solution.

        Returns:
            Self for context manager protocol
        """
        cap = self._capture = cv2.VideoCapture(self._webcam)

        cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=CAMERA_WIDTH)
        cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=CAMERA_HEIGHT)
        cap.set(propId=cv2.CAP_PROP_AUTO_EXPOSURE, value=CAMERA_AUTO_EXPOSURE)
        cap.set(propId=cv2.CAP_PROP_AUTOFOCUS, value=CAMERA_AUTOFOCUS)
        cap.set(propId=cv2.CAP_PROP_AUTO_WB, value=CAMERA_AUTO_WB)

        self._pose = mediapipe.solutions.face_mesh.FaceMesh(
            min_detection_confidence=FACEMESH_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACEMESH_TRACKING_CONFIDENCE,
            max_num_faces=FACEMESH_MAX_FACES,
            refine_landmarks=FACEMESH_REFINE_LANDMARKS,
        )
        return self

    def __exit__(
        self,
        _exc_type: Any,  # type: ignore[unused-argument]
        _exc_val: Any,  # type: ignore[unused-argument]
        _exc_tb: Any,  # type: ignore[unused-argument]
    ) -> None:
        """Exit context manager: release resources.

        Closes video capture and face mesh solution.

        Args:
            _exc_type: Exception type (if any, required by context manager protocol).
            _exc_val: Exception value (if any, required by context manager protocol).
            _exc_tb: Exception traceback (if any, required by context manager protocol).

        Example:
            >>> with EyeTracker(0) as tracker:
            ...     result = tracker.track()
        """
        self._capture.release()
        self._pose.close()

    def track(self) -> Optional[Dict[str, Any]]:
        """Track face and extract eye regions from current frame.

        Captures a frame from webcam, detects face mesh landmarks using MediaPipe,
        and extracts eye images. Handles both BGR and RGB color spaces for robustness.

        Returns:
            Dictionary with tracking results if face is detected, None otherwise.
            Dictionary keys:
                - 'time': Timestamp (float) of capture
                - 'face points': Face mesh points array (478, 2) normalized [0, 1]
                - 'left eye': Left eye image (32, 32) grayscale uint8
                - 'right eye': Right eye image (32, 32) grayscale uint8
                - 'lips distance': Distance between lip keypoints (pixels, float)
                - 'left eye area': Normalized eye area [[x_min, y_min], [x_max, y_max]] [0, 1]
                - 'right eye area': Normalized eye area [[x_min, y_min], [x_max, y_max]] [0, 1]
                - 'raw': Original frame (H, W, 3) uint8

        Example:
            >>> with EyeTracker(0) as tracker:
            ...     result = tracker.track()
            ...     if result is not None:
            ...         left_eye = result['left eye']  # (32, 32)
            ...         face_pts = result['face points']  # (478, 2)
        """
        ret, frame = self._capture.read()
        if not ret:
            return None
        # Make detection in BGR space
        result = self._track_with_color_space(frame=frame, is_bgr=True)
        if result is not None:
            return result

        # if eyes are invisible, try to find RGB
        result = self._track_with_color_space(frame=frame, is_bgr=False)
        return result  # pragma: no cover

    def _track_with_color_space(
        self, frame: np.ndarray, is_bgr: bool
    ) -> Optional[Dict[str, Any]]:
        """Track with specific color space conversion.

        Args:
            frame: Input frame (H, W, 3) uint8
            is_bgr: If True, process as BGR; if False, convert to RGB first

        Returns:
            Dictionary with tracking results or None if face not detected
        """
        image = frame if is_bgr else cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
        results = self._pose.process(frame=image)
        facePoints, LE, RE, lipsDistancePx = self._processFace(
            pose=results, image=frame
        )

        right_eye_visible = EYE_TRACKER_MIN_LANDMARK_COUNT < len(RE)
        left_eye_visible = EYE_TRACKER_MIN_LANDMARK_COUNT < len(LE)

        if not (right_eye_visible or left_eye_visible):
            return None

        leftEye, leftEyeArea = self._extract(image=frame, pts=LE, is_bgr=is_bgr)
        rightEye, rightEyeArea = self._extract(image=frame, pts=RE, is_bgr=is_bgr)
        return {
            # main data
            "time": time.time(),
            "face points": facePoints,
            "left eye": leftEye,
            "right eye": rightEye,
            # misc
            "lips distance": lipsDistancePx,
            "left eye area": leftEyeArea,
            "right eye area": rightEyeArea,
            "raw": frame,
        }

    def _circleROI(self, pts: np.ndarray, padding: float) -> Optional[np.ndarray]:
        """Compute circular region of interest around landmark points.

        Calculates a circle centered on the mean of provided points with radius
        determined by maximum distance from center. Used to define eye region bounds.

        Args:
            pts: Landmark points array of shape (N, 2) in pixel coordinates.
            padding: Radius multiplier for expansion (values > 1.0 expand the region).
                For example, padding=1.5 makes the ROI 1.5x larger than the minimum
                circle containing all points.

        Returns:
            Array of shape (2, 2) with [[x_min, y_min], [x_max, y_max]] bounding
            box in pixel coordinates, or None if computed radius is too small (< 5 px).

        Example:
            >>> eye_pts = np.array([[100, 200], [110, 210], [105, 205]])
            >>> roi = tracker._circleROI(eye_pts, padding=1.5)
            >>> if roi is not None:
            ...     [[x_min, y_min], [x_max, y_max]] = roi
        """
        # find center
        center = pts.mean(axis=0).astype(np.int32)[None]
        if center.shape != (1, 2):
            raise ValueError(
                f"Center shape must be (1, 2), got {center.shape}. Input pts shape: {pts.shape}"
            )
        # find radius
        diffs = pts - center
        dist = np.sum(diffs**2, axis=1)
        radius = np.sqrt(np.max(dist))
        if radius < EYE_TRACKER_MIN_ROI_RADIUS:
            return None
        radius = int(radius * padding)
        A = center - radius
        B = center + radius
        res = np.concatenate([A, B], axis=0)
        if res.shape != (2, 2):
            raise ValueError(
                f"Result shape must be (2, 2), got {res.shape}. This indicates internal computation error"
            )
        return res

    def _extract(
        self, image: np.ndarray, pts: np.ndarray, is_bgr: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract and normalize eye region from image.

        Crops eye region around landmark points using circular ROI, resizes to 32x32,
        converts to grayscale, and returns normalized coordinates for storage.

        Args:
            image: Input image array of shape (H, W, 3) in uint8.
            pts: Eye landmark points of shape (N, 2) in pixel coordinates.
            is_bgr: If True, convert from BGR to grayscale; if False, from RGB.

        Returns:
            Tuple of (eye_image, normalized_rect) where:
                - eye_image: Grayscale eye image (32, 32) as uint8. Returns zero
                  image if extraction fails (too few points, small ROI, etc).
                - normalized_rect: Normalized ROI bounds as 2D array [[x_min, y_min],
                  [x_max, y_max]] in range [0, 1], or None if no valid crop.

        Example:
            >>> tracker = EyeTracker(0)
            >>> image = cv2.imread("face.jpg")  # (480, 640, 3)
            >>> eye_pts = np.array([[100, 200], [120, 210], ...])  # Eye landmarks
            >>> eye_img, norm_rect = tracker._extract(image, eye_pts, is_bgr=True)
            >>> assert eye_img.shape == EYE_TRACKER_IMAGE_SIZE
        """
        EMPTY = np.zeros(EYE_TRACKER_IMAGE_SIZE, np.uint8), None
        if len(pts) < 1:
            return EMPTY

        image_width_height = np.array(image.shape[:2][::-1])
        roi = self._circleROI(pts=pts, padding=EYE_TRACKER_ROI_PADDING)
        if roi is None:
            return EMPTY
        roi_min, roi_max = roi
        roi_min = roi_min.clip(min=0, max=image_width_height)
        roi_max = roi_max.clip(min=0, max=image_width_height)

        rect = np.array([roi_min, roi_max], dtype=np.float32) / image_width_height
        crop = image[
            roi_min[1] : roi_max[1],
            roi_min[0] : roi_max[0],
        ]
        if np.min(crop.shape[:2]) < EYE_REGION_MIN_SIZE:
            return np.zeros(shape=EYE_TRACKER_IMAGE_SIZE, dtype=np.uint8), rect

        crop = cv2.resize(src=crop, dsize=EYE_TRACKER_INTERMEDIATE_SIZE)
        crop = cv2.cvtColor(
            src=crop, code=cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY
        )
        # center crop to final size
        crop_offset = (
            EYE_TRACKER_INTERMEDIATE_SIZE[0] - EYE_TRACKER_IMAGE_SIZE[0]
        ) // 2
        crop = crop[
            crop_offset : crop_offset + EYE_TRACKER_IMAGE_SIZE[0],
            crop_offset : crop_offset + EYE_TRACKER_IMAGE_SIZE[1],
        ]
        if len(crop.shape) == 2:
            crop = crop[..., None]
        return crop.astype(np.uint8), rect

    def _processFace(
        self, pose: SolutionOutputs, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Process face landmarks and extract eye/lips data.

        Decodes face mesh landmarks from MediaPipe FaceMesh results and extracts
        specific regions for eyes and lips. Handles cases where no face is detected
        by returning empty arrays.

        Args:
            pose: MediaPipe FaceMesh detection results (SolutionOutputs) with
                multi_face_landmarks attribute containing detected landmarks.
            image: Input image array (H, W, 3) for reference dimensions.

        Returns:
            Tuple of (face_points, left_eye_pts, right_eye_pts, lips_distance) where:
                - face_points: All 478 landmark points shape (478, 2) normalized [0, 1],
                  or empty array if no face detected.
                - left_eye_pts: Left eye landmark points in pixel coordinates,
                  or empty array if face not detected.
                - right_eye_pts: Right eye landmark points in pixel coordinates,
                  or empty array if face not detected.
                - lips_distance: Euclidean distance between key lip points (pixels),
                  or 0.0 if face not detected.

        Example:
            >>> tracker = EyeTracker(0)
            >>> results = tracker._pose.process(frame=frame)
            >>> face_pts, le, re, lips_dist = tracker._processFace(pose=results, image=frame)
        """
        facePoints = np.array([])
        LE = np.array([])
        RE = np.array([])
        lipsDistancePx = 0.0

        if pose.multi_face_landmarks is None:
            return (facePoints, LE, RE, lipsDistancePx)
        landmarks = pose.multi_face_landmarks[0]
        if landmarks:
            image_dims = np.array(image.shape[:2])[::-1][None]
            facePoints = Utils.decode_landmarks(
                landmarks=landmarks,
                visibility_threshold=self._VISIBILITY_THRESHOLD,
                presence_threshold=self._PRESENCE_THRESHOLD,
            )

            LE = np.multiply(facePoints[self._left_eye_idx], image_dims).astype(
                dtype=np.int32
            )
            RE = np.multiply(facePoints[self._right_eye_idx], image_dims).astype(
                dtype=np.int32
            )

            # measure distance between lips
            lip_point_a = np.array(facePoints[17, :2])
            lip_point_b = np.array(facePoints[0, :2])
            lipsDistancePx = float(
                np.linalg.norm(np.multiply(lip_point_a - lip_point_b, image_dims))
            )
        return (facePoints, LE, RE, lipsDistancePx)
