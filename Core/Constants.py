"""Constants for Core module data sampling and processing.

Centralizes magic numbers and configuration parameters used across
the Core package for data sampling, tracking, and augmentation.
"""

# ============================================================================
# Dataset Hierarchy and ID Keys
# ============================================================================

# Individual ID key names (used in stats dictionaries and samples)
ID_USER = "userId"
ID_SCREEN = "screenId"
ID_CAMERA = "cameraId"
ID_MONITOR = "monitorId"
ID_PLACE = "placeId"

# Hierarchy levels in order (top to bottom)
HIERARCHY_LEVELS = [ID_USER, ID_SCREEN, ID_CAMERA, ID_MONITOR, ID_PLACE]

# ID fields that should have single value per dataset (validation)
SINGLE_VALUE_FIELDS = [ID_USER, ID_SCREEN, ID_CAMERA, ID_MONITOR]

# All stats keys including metadata
STATS_KEYS = [ID_USER, ID_SCREEN, ID_CAMERA, ID_MONITOR, ID_PLACE, "blacklist"]

# Index field names (used in DatasetInfo namedtuple)
IDX_USER = "user_idx"
IDX_SCREEN = "screen_idx"
IDX_CAMERA = "camera_idx"
IDX_MONITOR = "monitor_idx"
IDX_PLACE = "place_idx"

# Mapping from ID key to index field name
ID_TO_IDX = {
    ID_USER: IDX_USER,
    ID_SCREEN: IDX_SCREEN,
    ID_CAMERA: IDX_CAMERA,
    ID_MONITOR: IDX_MONITOR,
    ID_PLACE: IDX_PLACE,
}

# MediaPipe FaceMesh settings
FACEMESH_VISIBILITY_THRESHOLD = 0.5  # minimum landmark visibility score
FACEMESH_PRESENCE_THRESHOLD = 0.5  # minimum landmark presence score
FACEMESH_DETECTION_CONFIDENCE = 0.5  # minimum detection confidence
FACEMESH_TRACKING_CONFIDENCE = 0.5  # minimum tracking confidence
FACEMESH_MAX_FACES = 1  # maximum number of faces to track
FACEMESH_REFINE_LANDMARKS = True  # whether to refine landmarks

# Camera settings
CAMERA_WIDTH = 1024  # default camera capture width
CAMERA_HEIGHT = 768  # default camera capture height
CAMERA_AUTO_EXPOSURE = -5  # camera auto-exposure setting (lower = darker)
CAMERA_AUTOFOCUS = 0  # camera autofocus mode (0 = disabled)
CAMERA_AUTO_WB = 0  # camera auto white balance (0 = disabled)

# Eye tracker settings
EYE_TRACKER_SMOOTHING_FACTOR = 0.8  # exponential smoothing for eye position
EYE_TRACKER_BLINK_THRESHOLD = 0.3  # eye aspect ratio threshold for blink detection
EYE_TRACKER_MIN_LANDMARK_COUNT = 5  # minimum eye landmarks required for visibility
EYE_TRACKER_MIN_ROI_RADIUS = 5  # minimum radius in pixels for eye region of interest
EYE_TRACKER_ROI_PADDING = 1.5  # padding multiplier for expanding eye region of interest
EYE_TRACKER_IMAGE_SIZE = (32, 32)  # final eye image size for model input
EYE_TRACKER_INTERMEDIATE_SIZE = (48, 48)  # intermediate resize size before center crop

# Data sampling settings
DATA_SAMPLER_MAX_RETRIES = 10  # maximum retry attempts for sampling

# Model architecture constants
FACEMESH_LANDMARK_COUNT = 478  # number of MediaPipe FaceMesh landmarks
EYE_REGION_SIZE = 32  # height and width of cropped eye region images
EYE_REGION_MIN_SIZE = 8  # minimum valid eye region size for validation
PREDICTOR_SHIFT = 0.5  # shift parameter for PredictorBlock


# Loss function settings
PSEUDO_HUBER_LOSS_DELTA = 0.01  # delta parameter for pseudo-Huber loss smoothing

# Training modes
TRAINING_MODE_FULL = "full"  # Two-stage pipeline (Face2Step + Step2Latent)
TRAINING_MODE_ENCODER = "encoder"  # Single-stage pipeline (Face2Step only)
VALID_TRAINING_MODES = (TRAINING_MODE_FULL, TRAINING_MODE_ENCODER)
