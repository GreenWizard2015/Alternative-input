"""Constants for preprocessing scripts.

This module centralizes all magic numbers and configuration constants used
across preprocessing scripts (preprocess-remote.py and related utilities).

All values here can be overridden via command-line arguments.

Example:
    >>> from scripts.Constants import MAX_DELTA_THRESHOLD
    >>> if min_delta > MAX_DELTA_THRESHOLD:
    ...     print("Dataset is too sparse")
"""

# Dataset validation thresholds
MAX_DELTA_THRESHOLD = (
    0.3  # Maximum allowed minimum time delta between consecutive frames (seconds)
)
DEFAULT_MIN_FRAMES = 5  # Minimum frames required for valid trajectory
DEFAULT_MAX_T = 1.0  # Maximum time window for trajectory validation (seconds)
DEFAULT_TEST_RATIO = 0.1  # Default fraction of samples to use for testing (10%)

# Frame filtering and selection
MINIMUM_TEST_SAMPLE_SPACING = (
    5  # Minimum frame spacing between test samples to avoid trajectory overlap
)

# NPZ file handling
DEFAULT_NPZ_COMPRESSION = "uncompressed"  # Compression level for saved NPZ files

# Logging and output
DEFAULT_LOG_LEVEL = "INFO"  # Default logging level

# Fields to validate (must have exactly one unique value per dataset)
SINGLE_VALUE_FIELDS = [
    "userId",  # User identifier - should be consistent across dataset
    "screenId",  # Screen identifier
    "cameraId",  # Camera identifier
    "monitorId",  # Monitor identifier
]

# Expected hierarchy levels for dataset folder structure
HIERARCHY_LEVELS = ["userId", "screenId", "cameraId", "monitorId", "placeId"]

# Statistics keys
STATS_KEYS = ["userId", "screenId", "cameraId", "monitorId", "placeId", "blacklist"]
