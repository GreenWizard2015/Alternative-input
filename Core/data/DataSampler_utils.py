"""Data augmentation and tensor conversion - backward compatibility re-exports.

This module re-exports functions from refactored modules for backward compatibility.
New code should import from specific modules:
- Core.data.gaussian_utils: Gaussian distribution and light blob functions
- Core.data.augmentation: Image and point augmentation functions
- Core.data.tensor_conversion: Main toTensor function and constants
"""

# Re-export from gaussian_utils
from Core.data.gaussian_utils import gaussian, get_gaussian, addLightBlob

# Re-export from augmentation
from Core.data.augmentation import (
    apply_brightness_augmentation,
    apply_additive_noise,
    apply_dropout,
    apply_points_noise,
    apply_points_dropout,
    BRIGHTNESS_TRUNCATED_NORMAL_STDDEV,
)

# Re-export from tensor_conversion
from Core.data.tensor_conversion import (
    toTensor,
    EYE_IMAGE_SIZE,
    EYE_CROP_SIZE,
    EYE_CROP_FRACTION,
    LIGHT_BLOB_POSITION_MIN,
)


__all__ = [
    "gaussian",
    "get_gaussian",
    "addLightBlob",
    "toTensor",
    "EYE_IMAGE_SIZE",
    "EYE_CROP_SIZE",
    "EYE_CROP_FRACTION",
    "LIGHT_BLOB_POSITION_MIN",
    "BRIGHTNESS_TRUNCATED_NORMAL_STDDEV",
    "apply_brightness_augmentation",
    "apply_additive_noise",
    "apply_dropout",
    "apply_points_noise",
    "apply_points_dropout",
]
