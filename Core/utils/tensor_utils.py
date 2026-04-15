"""Tensor utility functions for data processing."""

from typing import Dict

import tensorflow as tf
from Core.landmarks import FACE_MESH_INVALID_VALUE


def only_valid_points(points: tf.Tensor) -> tf.Tensor:
    """Replace points outside [0, 1] range with FACE_MESH_INVALID_VALUE.

    Args:
        points: Tensor of shape (..., 2) containing point coordinates.

    Returns:
        Points tensor with out-of-range values replaced by FACE_MESH_INVALID_VALUE.
    """
    is_valid = tf.reduce_all(
        tf.logical_and(points >= 0.0, points <= 1.0), axis=-1, keepdims=True
    )
    return tf.where(is_valid, points, FACE_MESH_INVALID_VALUE)


def validate_data_dict(data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Validate points in a data dictionary, replacing out-of-range values.

    Applies only_valid_points to 'points' key if present in the dictionary.

    Args:
        data: Dictionary containing tensor data, may include 'points' key.

    Returns:
        Dictionary with validated points (if present) or unchanged if no 'points' key.
    """
    if "points" not in data:
        return data

    validated = {**data}
    validated["points"] = only_valid_points(validated["points"])
    return validated
