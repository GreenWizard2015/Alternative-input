"""Core utility functions - backward compatibility re-exports.

This module re-exports functions from refactored modules for backward compatibility.
New code should import from specific modules:
- Core.landmarks: Face mesh landmark data and decoding
- Core.processing: Data conversion utilities
- Core.dataset_loading: Dataset loading and session extraction
"""

# All imports at the top
from typing import Any
import tensorflow as tf
from Core.landmarks import (
    FACE_PARTS_CONNECTIONS,
    COLORS,
    INDEX_TO_PART,
    PART_TO_INDICES,
    FACE_MESH_INVALID_VALUE,
    FACE_MESH_POINTS,
    decode_landmarks,
)
from Core.processing import tracked2sample, samples2inputs
from Core.dataset_loading import (
    DatasetInfo,
    data_from_folder,
    dataset_from,
    extract_sessions,
    count_samples_in,
    read_json,
    dataset_from_stats,
)
from Core.utils.DatasetPath import DatasetPath


def to_numpy(obj: Any) -> Any:
    """Convert tensors to numpy arrays recursively.

    Handles lists, tuples, dicts, or any object with a .numpy() method.
    Recursively processes nested structures.

    Args:
        obj: Input object (tensor, list, tuple, dict, or any type)

    Returns:
        Converted object with all tensors converted to numpy arrays
    """
    if isinstance(obj, list):
        return [to_numpy(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(to_numpy(item) for item in obj)
    if isinstance(obj, dict):
        return {key: to_numpy(val) for key, val in obj.items()}
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return obj


def to_tensor(obj: Any) -> Any:
    """Convert numpy arrays to tensors recursively.

    Handles lists, tuples, dicts, or any array-like object.
    Recursively processes nested structures.

    Skips objects that are already tensors for efficiency.

    Args:
        obj: Input object (numpy array, list, tuple, dict, or any type)

    Returns:
        Converted object with all numpy arrays converted to tensors
    """
    # Skip if already a tensor
    if isinstance(obj, tf.Tensor):
        return obj
    if isinstance(obj, list):
        return [to_tensor(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(to_tensor(item) for item in obj)
    if isinstance(obj, dict):
        return {key: to_tensor(val) for key, val in obj.items()}
    return tf.convert_to_tensor(obj)


__all__ = [
    "FACE_PARTS_CONNECTIONS",
    "COLORS",
    "INDEX_TO_PART",
    "PART_TO_INDICES",
    "FACE_MESH_INVALID_VALUE",
    "FACE_MESH_POINTS",
    "decode_landmarks",
    "tracked2sample",
    "samples2inputs",
    "DatasetInfo",
    "data_from_folder",
    "dataset_from",
    "extract_sessions",
    "count_samples_in",
    "read_json",
    "dataset_from_stats",
    "DatasetPath",
    "to_numpy",
    "to_tensor",
]
