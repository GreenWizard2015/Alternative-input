"""Data processing utilities for converting between data formats.

Provides functions for converting tracked data to samples and samples to
batched input arrays with proper normalization.
"""

from typing import Dict, List, Any
import numpy as np


def tracked2sample(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tracked data to sample dictionary format.

    Extracts relevant fields from tracked data and returns in sample format
    used by data samplers.

    Args:
        data: Tracked data dictionary with keys:
            - 'time': Timestamp (float)
            - 'face points': Face mesh points (array)
            - 'left eye': Left eye image (array)
            - 'right eye': Right eye image (array)

    Returns:
        Sample dictionary with keys: time, points, left eye, right eye
    """
    return {
        "time": data["time"],
        "points": data["face points"],
        "left eye": data["left eye"],
        "right eye": data["right eye"],
    }


def samples2inputs(samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Convert list of samples to input arrays.

    Stacks sample data into batch arrays with proper normalization for images.

    Args:
        samples: List of sample dictionaries with keys: points, left eye, right eye, time

    Returns:
        Dictionary with batched arrays:
            - 'points': Shape (N, 478, 2) - face mesh points
            - 'left eye': Shape (N, H, W, 3) normalized to [0, 1]
            - 'right eye': Shape (N, H, W, 3) normalized to [0, 1]
            - 'time': Shape (N,) - timestamps
    """
    return {
        "points": np.array([x["points"] for x in samples], dtype=np.float32),
        "left eye": np.array([x["left eye"] for x in samples], dtype=np.float32)
        / 255.0,
        "right eye": np.array([x["right eye"] for x in samples], dtype=np.float32)
        / 255.0,
        "time": np.array([x["time"] for x in samples], dtype=np.float32),
    }
