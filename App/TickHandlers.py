#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tick/frame update handling for the application.

Provides helper functions for updating prediction smoothing,
managing prediction history, and transforming tracked data.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from Core.Utils import FACE_MESH_INVALID_VALUE


def transform_tracked_data(
    tracked: Optional[Dict[str, Any]],
    mask_face: bool,
    mask_left_eye: bool,
    mask_right_eye: bool,
) -> Optional[Dict[str, Any]]:
    """Transform tracked facial data by applying predictor masks.

    Args:
        tracked: Dictionary with 'tracked' key containing facial data, or None.
        mask_face: Whether to mask face data.
        mask_left_eye: Whether to mask left eye data.
        mask_right_eye: Whether to mask right eye data.

    Returns:
        Modified tracked data dictionary or None if input is None.
    """
    if tracked is None:
        return None

    tracked_data = tracked.get("tracked")
    if tracked_data is None:
        return None

    res = tracked_data.copy()

    if mask_face:
        res["face points"] = np.full_like(
            tracked_data["face points"], FACE_MESH_INVALID_VALUE
        )

    if mask_left_eye:
        res["left eye"] = np.full_like(tracked_data["left eye"], 0.0)

    if mask_right_eye:
        res["right eye"] = np.full_like(tracked_data["right eye"], 0.0)

    return res


def update_prediction_smoothing(
    current_smoothed: np.ndarray,
    prediction_pos: np.ndarray,
    smoothing_factor: float,
) -> np.ndarray:
    """Apply exponential smoothing to prediction position.

    Args:
        current_smoothed: Current smoothed prediction (normalized [0,1]²).
        prediction_pos: New prediction position (normalized [0,1]²).
        smoothing_factor: Exponential smoothing factor [0-1].

    Returns:
        Updated smoothed prediction, clipped to [0,1]².
    """
    return np.clip(
        np.multiply(current_smoothed, smoothing_factor)
        + np.multiply(prediction_pos, 1.0 - smoothing_factor),
        0.0,
        1.0,
    )


def update_prediction_history(
    history: List[np.ndarray],
    prediction_pos: np.ndarray,
    max_length: int = 15,
) -> List[np.ndarray]:
    """Update prediction history with new position.

    Args:
        history: List of historical prediction positions.
        prediction_pos: New prediction position to add.
        max_length: Maximum number of entries to keep (default: 15).

    Returns:
        Updated history list (truncated if necessary).
    """
    history.append(prediction_pos)
    return history[-max_length:]
