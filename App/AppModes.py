"""Application modes for eye-tracking data collection and testing.

This module provides backward compatibility imports. New code should import
directly from App.modes.

Provides multiple interactive modes including target following, circular patterns,
spline paths, corner positioning, and game-based accuracy assessment.
"""

from App.modes import (
    MoveToGoal,
    CircleMovingMode,
    LookAtMode,
    SplineMode,
    CornerMode,
    GameMode,
    APP_MODES,
)

__all__ = [
    "MoveToGoal",
    "CircleMovingMode",
    "LookAtMode",
    "SplineMode",
    "CornerMode",
    "GameMode",
    "APP_MODES",
]
