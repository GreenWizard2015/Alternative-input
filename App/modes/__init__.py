"""Application modes for eye-tracking data collection and testing.

Provides multiple interactive modes including target following, circular patterns,
spline paths, corner positioning, and game-based accuracy assessment.
"""

from App.modes.MoveToGoal import MoveToGoal
from App.modes.CircleMovingMode import CircleMovingMode
from App.modes.LookAtMode import LookAtMode
from App.modes.SplineMode import SplineMode
from App.modes.CornerMode import CornerMode
from App.GameMode import GameMode

__all__ = [
    "MoveToGoal",
    "CircleMovingMode",
    "LookAtMode",
    "SplineMode",
    "CornerMode",
    "GameMode",
]

APP_MODES = [
    LookAtMode,
    CornerMode,
    SplineMode,
    CircleMovingMode,
    GameMode,
]
