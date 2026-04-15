"""Spline path mode with smooth target movement for eye-tracking."""

from typing import Any, Optional, Callable, List
import numpy as np
import scipy.interpolate as sInterp
from App.AppMode import AppMode
from App.SpinningTarget import SpinningTarget
from App.Constants import (
    SPLINE_MODE_OVERLAP_POINTS,
    SPLINE_SCALE_MIN,
    SPLINE_SCALE_MAX,
    SPLINE_POINT_OFFSET,
    SPLINE_NORMALIZATION_EPSILON,
    SPLINE_POINT_CLIPPING_MIN,
    SPLINE_POINT_CLIPPING_MAX,
    SPLINE_SPEED_MIN,
    SPLINE_SPEED_MAX,
    SPLINE_DURATION_MIN_MULTIPLIER,
    SPLINE_DURATION_MAX_MULTIPLIER,
    SPLINE_POSITION_CLIPPING_MIN,
    SPLINE_POSITION_CLIPPING_MAX,
)


class SplineMode(AppMode):
    """Application mode with smooth spline path following.

    Generates random cubic spline paths and animates a spinning target along them.
    Paths have variable smoothness/complexity and duration. Target follows path
    smoothly with rotation animation for visual tracking feedback.

    Attributes:
        _pos: Current target position (0.0-1.0 normalized)
        _target: SpinningTarget instance for rendering
        _time_accumulator: Time accumulator for current spline (seconds)
        _max_time: Duration for current spline segment (seconds)
        _scale: Scale factor for random path generation
        _points: Control points for current spline segment
        _evaluate_point: Method for evaluating spline position at time t
    """

    def __init__(self, app: Any) -> None:
        """Initialize spline mode.

        Args:
            app: Parent application instance.

        Raises:
            ValueError: If app is None.
        """
        if app is None:
            raise ValueError(f"app parameter cannot be None, got: {app}")
        super().__init__(app=app)
        self._pos = np.zeros((2,)) + 0.5
        self._target = SpinningTarget(app=app)
        self._time_accumulator = 0.0
        self._max_time = 1.0
        self._scale = 1.0
        self._points: Optional[np.ndarray] = None
        self._splines: List[Any] = []
        self._shift: float = 0.0
        self._evaluate_point: Optional[Callable] = None
        self._newSpline(extend=False)

    def _evaluate_spline(self, t: float) -> np.ndarray:
        """Evaluate spline position at normalized time.

        Args:
            t: Normalized time value (0.0 to 1.0).

        Returns:
            Position as numpy array with shape (2,).
        """
        shift = self._shift
        return np.array([s((t * (1 - shift)) + shift) for s in self._splines])

    def _updateScale(self) -> float:
        """Update scale factor for spline generation.

        Returns:
            New scale value.
        """
        newScale = np.random.uniform(SPLINE_SCALE_MIN, SPLINE_SCALE_MAX) + self._scale
        self._scale = newScale
        if 1.0 < self._scale:
            self._scale = 0.0
        return newScale

    def _newSpline(self, extend: bool = True) -> None:
        """Generate new cubic spline path.

        Args:
            extend: Whether to extend from previous spline segment.
        """
        self._time_accumulator = 0.0
        overlap_points = SPLINE_MODE_OVERLAP_POINTS
        scale = self._updateScale()
        points = np.random.uniform(size=(overlap_points + 1, 2)) - SPLINE_POINT_OFFSET
        points /= (
            np.linalg.norm(points, axis=-1, keepdims=True)
            + SPLINE_NORMALIZATION_EPSILON
        )
        points = SPLINE_POINT_OFFSET + (points * scale)
        if extend:
            if self._points is None:
                raise ValueError("points must be initialized when extending")
            points = np.concatenate([self._points[-overlap_points:], points], axis=0)

        self._points = np.clip(
            points, SPLINE_POINT_CLIPPING_MIN, SPLINE_POINT_CLIPPING_MAX
        )
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=-1)))
        distance = np.insert(distance, 0, 0)

        speed = np.random.uniform(SPLINE_SPEED_MIN, SPLINE_SPEED_MAX, size=1)[0]
        duration = distance[-1] / speed
        self._max_time = np.clip(
            duration,
            overlap_points * SPLINE_DURATION_MIN_MULTIPLIER,
            overlap_points * SPLINE_DURATION_MAX_MULTIPLIER,
        )
        distance /= distance[-1]

        shift = distance[overlap_points - 1] if extend else 0.0
        splines = [sInterp.CubicSpline(distance, coords) for coords in points.T]
        self._splines = splines
        self._shift = shift
        self._evaluate_point = self._evaluate_spline

    def on_tick(self, delta_t: float) -> None:
        """Update spline position along path.

        Args:
            delta_t: Time delta since last frame in seconds.

        Raises:
            ValueError: If _evaluate_point is not initialized.
        """
        self._time_accumulator += delta_t
        if self._max_time < self._time_accumulator:
            self._newSpline()

        if self._evaluate_point is None:
            raise ValueError("_evaluate_point must be initialized")
        pos = self._evaluate_point(self._time_accumulator / self._max_time)
        pos = np.clip(pos, SPLINE_POSITION_CLIPPING_MIN, SPLINE_POSITION_CLIPPING_MAX)
        self._pos = self._target.on_tick(delta_t, pos)

    def on_render(self, window: Any) -> None:
        """Render spinning target.

        Args:
            window: Pygame surface to render to.
        """
        super().on_render(window)
        self._target.on_render()
