"""Animated light source with smooth cubic spline trajectory.

Creates a light source that moves along a randomly generated path
using cubic spline interpolation, with periodic path regeneration.
"""

from typing import Any, Callable, Optional

import numpy as np
import pygame
import scipy.interpolate as sInterp
from App.Constants import (
    ILLUMINATION_SPLINE_OVERLAP_POINTS,
    ILLUMINATION_SPEED_MIN,
    ILLUMINATION_SPEED_MAX,
    ILLUMINATION_DURATION_MIN,
    ILLUMINATION_DURATION_MAX,
    ILLUMINATION_RADIUS,
    ILLUMINATION_SPLINE_CONTROL_POINTS,
    ILLUMINATION_POSITION_CENTER,
    ILLUMINATION_POSITION_SCALE,
    ILLUMINATION_CLIP_MIN,
    ILLUMINATION_CLIP_MAX,
    ILLUMINATION_RENDER_RADIUS_SCALE,
)


class IlluminationSource:
    """Animated light source with smooth cubic spline trajectory.

    Creates a light source that moves along a randomly generated path
    using cubic spline interpolation, with periodic path regeneration.

    Attributes:
        _radius: Rendering radius for the light source
        _color: RGB color of the light source
        _points: Path control points for spline interpolation
        _T: Current time within the current spline
        _maxT: Duration of the current spline
        _pos: Current position (0.0-1.0 normalized coordinates)
        _getPoint: Callable that evaluates spline at time t
    """

    def __init__(self) -> None:
        """Initialize illumination source with random color and trajectory."""
        self._radius: int = ILLUMINATION_RADIUS
        self._color: np.ndarray = np.random.random((3,))
        self._points: Optional[np.ndarray] = None
        self._T: float = 0.0
        self._maxT: float = 0.0
        self._pos: np.ndarray = np.array(
            [ILLUMINATION_POSITION_CENTER, ILLUMINATION_POSITION_CENTER]
        )
        self._getPoint: Callable[[float], np.ndarray] = lambda t: np.array(
            [ILLUMINATION_POSITION_CENTER, ILLUMINATION_POSITION_CENTER]
        )
        self._new_spline(extend=False)

    def _new_spline(self, extend: bool = True) -> None:
        """Generate new cubic spline trajectory for the light source.

        Args:
            extend: If True, continue from end of previous spline. If False, start fresh.

        Raises:
            ValueError: If extend=True but no existing points available.
        """
        self._T = 0.0
        overlap_points = ILLUMINATION_SPLINE_OVERLAP_POINTS
        points = np.random.normal(
            size=(ILLUMINATION_SPLINE_CONTROL_POINTS, 2),
            loc=ILLUMINATION_POSITION_CENTER,
            scale=ILLUMINATION_POSITION_SCALE,
        )
        if extend:
            if self._points is None:
                raise ValueError(
                    f"Cannot extend without existing points, got: {self._points}"
                )
            points = np.concatenate([self._points[-overlap_points:], points], axis=0)

        self._points = points = np.clip(
            points, ILLUMINATION_CLIP_MIN, ILLUMINATION_CLIP_MAX
        )
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=-1)))
        distance = np.insert(distance, 0, 0)

        speed = np.random.uniform(
            ILLUMINATION_SPEED_MIN, ILLUMINATION_SPEED_MAX, size=1
        )[0]
        duration = distance[-1] / speed
        self._maxT = np.clip(
            duration, ILLUMINATION_DURATION_MIN, ILLUMINATION_DURATION_MAX
        )
        distance /= distance[-1]

        shift = distance[overlap_points - 1] if extend else 0.0
        splines = [sInterp.CubicSpline(distance, coords) for coords in points.T]
        self._getPoint = lambda t: np.array(
            [s((t * (1 - shift)) + shift) for s in splines]
        )

    def on_tick(self, delta_t: float) -> None:
        """Update light source position.

        Args:
            delta_t: Time delta since last frame in seconds.
        """
        self._T += delta_t
        if self._maxT < self._T:
            self._new_spline(extend=True)

        pos = self._getPoint(self._T / self._maxT)
        self._pos = np.clip(pos, a_min=0.0, a_max=1.0)

    def on_render(self, window: Any) -> None:
        """Render the light source as a circle.

        Args:
            window: Pygame surface to render to.
        """
        wh = np.array(window.get_size())
        pos = tuple((self._pos * wh).astype(np.int32))
        color = tuple((self._color * 255).astype(np.int32))
        radius = int((ILLUMINATION_RENDER_RADIUS_SCALE * wh).astype(np.int32).min())
        pygame.draw.circle(window, color, pos, radius)
