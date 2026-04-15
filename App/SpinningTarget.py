"""Spinning target visualization for game mode.

Renders an animated target that rotates and scales based on time,
with visual indicators for on-screen/off-screen positions.
"""

from typing import Any
import numpy as np
from App.Utils import Colors, rotate
from App.Constants import (
    SPINNING_TARGET_SATELLITE_COUNT,
    SPINNING_TARGET_TIME_SCALE,
    SPINNING_TARGET_RADIUS_AMPLITUDE,
    SPINNING_TARGET_ROTATION_SPEED,
    SPINNING_TARGET_INITIAL_ANGLE_MAX,
    SPINNING_TARGET_INITIAL_POSITION_CENTER,
)


class SpinningTarget:
    """Animated spinning target with visual indicators.

    Renders a target that spins around a given position with satellites.
    Includes off-screen edge indicators and radius animation (currently disabled).

    Attributes:
        _app: Reference to parent application
        _angle: Current rotation angle in radians
        _radius: Orbital radius of satellite objects
        _pos: Current normalized position (0.0-1.0)
        _T: Current time accumulator
        _TScale: Time scale for radius animation
    """

    def __init__(self, app: Any) -> None:
        """Initialize spinning target.

        Args:
            app: Parent application instance.

        Raises:
            ValueError: If app is None.
        """
        if app is None:
            raise ValueError(f"app parameter cannot be None, got: {app}")
        self._app: Any = app
        self._angle: float = np.random.uniform(
            low=0.0, high=SPINNING_TARGET_INITIAL_ANGLE_MAX
        )
        self._radius: float = 0.01
        self._pos: np.ndarray = (
            np.zeros(shape=(2,)) + SPINNING_TARGET_INITIAL_POSITION_CENTER
        )
        self._T: float = 0.0
        self._TScale: int = SPINNING_TARGET_TIME_SCALE

    def on_tick(self, delta_t: float, wheelPos: np.ndarray) -> np.ndarray:
        """Update target position and rotation.

        Args:
            delta_t: Time delta since last frame in seconds.
            wheelPos: Current normalized wheel position.

        Returns:
            Updated normalized position.
        """
        self._T += delta_t
        self._pos = wheelPos

        scaled_time = (self._T / self._TScale) % (2 * np.pi)
        self._radius = np.clip(
            np.cos(scaled_time) * SPINNING_TARGET_RADIUS_AMPLITUDE,
            a_min=0.0,
            a_max=None,
        )
        self._radius = 0.0  # disable radius animation
        self._angle = (self._angle + SPINNING_TARGET_ROTATION_SPEED) % (2 * np.pi)

        wh = self._app.WH
        mainPos = np.multiply(wh, self._pos)
        vec = np.multiply(wh, (self._radius, 0.0))
        return np.divide(mainPos + rotate(vec, self._angle), wh)

    def on_render(self) -> None:
        """Render the spinning target and satellites to application window.

        Renders the main target and satellite objects at their current positions,
        with an edge indicator if target is partially off-screen.
        """
        wh = self._app.WH
        mainPos = np.multiply(wh, self._pos)
        vec = np.multiply(wh, (self._radius, 0.0))
        satellite_count = SPINNING_TARGET_SATELLITE_COUNT
        angles = np.linspace(0.0, 2 * np.pi, num=satellite_count, endpoint=False)
        for i, angle in enumerate(angles[1:]):
            pos = mainPos + rotate(vec, self._angle + angle)
            self._app.drawObject(
                tuple(int(x) for x in pos),
                color=Colors.PURPLE,
                R=satellite_count + 3 - i,
            )

        pos = mainPos + rotate(vec, self._angle)
        self._app.drawTarget(tuple(int(x) for x in pos), R=satellite_count + 4)

        clipped_pos = np.clip(pos, 0, wh)
        if not np.allclose(clipped_pos, pos):
            self._app.drawObject(
                tuple(int(x) for x in clipped_pos), color=Colors.PURPLE, R=3
            )
