"""Move-to-goal application mode for eye-tracking."""

from typing import Any
import numpy as np
from App.AppMode import AppMode
from App.Utils import normalized
from App.Constants import (
    MOVE_TO_GOAL_BASE_SPEED,
    MOVE_TO_GOAL_SPEED_MULTIPLIER,
    MOVE_TO_GOAL_DISTANCE_THRESHOLD,
)


class MoveToGoal(AppMode):
    """Mode where cursor smoothly moves towards a goal position.

    Implements a smooth movement algorithm that moves the cursor from current
    position toward a goal position at constant speed.

    Attributes:
        _speed: Movement speed in pixels per second
        _pos: Current cursor position (0.0-1.0 normalized)
        _goal: Goal position to move towards
    """

    def __init__(self, app: Any) -> None:
        """Initialize move-to-goal mode.

        Args:
            app: Parent application instance.

        Raises:
            ValueError: If app is None.
        """
        if app is None:
            raise ValueError(f"app parameter cannot be None, got: {app}")
        super().__init__(app=app)
        self._speed: float = MOVE_TO_GOAL_BASE_SPEED * MOVE_TO_GOAL_SPEED_MULTIPLIER
        self._pos: np.ndarray = np.zeros((2,)) + 0.5
        self._goal: np.ndarray = np.zeros((2,)) + 0.5

    def on_tick(self, delta_t: float) -> None:
        """Update cursor position towards goal.

        Calculates movement vector towards goal and updates position based on speed
        and elapsed time. When distance to goal drops below threshold, requests next goal.

        Args:
            delta_t: Time delta since last frame in seconds.
        """
        wh = self._app.WH
        pos = np.multiply(wh, self._pos)
        goal = np.multiply(wh, self._goal)

        vec = normalized(np.subtract(goal, pos))[0]
        self._pos = np.add(pos, vec * self._speed * delta_t) / wh

        dist = np.sqrt(np.square(np.subtract(pos, goal)).sum())
        if dist < MOVE_TO_GOAL_DISTANCE_THRESHOLD:
            self._goal = self._nextGoal(old=self._goal)

    def on_render(self, window: Any) -> None:
        """Render cursor position.

        Args:
            window: Pygame surface to render to.
        """
        super().on_render(window)

        wh = np.array(window.get_size())
        pos = tuple(int(x) for x in np.multiply(wh, self._pos))
        self._app.drawObject(pos)

    def _nextGoal(self, old: np.ndarray) -> np.ndarray:
        """Generate next goal position. Override in subclasses for custom behavior.

        Default implementation generates random position. Subclasses should override
        this method to implement custom goal generation logic (e.g., circular paths,
        spline following, corner positioning).

        Args:
            old: Previous goal position, shape (2,).

        Returns:
            New goal position with shape (2,), normalized to [0, 1].
        """
        return np.random.uniform(size=(2,))
