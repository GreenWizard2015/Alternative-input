"""Circle moving mode with adjustable difficulty for eye-tracking."""

from typing import Any, List, Optional
import numpy as np
import pygame.locals as G
import time
from App.modes.MoveToGoal import MoveToGoal
from App.Constants import (
    CIRCLE_MODE_TRANSITION_TIME,
    CIRCLE_MODE_MAX_LEVEL,
    CIRCLE_MODE_PATH_OFFSET,
    CIRCLE_MODE_DIFFICULTY_DIVISOR,
)


class CircleMovingMode(MoveToGoal):
    """Mode with circular target path at adjustable difficulty levels.

    Targets move in a circle pattern with difficulty adjustable via level parameter.

    Attributes:
        _transition_time: Transition time between goals in seconds
        _max_level: Maximum difficulty level
        _level: Current difficulty level
        _path: Queue of remaining goal positions
        _transition_start: Time when transition started
        _active: Whether mode is currently active
    """

    def __init__(self, app: Any) -> None:
        """Initialize circle moving mode.

        Args:
            app: Parent application instance.

        Raises:
            ValueError: If app is None.
        """
        if app is None:
            raise ValueError(f"app parameter cannot be None, got: {app}")
        super().__init__(app=app)
        self._transition_time: float = CIRCLE_MODE_TRANSITION_TIME
        self._max_level: int = CIRCLE_MODE_MAX_LEVEL
        self._level: int = 0
        self._path: List[np.ndarray] = []
        self._transition_start: Optional[float] = None
        self._active: bool = False
        self._reset()

    def _nextGoal(self, old: np.ndarray) -> np.ndarray:
        """Get next goal in circular path.

        Args:
            old: Previous goal position, shape (2,).

        Returns:
            Next goal position with shape (2,).
        """
        if self._transition_start is None:
            self._transition_start = time.time()
        if (time.time() - self._transition_start) < self._transition_time:
            return old

        self._transition_start = None
        if len(self._path) <= 0:
            self._reset()
            return self._goal

        goal, *self._path = self._path
        return goal

    def on_event(self, event: Any) -> None:
        """Handle keyboard events for difficulty and direction.

        Args:
            event: Pygame event object.
        """
        super().on_event(event=event)
        if event.type != G.KEYDOWN:
            return

        if G.K_UP == event.key:
            self._level = min(self._max_level, self._level + 1)
            self._reset()
            return

        if G.K_DOWN == event.key:
            self._level = max(0, self._level - 1)
            self._reset()
            return

        if G.K_RIGHT == event.key:
            self._reset(clockwise=False)
            self._active = True
            return

        if G.K_LEFT == event.key:
            self._reset(clockwise=True)
            self._active = True

    def on_sample(self, tracked: Any) -> None:
        """Store sample when active.

        Args:
            tracked: Tracked facial data dictionary.
        """
        if self._active:
            super().on_sample(tracked=tracked)

    def on_tick(self, delta_t: float) -> None:
        """Update mode when active.

        Args:
            delta_t: Time delta since last frame in seconds.
        """
        if self._active:
            super().on_tick(delta_t)

    def _reset(self, clockwise: bool = False) -> None:
        """Reset circular path configuration.

        Reinitializes the circular movement path based on current difficulty level
        and direction. Path consists of four corners arranged in a square pattern,
        with difficulty controlling the size of the path.

        Args:
            clockwise: Whether to reverse the path to counter-clockwise direction.
                      Defaults to False (regular clockwise direction).
        """
        path = np.array(
            [
                [-1, 1],
                [1, 1],
                [1, -1],
                [-1, -1],
                [-1, 1],
            ],
            np.float32,
        )
        lvl = (self._max_level - self._level) / (
            CIRCLE_MODE_DIFFICULTY_DIVISOR * self._max_level
        )
        path = CIRCLE_MODE_PATH_OFFSET + lvl * path
        if clockwise:
            path = path[::-1]
        self._pos, self._goal, *self._path = path
        self._active = False
        self._transition_start = None
