"""Corner mode with target moving around screen corners for eye-tracking."""

from typing import Any
import numpy as np
import pygame.locals as G
from App.AppMode import AppMode
from App.SpinningTarget import SpinningTarget
from App.Constants import (
    CORNER_MODE_RADIUS,
    CORNER_MODE_FREQUENCY_MULTIPLIER,
    CORNER_MODE_CLIP_MIN,
    CORNER_MODE_CLIP_MAX,
)


class CornerMode(AppMode):
    """Application mode with target moving around screen corners.

    Displays a spinning target that moves in circular patterns around one of the
    four screen corners. User can select between corners using left/right arrow keys.
    Target oscillates with sinusoidal motion for smooth, predictable movement patterns.

    Attributes:
        _pos: Current target position (0.0-1.0 normalized)
        _target: SpinningTarget instance for rendering
        _time_accumulator: Time accumulator for circular motion (seconds)
        _radius: Radius of circular motion around corner (normalized 0-1)
        _corner_id: Index of currently selected corner (0-3)
        _corners: Array of 4 corner positions: (0,0), (0,1), (1,0), (1,1)
    """

    def __init__(self, app: Any) -> None:
        """Initialize corner mode.

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
        self._radius = CORNER_MODE_RADIUS
        self._corners = np.array(
            [
                [CORNER_MODE_CLIP_MIN, CORNER_MODE_CLIP_MIN],
                [CORNER_MODE_CLIP_MIN, CORNER_MODE_CLIP_MAX],
                [CORNER_MODE_CLIP_MAX, CORNER_MODE_CLIP_MIN],
                [CORNER_MODE_CLIP_MAX, CORNER_MODE_CLIP_MAX],
            ],
            np.float32,
        )
        self._corner_id = 0

    def on_tick(self, delta_t: float) -> None:
        """Update target position around corner.

        Args:
            delta_t: Time delta since last frame in seconds.
        """
        self._time_accumulator += delta_t
        radius = (
            np.abs(np.sin(self._time_accumulator * CORNER_MODE_FREQUENCY_MULTIPLIER))
            * self._radius
        )
        pos = (
            np.array([np.cos(self._time_accumulator), np.sin(self._time_accumulator)])
            * radius
        )
        pos = np.clip(
            self._corners[self._corner_id] + pos,
            CORNER_MODE_CLIP_MIN,
            CORNER_MODE_CLIP_MAX,
        )
        self._pos = self._target.on_tick(delta_t, pos)

    def on_render(self, window: Any) -> None:
        """Render spinning target.

        Args:
            window: Pygame surface to render to.
        """
        super().on_render(window)
        self._target.on_render()

    def on_event(self, event: Any) -> None:
        """Handle keyboard events for corner selection.

        Args:
            event: Pygame event object.
        """
        super().on_event(event=event)
        if event.type != G.KEYDOWN:
            return

        num_corners = len(self._corners)
        if G.K_LEFT == event.key:
            self._paused = True
            self._corner_id = (num_corners + self._corner_id - 1) % num_corners
            return

        if G.K_RIGHT == event.key:
            self._paused = True
            self._corner_id = (num_corners + self._corner_id + 1) % num_corners
