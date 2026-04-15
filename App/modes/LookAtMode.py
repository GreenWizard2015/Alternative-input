"""Look-at target mode for eye-tracking calibration."""

from typing import Any, Optional
import numpy as np
import pygame.locals as G
import time
from App.AppMode import AppMode
from App.Utils import Colors
from App.Constants import LOOK_AT_MODE_VISIBLE_TIME


class LookAtMode(AppMode):
    """Application mode for look-at target with timed appearance.

    Displays a target that randomly relocates every N seconds. User must focus on
    the target for a fixed duration, with visual feedback indicating active state
    (red when collecting samples, white when inactive). Designed for calibration
    and accuracy testing.

    Attributes:
        _visible_time: Duration target remains visible before relocating (seconds)
        _active: Whether currently collecting samples
        _start_time: Timestamp when current inactive period started
    """

    def __init__(self, app: Any) -> None:
        """Initialize look-at mode.

        Args:
            app: Parent application instance.

        Raises:
            ValueError: If app is None.
        """
        if app is None:
            raise ValueError(f"app parameter cannot be None, got: {app}")
        super().__init__(app=app)
        self._visible_time = LOOK_AT_MODE_VISIBLE_TIME
        self._start_time: Optional[float] = None
        self._active: bool = False
        self._pos: np.ndarray = np.zeros((2,))
        self._next()

    def _next(self) -> None:
        """Generate next target position.

        Creates a random position in the normalized coordinate space [0, 1],
        resets active state, and clears the start time for the next collection period.
        """
        self._pos = np.random.uniform(size=(2,))
        self._active = False
        self._start_time = None

    def on_event(self, event: Any) -> None:
        """Handle keyboard events.

        Args:
            event: Pygame event object.
        """
        super().on_event(event=event)
        if event.type == G.KEYDOWN:
            if G.K_RIGHT == event.key:
                self._active = True

    def on_tick(self, delta_t: float) -> None:
        """Update mode state.

        Args:
            delta_t: Time delta since last frame in seconds.

        Raises:
            ValueError: If start_time is not set when mode is active.
        """
        if not self._active:
            self._start_time = time.time()
            return

        if self._start_time is None:
            raise ValueError("start_time must be set when mode is active")
        elapsed_time: float = time.time() - self._start_time
        if self._visible_time < elapsed_time:
            self._next()

    def on_sample(self, tracked: Any) -> None:
        """Store sample when active.

        Args:
            tracked: Tracked facial data dictionary.
        """
        if self._active:
            super().on_sample(tracked)

    def on_render(self, window: Any) -> None:
        """Render target position.

        Args:
            window: Pygame surface to render to.
        """
        super().on_render(window)
        wh = np.array(window.get_size())
        pos = tuple(int(x) for x in np.multiply(wh, self._pos))

        if self._active:
            self._app.drawObject(pos, color=Colors.RED)
        else:
            self._app.drawObject(pos, color=Colors.WHITE)
