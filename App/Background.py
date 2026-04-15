"""Dynamic background rendering with optional animated brightness.

Provides background fill with optional time-based color cycling and
smooth brightness modulation.
"""

from typing import Any, Tuple
import pygame.locals as G
import numpy as np
import time
from App.Utils import Colors
from App.Constants import (
    BACKGROUND_BRIGHTNESS_CYCLE_DURATION,
    BACKGROUND_BRIGHTNESS_AMPLITUDE,
    BACKGROUND_COLOR_CYCLE_INTERVAL,
)


class Background:
    """Animated background with dynamic color and brightness.

    Renders a background with optional dynamic color cycling and smooth
    brightness changes over time. Can be toggled between static and
    dynamic modes with the 'B' key.

    Attributes:
        _backgroundDynamic: Whether dynamic color cycling is enabled
    """

    def __init__(self) -> None:
        """Initialize background renderer."""
        self._backgroundDynamic: bool = False

    def on_tick(self, delta_t: float) -> None:
        """Update background state (currently no-op).

        Args:
            delta_t: Time delta since last frame in seconds.
        """

    def _brightness(self) -> float:
        """Calculate time-based brightness modulation.

        Produces a smooth sinusoidal brightness variation over BACKGROUND_BRIGHTNESS_CYCLE_DURATION seconds.

        Returns:
            Brightness multiplier (0.5 to 1.5).
        """
        current_time = time.time()
        # smooth brightness change over cycle duration
        sin_value = np.sin(
            2.0 * np.pi * current_time / BACKGROUND_BRIGHTNESS_CYCLE_DURATION
        )
        brightness_multiplier = 1.0 + BACKGROUND_BRIGHTNESS_AMPLITUDE * sin_value
        return brightness_multiplier

    def on_render(self, window: Any) -> None:
        """Render background fill to pygame surface.

        Args:
            window: Pygame surface to render to.
        """
        bg_color: Tuple[int, int, int] = Colors.SILVER
        if self._backgroundDynamic:
            # take color from Colors.asList based on current time, change every 5 seconds
            bg_color = Colors.asList[
                int(time.time() / BACKGROUND_COLOR_CYCLE_INTERVAL) % len(Colors.asList)
            ]
        # apply brightness
        bg: np.ndarray = (
            np.multiply(bg_color, self._brightness()).clip(0, 255).astype(np.uint8)
        )
        window.fill(color=bg)

    def on_event(self, event: Any) -> None:
        """Handle keyboard events for background control.

        Responds to the 'B' key press to toggle dynamic background color cycling.

        Args:
            event: Pygame event object.
        """
        if event.type != G.KEYDOWN:
            return
        # toggle background dynamic (B)
        if G.K_b == event.key:
            self._backgroundDynamic = not self._backgroundDynamic
