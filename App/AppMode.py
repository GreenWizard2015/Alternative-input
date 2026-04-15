"""Base class for application modes in the eye-tracking UI.

Provides event handling, rendering, and sampling functionality for different
application modes (game mode, data collection, visualization, etc.).
"""

from typing import Any, Dict
import pygame as G
import numpy as np


class AppMode:
    """Base application mode for handling user interactions and rendering.

    Abstract base class for different application modes that handle events,
    sampling, prediction display, and rendering. Includes pause/resume functionality
    for data collection.

    Attributes:
        _app: Reference to parent application instance
        _paused: Whether the current mode is paused
        _pos: Current position (varies by mode)
    """

    def __init__(self, app: Any) -> None:
        """Initialize application mode.

        Sets up mode state including pause flag and initial position.

        Args:
            app: Parent application instance with dataset, drawing utilities, etc.

        Raises:
            ValueError: If app is None.

        Example:
            >>> mode = AppMode(application)
            >>> assert mode.paused is True  # Starts paused
        """
        if app is None:
            raise ValueError(f"app parameter cannot be None, got: {app}")
        self._app: Any = app
        self._paused: bool = True
        self._pos: np.ndarray = np.zeros(
            shape=(2,)
        )  # Default position, subclasses override

    def on_event(self, event: G.event.Event) -> None:
        """Handle pygame events (keyboard input, mouse, etc.).

        Responds to key press events (P, Enter for pause/resume, Space to pause).

        Args:
            event: Pygame event object containing type and key information.

        Example:
            >>> mode = AppMode(app)
            >>> for event in pygame.event.get():
            ...     mode.on_event(event)
        """
        if event.type == G.KEYDOWN:
            if event.key in [G.K_p, G.K_RETURN]:
                self._paused = not self._paused

            if event.key == G.K_SPACE:
                self._paused = True

    def on_render(self, window: G.Surface) -> None:
        """Render mode-specific visualization.

        Base implementation does nothing. Subclasses override to draw mode-specific
        content (game objects, UI elements, etc.).

        Args:
            window: Pygame surface to render to. Typically the main display surface.

        Example:
            >>> mode = AppMode(app)
            >>> mode.on_render(screen)  # Draws on screen surface
        """

    def on_sample(self, tracked: Dict[str, Any]) -> None:
        """Handle incoming tracked data sample.

        Stores the tracked data with current position if mode is not paused.

        Args:
            tracked: Dictionary containing tracked facial data (face landmarks,
                eye regions, pose info, etc.).

        Example:
            >>> mode = AppMode(app)
            >>> tracked = {"face": ..., "eyes": ..., "pose": ...}
            >>> mode.on_sample(tracked)  # Stores if not paused
        """
        if self._paused:
            return
        self._app._dataset.store(data=tracked, position=np.array(self._pos))

    def on_prediction(self, pos: np.ndarray, data: Any) -> None:
        """Handle model prediction output.

        Base implementation does nothing. Subclasses override to handle
        predicted gaze position and confidence data.

        Args:
            pos: Predicted position coordinates, shape (2,) or (batch, 2).
            data: Additional prediction data (confidence scores, debug info, etc.).

        Example:
            >>> mode = AppMode(app)
            >>> pos = np.array([640, 360])  # Predicted gaze position
            >>> mode.on_prediction(pos, confidence=0.95)
        """

    def on_tick(self, delta_t: float) -> None:
        """Update mode state for a single frame.

        Called once per frame to update any time-dependent state. Base implementation
        does nothing. Subclasses override for mode-specific updates (movement, animation, etc.).

        Args:
            delta_t: Time delta since last frame in seconds.

        Example:
            >>> mode = AppMode(app)
            >>> mode.on_tick(0.016)  # ~60 FPS, 16ms per frame
        """

    @property
    def paused(self) -> bool:
        """Get current pause state.

        Returns:
            True if mode is paused, False otherwise.
        """
        return self._paused
