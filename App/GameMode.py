"""Game mode for eye-tracking accuracy assessment with hitting targets.

Implements a game where targets appear at random positions and the user
must focus their eyes on them. Tracks accuracy metrics like hit distance and speed.
"""

from typing import Any
import numpy as np
import pygame
import pygame.locals as G
from App.Utils import Colors
from App.AppMode import AppMode
from App.Constants import (
    GAME_MODE_RADIUS_DELTA_PER_SECOND,
    GAME_MODE_MAX_HITS,
    GAME_MODE_MAX_HITS_IN_RANGE,
    GAME_MODE_TARGET_SCALE,
    GAME_MODE_MAX_IN_RANGE_HITS,
    GAME_MODE_INITIAL_POSITION,
    GAME_MODE_POSITION_CLIP_MIN,
    GAME_MODE_POSITION_CLIP_MAX,
    GAME_MODE_MAX_PROB_POWER,
    GAME_MODE_ADJUST_POS_THRESHOLD,
    GAME_MODE_ADJUST_POS_POWER,
)


def _adjustPos(x: float, power: int = GAME_MODE_ADJUST_POS_POWER) -> float:
    """Adjust position with non-linear power scaling for edge concentration.

    Maps values to concentrate targets more often at edges (0.0 and 1.0).

    Args:
        x: Input position (0.0-1.0).
        power: Power exponent for non-linear scaling (default: GAME_MODE_ADJUST_POS_POWER).

    Returns:
        Adjusted position with edge concentration.
    """
    if x < GAME_MODE_ADJUST_POS_THRESHOLD:
        x = GAME_MODE_TARGET_SCALE * x  # scale to 0..1
        x = np.power(x, power)  # make it more often on edges
        return x / GAME_MODE_TARGET_SCALE  # scale back to 0..0.5

    return 1.0 - _adjustPos(1.0 - x, power=power)


class GameMode(AppMode):
    """Game mode for eye-tracking accuracy assessment.

    Presents targets at random positions with expanding accuracy circles.
    Tracks metrics like hit distance, time to hit, and observation accuracy.

    Attributes:
        _pos: Current target position (0.0-1.0)
        _T: Time accumulator for current target
        _currentRadius: Expanding circle radius for hitting
        _hits: Number of targets hit in current round
        _maxHits: Number of targets needed before difficulty increase
        _inRangeHits: Number of consecutive on-target predictions
        _inRangeMaxHits: Hits required to trigger target hit
        _probPower: Difficulty parameter for target position distribution
        _totalTime: Sum of time to hit for all targets
        _totalHits: Total targets hit
        _totalDistance: Sum of prediction distances at hit
        _totalObservations: Number of predictions made
        _totalObservationsDistance: Sum of prediction distances
    """

    def __init__(self, app: Any) -> None:
        """Initialize game mode.

        Args:
            app: Parent application instance.
        """
        super().__init__(app)
        self._pos: np.ndarray = np.zeros((2,)) + GAME_MODE_INITIAL_POSITION
        self._T: float = 0.0
        self._currentRadius: float = 0.0
        self._radiusPerSecond: float = GAME_MODE_RADIUS_DELTA_PER_SECOND
        self._hits: int = 0
        self._maxHits: int = GAME_MODE_MAX_HITS

        self._inRangeHits: int = 0
        self._inRangeMaxHits: int = GAME_MODE_MAX_HITS_IN_RANGE

        self._totalTime: float = 0.0
        self._totalHits: int = 0
        self._totalDistance: float = 0.0

        self._totalObservations: int = 0
        self._totalObservationsDistance: float = 0.0
        self._probPower: int = 1

    def on_sample(self, tracked: Any) -> None:
        """Store sample when target is hit.

        Args:
            tracked: Tracked facial data dictionary.
        """
        if not self._app.hasPredictions:
            return
        if self._paused:
            return
        if 0 < self._hits:
            self._app._dataset.store(data=tracked, position=np.array(self._pos))

    def on_tick(self, delta_t: float) -> None:
        """Update target radius and time accumulator.

        Args:
            delta_t: Time delta since last frame in seconds.
        """
        self._T += delta_t
        elapsed_time = self._T
        self._currentRadius = elapsed_time * self._radiusPerSecond

    def on_render(self, window: Any) -> None:
        """Render game target, circle, and statistics.

        Args:
            window: Pygame surface to render to.
        """
        if not self._app.hasPredictions:
            self._app.drawText(
                text="Game mode requires predictions to be provided by the model",
                pos=(window.get_width() // 2, window.get_height() // 3),
                color=Colors.RED,
                center=True,
                scale=4.0,
            )

        wh = np.array(window.get_size())
        pos = tuple(np.multiply(wh, self._pos).astype(np.int32))
        self._app.drawObject(
            pos=pos, color=Colors.WHITE
        )  # draw the target for focusing on
        # second circle
        radius = np.multiply(wh, self._currentRadius).min().astype(np.int32)
        clr = Colors.RED if self._hits == 0 else Colors.GREEN
        # Use fixed radius for hit detection circle (overrides calculated radius)
        radius = GAME_MODE_MAX_IN_RANGE_HITS
        pygame.draw.circle(
            surface=window, color=clr, center=pos, radius=int(radius), width=1
        )

        # score at the top center
        hits = self._totalHits
        if 0 < hits:
            self._app.drawText(
                text="Hits: %d, accuracy: %.4f, time: %.2f, obs. dist.: %.4f"
                % (
                    hits,
                    self._totalDistance / hits,
                    self._totalTime / hits,
                    (
                        self._totalObservationsDistance / self._totalObservations
                        if 0 < self._totalObservations
                        else 0.0
                    ),
                ),
                pos=(wh[0] // 2, 80),
                color=Colors.BLACK,
                center=True,
            )

        self._app.drawText(
            text="Power: %d, Hits: %d / %d, In range: %d / %d"
            % (
                self._probPower,
                self._hits,
                self._maxHits,
                self._inRangeHits,
                self._inRangeMaxHits,
            ),
            color=Colors.BLACK,
            pos=(wh[0] // 2, 80 + 30),
            scale=0.75,
            center=True,
        )

    def on_prediction(self, pos: Any, tracked: Any) -> None:
        """Handle gaze prediction and check for target hits.

        Args:
            pos: Predicted gaze position.
            tracked: Tracked facial data.

        Raises:
            ValueError: If predicted position shape does not match target shape.
        """
        if not self._app.hasPredictions:
            return
        pos = np.array(pos).reshape((2,))
        if pos.shape != self._pos.shape:
            raise ValueError(
                f"Position shape mismatch: predicted {pos.shape} vs target {self._pos.shape}"
            )
        # check if the prediction is inside the expanding circle
        distance = np.square(np.subtract(pos, self._pos)).sum()
        distance = np.sqrt(distance)
        # calculate the global accuracy
        if 0 < self._hits:
            self._totalObservations += 1
            self._totalObservationsDistance += distance

        if distance < self._currentRadius:
            self._inRangeHits += 1
            if self._inRangeMaxHits <= self._inRangeHits:
                self._inRangeHits = 0
                self._hit(distance)
        else:
            self._inRangeHits = 0

    def _hit(self, distance: float) -> None:
        """Process a successful target hit.

        Args:
            distance: Distance from target center at hit.
        """
        if 0 < self._hits:  # Only if active
            self._totalHits += 1
            self._totalDistance += distance
            self._totalTime += self._T

        self._T = 0.0
        self._currentRadius = 0.0

        self._hits += 1
        if self._maxHits <= self._hits:
            self._nextGoal()

    def _nextGoal(self) -> None:
        """Generate next target position based on difficulty."""
        pos = np.random.random((2,))
        if 1 < self._probPower:
            pos = np.array([_adjustPos(x, self._probPower) for x in pos])

        self._pos = np.clip(
            pos, GAME_MODE_POSITION_CLIP_MIN, GAME_MODE_POSITION_CLIP_MAX
        )
        self._hits = 0
        self._inRangeHits = 0

    def on_event(self, event: Any) -> None:
        """Handle keyboard events for difficulty adjustment.

        Args:
            event: Pygame event object.
        """
        if G.KEYDOWN == event.type:
            if G.K_UP == event.key:
                self._probPower += 1
            if G.K_DOWN == event.key:
                self._probPower -= 1
            self._probPower = int(np.clip(self._probPower, 1, GAME_MODE_MAX_PROB_POWER))

            # numpad plus and minus
            if G.K_KP_PLUS == event.key:
                self._inRangeMaxHits += 1
            if G.K_KP_MINUS == event.key:
                self._inRangeMaxHits -= 1
            self._inRangeMaxHits = int(
                np.clip(self._inRangeMaxHits, 1, GAME_MODE_MAX_IN_RANGE_HITS)
            )
        super().on_event(event)
