#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Eye-tracking based mouse control application (Windows-only).

Provides real-time mouse control using eye gaze predictions and mouth gestures.
Designed for testing and demonstration purposes on Windows systems.

Dependencies:
    - pywin32: Windows API access for mouse control
    - pywinauto: Windows automation framework
    - keyboard: Global keyboard event handling

Note:
    This application is highly resource-intensive (CPU/GPU intensive) and requires:
    1. A trained eye-tracking model
    2. Installation of Windows dependencies: pip install pywin32 pywinauto keyboard
    3. Running the pywin32 post-install script as described at
       https://github.com/mhammond/pywin32#installing-globally
"""
import argparse
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import keyboard
import numpy as np

# pywintypes must be imported before win32api due to a bug in pywin32
# This import is required for Windows API initialization even though it's not used directly
import pywintypes  # noqa: F401  # Required for Windows API initialization
import pywinauto
from pywinauto import win32defines, win32functions

import Core.Utils as Utils
from Core.models.ModelWrapper import ModelWrapper
from Core.tracking.LearnablePredictor import LearnablePredictor, PredictionResult
from Core.tracking.ThreadedEyeTracker import ThreadedEyeTracker
from Core.logging_config import get_logger

logger = get_logger(__name__)

# Mouse control timing
TIMINGS_CURSOR_WAIT = 0.0
TIMINGS_CLICK_WAIT = 0.0
QUIT_KEY = "esc"
DEFAULT_INITIALIZATION_TIMEOUT = 1.0
MODEL_DATA_FOLDER = "Data"


class App:
    """Mouse control application using eye gaze and mouth gestures.

    Performs real-time eye tracking and gaze prediction to control mouse position,
    using mouth open/close detection to trigger mouse clicks. Includes exponential
    smoothing for stable cursor movement and supports multiple tracking framerate.

    Attributes:
        _tracker: ThreadedEyeTracker instance for real-time facial tracking
        _predictor: LearnablePredictor instance for gaze prediction
        _smoothingFactor: Exponential smoothing factor for position averaging [0-1]
        _smoothedPrediction: Current smoothed gaze position (normalized [0,1]²)
        _mouthOpen: Current mouth open/close state
        _screen: Screen resolution tuple (width, height)
    """

    def __init__(
        self,
        tracker: ThreadedEyeTracker,
        predictor: LearnablePredictor,
        smoothingFactor: float,
        fps: int,
        lipsMinDistance: float,
    ) -> None:
        """Initialize mouse control application.

        Args:
            tracker: ThreadedEyeTracker instance for facial tracking.
            predictor: Callable for gaze prediction (e.g., LearnablePredictor.async_infer).
            smoothingFactor: Exponential smoothing factor for position [0-1].
            fps: Target frames per second for main loop.
            lipsMinDistance: Minimum lips distance (in pixels) to detect mouth open.
        """
        self._smoothingFactor = smoothingFactor
        self._fps = fps
        self._lipsMinDistance = lipsMinDistance

        self._lastPrediction: Optional[PredictionResult] = None
        self._smoothedPrediction: np.ndarray = np.array([0.0, 0.0])
        self._tracker = tracker
        self._predictor = predictor

        self._mouthOpen = False
        self._mouthWasOpen = False

        self._done = threading.Event()

    def on_keypress(self, event: Any) -> None:
        """Handle keyboard events.

        Args:
            event: Keyboard event object.
        """
        if event.name == QUIT_KEY:
            self._done.set()

    def on_tick(self) -> None:
        """Update application state for a single frame."""
        tracked = self._tracker.track()
        if tracked is not None:
            self._mouthOpen = self._lipsMinDistance <= tracked["lips distance"]
            current_time = time.time()
            tracked_data = {**tracked, "time": current_time}
            prediction = self._predictor(tracked_data)
            if prediction is not None:
                self._lastPrediction = prediction

        if self._lastPrediction:
            factor = self._smoothingFactor
            predPos = self._lastPrediction.prediction.result
            self._smoothedPrediction = np.clip(
                np.multiply(self._smoothedPrediction, factor)
                + np.multiply(predPos, 1.0 - factor),
                0.0,
                1.0,
            )

        pos_array = np.multiply(self._smoothedPrediction, self._screen)
        pos_tuple: Tuple[int, int] = tuple(pos_array.astype(np.int32))
        try:
            if self._mouthOpen and not self._mouthWasOpen:
                pywinauto.mouse.click(button="left", coords=pos_tuple)
            else:
                pywinauto.mouse.move(pos_tuple)
        except pywinauto.PyAutoError as e:
            logger.debug(f"Mouse control failed: {e}")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Invalid mouse coordinates: {e}")
        self._mouthWasOpen = self._mouthOpen

    def run(self) -> None:
        """Run main application loop with keyboard and tracker polling.

        Initializes mouse control system, registers keyboard handler, and continuously
        updates gaze position and mouse cursor until ESC key is pressed.
        """
        keyboard.on_press(self.on_keypress, suppress=False)
        try:
            pywinauto.timings.Timings.after_setcursorpos_wait = TIMINGS_CURSOR_WAIT
            pywinauto.timings.Timings.after_clickinput_wait = TIMINGS_CLICK_WAIT

            self._screen = np.array(
                [
                    win32functions.GetSystemMetrics(win32defines.SM_CXSCREEN),
                    win32functions.GetSystemMetrics(win32defines.SM_CYSCREEN),
                ]
            )
            while not self._done.wait(
                timeout=DEFAULT_INITIALIZATION_TIMEOUT / self._fps
            ):
                self.on_tick()
        finally:
            keyboard.unhook_all()


def main(args: Any) -> None:
    """Main entry point for mouse control application.

    Args:
        args: Command line arguments with configuration parameters.
    """
    folder = Path(__file__).parent / MODEL_DATA_FOLDER
    stats = Utils.read_json(str(folder / "stats.json"))
    model = ModelWrapper(
        timesteps=5,
        stats=stats,
        weights=dict(folder=folder, postfix=args.model),  # type: ignore[arg-type]
    )

    with ThreadedEyeTracker(fps=args.fps) as tracker:
        with LearnablePredictor(model=model, fps=args.fps) as predictor:
            app = App(
                tracker=tracker,
                predictor=predictor,
                smoothingFactor=args.smoothing,
                fps=args.fps,
                lipsMinDistance=args.lips_distance,
            )
            app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoothing", type=float, default=0.95)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--lips-distance", type=float, default=50)
    parser.add_argument("--model", type=str, default="best")
    main(parser.parse_args())
