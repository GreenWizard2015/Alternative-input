#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Eye-tracking UI application with multiple interaction modes.

Provides a pygame-based application for eye-tracking data collection and visualization
with support for multiple application modes (games, calibration, data collection, etc.),
real-time gaze prediction, animated effects, and flexible tracking/prediction pipeline.
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pygame
import pygame.locals as G

import App.AppModes as AppModes
import Core.Utils as Utils
from App.AppInitializer import (
    create_camera_view,
    create_eyes_view,
    get_available_webcams,
    initialize_pygame_display,
)
from App.Background import Background
from App.DrawingUtils import draw_text
from App.EventHandlers import handle_key_down_event
from App.RandomIllumination import RandomIllumination
from App.RenderingHelpers import render_info, render_predictions
from App.TickHandlers import (
    transform_tracked_data,
    update_prediction_history,
    update_prediction_smoothing,
)
from App.Utils import Colors, numpyToSurfaceBind
from Core.data.Dataset import Dataset
from Core.models.ModelWrapper import ModelWrapper
from Core.tracking.DummyPredictor import DummyPredictor
from Core.tracking.LearnablePredictor import LearnablePredictor, PredictionResult
from Core.tracking.ThreadedEyeTracker import ThreadedEyeTracker
from Core.logging_config import get_logger

logger = get_logger(__name__)

# Prediction smoothing parameters
GAZE_PREDICTION_SMOOTHING_FACTOR = 0.9

# Display layout constants
CAMERA_VIEW_X = 50
CAMERA_VIEW_Y = 200
CAMERA_VIEW_SIZE = 300
EYES_VIEW_Y_OFFSET = 50
EYES_VIEW_HEIGHT = 100
TEXT_INFO_START_X = 5
TEXT_INFO_START_Y = 95
TEXT_LINE_HEIGHT = 25
PREDICTION_HISTORY_LIMIT = 15
TARGET_DRAW_STEP = 2
CIRCLE_WIDTH_MULTIPLIER = 7
TEXT_POSITION_X = 5
TEXT_POSITION_Y = 5

# User, screen, camera, monitor, and place identifiers for model configuration
DEFAULT_USER_ID = "ce42c1a9-f4ef-42d6-a219-cf25fad912ed"
DEFAULT_SCREEN_ID = "a28a2ad8-4349-b038-5af8-46a658e82543"
DEFAULT_CAMERA_ID = "camera1"
DEFAULT_MONITOR_ID = "monitor1"
DEFAULT_PLACE_ID = "de0ce61b-2fc0-02f6-efb5-22af447bfb05"


class App:
    """Main eye-tracking application with pygame interface.

    Manages the complete eye-tracking pipeline including facial data capture,
    model prediction, data collection, and interactive mode selection. Supports
    multiple visualization modes and real-time data processing with optional
    webcam and eye region display.

    Attributes:
        _tracker: Eye/face tracker providing real-time facial landmarks and eye images
        _dataset: Storage for collecting training samples during collection modes
        _predictor: Model wrapper for making gaze predictions from facial data
        _currentMode: Active application mode (game, collection, visualization, etc.)
        _illumination: Animated light sources for background effects
        _background: Background renderer with dynamic color and brightness
        _smoothedPrediction: Exponentially smoothed gaze prediction for display
    """

    def __init__(
        self,
        tracker: ThreadedEyeTracker,
        dataset: Dataset,
        predictor: Union[LearnablePredictor, DummyPredictor],
        fps: int = 30,
        hasPredictions: bool = True,
        showWebcam: bool = False,
        showFaceMesh: bool = False,
        showEyes: bool = True,
        current_webcam: Union[int, str] = 0,
    ) -> None:
        self._showFaceMesh = showFaceMesh
        self._faceMesh: Optional[np.ndarray] = None
        self._canPredict = hasPredictions
        self._fps = fps
        self._running = True

        self._lastPrediction: Optional[PredictionResult] = None
        self._smoothedPrediction = np.array([0.0, 0.0])
        self._showPredictions = True

        self._tracker = tracker
        self._dataset = dataset
        self._predictor = predictor

        self._currentModeId = 0
        self._currentMode = AppModes.APP_MODES[0](app=self)

        self._history: List[np.ndarray] = []
        self._enableIllumination = False
        self._illumination = RandomIllumination()
        self._background = Background()

        self._cameraView = self._cameraSurface = None
        if showWebcam:
            self._cameraView, self._cameraSurface = create_camera_view(
                CAMERA_VIEW_X, CAMERA_VIEW_Y, CAMERA_VIEW_SIZE
            )

        self._eyesView = self._eyesSurface = None
        if showEyes:
            self._eyesView, self._eyesSurface = create_eyes_view(
                CAMERA_VIEW_X,
                CAMERA_VIEW_Y,
                CAMERA_VIEW_SIZE,
                EYES_VIEW_Y_OFFSET,
                EYES_VIEW_HEIGHT,
            )

        self._predictorMaskFace = False
        self._predictorMaskLeftEye = False
        self._predictorMaskRightEye = False

        self._webcam = (
            int(current_webcam)
            if isinstance(current_webcam, str) and current_webcam.isdigit()
            else (current_webcam if isinstance(current_webcam, int) else 0)
        )
        self._webcams_list = get_available_webcams()

    @property
    def hasPredictions(self) -> bool:
        """Check if prediction model is available.

        Returns:
            True if predictions can be made, False otherwise.
        """
        return self._canPredict

    def _transformTracked(
        self, tracked: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Transform tracked facial data by applying predictor masks.

        Args:
            tracked: Dictionary with 'tracked' key containing facial data.

        Returns:
            Modified tracked data dictionary or None if input is None.
        """
        return transform_tracked_data(
            tracked,
            self._predictorMaskFace,
            self._predictorMaskLeftEye,
            self._predictorMaskRightEye,
        )

    @property
    def _display_surf(self) -> pygame.Surface:
        """Get the pygame display surface.

        Returns:
            The current pygame display surface.

        Raises:
            RuntimeError: If pygame display has not been initialized.
        """
        return pygame.display.get_surface()

    @property
    def WH(self) -> np.ndarray:
        """Get window width and height.

        Returns:
            (width, height) as float32 numpy array.
        """
        return np.array(pygame.display.get_surface().get_size(), np.float32)

    def on_init(self) -> bool:
        """Initialize pygame and set up display window."""
        _, self._font = initialize_pygame_display()
        self._running = True
        return True

    def on_event(self, event: Any) -> None:
        """Handle pygame events.

        Args:
            event: Pygame event object.
        """
        if event.type == G.QUIT:
            self._running = False
            return

        self._background.on_event(event)
        self._currentMode.on_event(event)

        if event.type == G.KEYDOWN:
            if event.key == G.K_ESCAPE:
                self._running = False
                return

            handle_key_down_event(
                event.key,
                len(AppModes.APP_MODES),
                self._toggle_predictions,
                self._toggle_illumination,
                self._toggle_face_mask,
                self._toggle_left_eye_mask,
                self._toggle_right_eye_mask,
                self._change_mode,
            )

    def _toggle_predictions(self) -> None:
        """Toggle predictions display."""
        self._showPredictions = not self._showPredictions

    def _toggle_illumination(self) -> None:
        """Toggle illumination."""
        self._enableIllumination = not self._enableIllumination

    def _toggle_face_mask(self) -> None:
        """Toggle face mask."""
        self._predictorMaskFace = not self._predictorMaskFace

    def _toggle_left_eye_mask(self) -> None:
        """Toggle left eye mask."""
        self._predictorMaskLeftEye = not self._predictorMaskLeftEye

    def _toggle_right_eye_mask(self) -> None:
        """Toggle right eye mask."""
        self._predictorMaskRightEye = not self._predictorMaskRightEye

    def _change_mode(self, mode_index: int) -> None:
        """Change application mode.

        Args:
            mode_index: Index of the new mode.
        """
        self._currentModeId = mode_index
        self._currentMode = AppModes.APP_MODES[mode_index](app=self)

    def _updateEyesImage(self, tracked: Optional[Dict[str, Any]]) -> None:
        """Update eye region display from tracked facial data.

        Args:
            tracked: Tracked facial data dictionary or None.

        Note:
            Eye surface rendering is currently not implemented.
        """
        if self._eyesSurface is None or tracked is None:
            return

    def on_tick(self, delta_t: float) -> None:
        """Update application state for a single frame.

        Args:
            delta_t: Time delta since last frame in seconds.
        """
        # Track facial landmarks and update prediction
        lastTracked = None
        tracked = self._tracker.track()
        if tracked is not None:
            self._currentMode.on_sample(tracked)
            self._faceMesh = tracked["face points"].copy()

            if self._cameraView is not None and self._cameraSurface is not None:
                numpyToSurfaceBind(tracked["raw"][..., ::-1], self._cameraSurface)

            self._updateEyesImage(tracked)
            lastTracked = {
                "tracked": tracked,
                "pos": np.array(self._smoothedPrediction, np.float32),
            }

        # Get gaze prediction from model
        prediction: Optional[PredictionResult] = self._predictor(
            self._transformTracked(lastTracked)
        )
        if prediction is not None:
            self._lastPrediction = prediction
            predPos = prediction.prediction.result[0, -1]
            self._history = update_prediction_history(
                self._history, predPos, PREDICTION_HISTORY_LIMIT
            )
            self._currentMode.on_prediction(predPos, lastTracked)

        # Apply exponential smoothing to prediction
        if self._lastPrediction:
            pred_tuple = self._lastPrediction
            predPos = pred_tuple.prediction.result[0, -1]
            self._smoothedPrediction = update_prediction_smoothing(
                self._smoothedPrediction, predPos, GAZE_PREDICTION_SMOOTHING_FACTOR
            )

        # Update background and current mode
        self._background.on_tick(delta_t)
        self._currentMode.on_tick(delta_t)
        if self._enableIllumination:
            self._illumination.on_tick(delta_t)

    def on_render(self, fps: float = 0.0) -> None:
        """Render current frame.

        Args:
            fps: Frames per second for display (default: 0.0).
        """
        window = self._display_surf
        self._background.on_render(window)

        if self._enableIllumination:
            self._illumination.on_render(window)

        if self._cameraSurface is not None and self._cameraView is not None:
            window.blit(self._cameraSurface, tuple(self._cameraView[0].astype(int)))

        if self._eyesSurface is not None and self._eyesView is not None:
            window.blit(self._eyesSurface, tuple(self._eyesView[0].astype(int)))

        self._currentMode.on_render(window)
        if self._currentMode.paused:
            wh = np.array(window.get_size())
            txt = "Collection paused"
            draw_text(
                self._font,
                window,
                text=txt,
                pos=tuple((wh // 2).astype(int)),
                color=Colors.RED,
                scale=2.0,
                center=True,
            )

        render_predictions(
            self._display_surf,
            self._font,
            self._history,
            self._smoothedPrediction,
            self._showPredictions,
            TEXT_POSITION_X,
            TEXT_POSITION_Y,
        )

        self._renderInfo(fps=fps)
        pygame.display.flip()

    def _renderInfo(self, fps: float) -> None:
        """Render debug information overlay.

        Args:
            fps: Frames per second to display.
        """
        render_info(
            self._display_surf,
            self._font,
            self._dataset.totalSamples,
            (
                self._predictorMaskFace,
                self._predictorMaskLeftEye,
                self._predictorMaskRightEye,
            ),
            fps,
            self.WH,
            self._showFaceMesh,
            self._faceMesh,
            self._webcams_list,
            self._webcam,
            TEXT_INFO_START_X,
            TEXT_INFO_START_Y,
            TEXT_LINE_HEIGHT,
        )

    def run(self) -> None:
        """Run main application loop.

        Initializes pygame window, processes events, updates state, and renders frames
        at the configured FPS rate until the application is closed.
        """
        if not self.on_init():
            self._running = False

        T = pygame.time.get_ticks()
        clock = pygame.time.Clock()
        while self._running:
            for event in pygame.event.get():
                self.on_event(event)

            TMs = (pygame.time.get_ticks() - T) / 1000.0
            fps = 1.0 / TMs if 0.0 < TMs else 0.0
            self.on_tick(TMs)
            self.on_render(fps=fps)
            T = pygame.time.get_ticks()
            clock.tick(self._fps)

        pygame.quit()


def _modelFromArgs(args: Any) -> Optional[ModelWrapper]:
    """Create model wrapper from command line arguments.

    Args:
        args: Command line arguments with model configuration.

    Returns:
        ModelWrapper instance or None if model is 'none'.
    """
    if args.model.lower() == "none":
        return None

    stats = Utils.read_json(str(Path(args.folder) / "stats.json"))

    return ModelWrapper(
        timesteps=args.steps,
        user=dict(
            userId=DEFAULT_USER_ID,
            screenId=DEFAULT_SCREEN_ID,
            cameraId=DEFAULT_CAMERA_ID,
            monitorId=DEFAULT_MONITOR_ID,
            placeId=DEFAULT_PLACE_ID,
        ),
        stats=stats,
    )


def _predictorFromArgs(args: Any) -> Union[LearnablePredictor, DummyPredictor]:
    """Create predictor instance from command line arguments.

    Args:
        args: Command line arguments with predictor configuration.

    Returns:
        LearnablePredictor or DummyPredictor depending on model availability.
    """
    model = _modelFromArgs(args)
    if model is None:
        return DummyPredictor()
    return LearnablePredictor(model=model, fps=args.fps)


def main(args: Any) -> None:
    """Main entry point for eye-tracking UI application.

    Args:
        args: Command line arguments with configuration parameters.
    """
    if args.webcam.isdigit():
        args.webcam = int(args.webcam)
    folder = args.folder
    with ThreadedEyeTracker(webcam=args.webcam) as tracker, Dataset(
        str(Path(folder) / "Dataset"), args.steps
    ) as dataset:
        with _predictorFromArgs(args) as predictor:
            app = App(
                tracker=tracker,
                dataset=dataset,
                predictor=predictor,
                fps=args.fps,
                hasPredictions=predictor.canPredict,
                current_webcam=args.webcam,
            )
            app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, default=str(Path(__file__).parent / "Data")
    )
    parser.add_argument("--steps", type=int, default=5)
    # if 'none' - no model will be used
    parser.add_argument("--model", type=str, default="best")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--webcam", type=str, default="0")
    main(parser.parse_args())
