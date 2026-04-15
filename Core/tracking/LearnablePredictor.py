"""Learnable predictor with asynchronous inference.

Wraps a trained model and runs inference in a background thread, allowing
non-blocking predictions with configurable frame rate.
"""

from typing import Any, Dict, List, Optional, Protocol
import threading
import Core.Utils as Utils
import numpy as np
from Core.logging_config import get_logger
from Core.models.ModelWrapper import ModelWrapper
from Core.tracking.PredictionResult import PredictionResult

logger = get_logger(__name__)


class InferenceData(Protocol):
    """Protocol for data submitted for inference.

    Can be either:
    - Dict[str, Any]: Single frame data (when timesteps not required)
    - List[Dict[str, Any]]: Multiple frames for temporal models
    """

    def __iter__(self) -> Any:
        """Support iteration over inference data."""
        ...


class LearnablePredictor:
    """Async predictor that runs inference in background thread.

    Accepts frames at any rate, buffers them to match model timesteps,
    and produces predictions asynchronously at a fixed frame rate.

    Attributes:
        _lock: Thread lock for synchronization
        _done: Event to signal thread termination
        _inferData: Current data queued for inference
        _inferResults: Latest inference results
        _model: Trained model for prediction
        _timesteps: Required number of frames for model
        _prevSteps: Buffer of previous frames
        _fps: Inference frame rate
        _thread: Background inference thread
    """

    def __init__(self, model: ModelWrapper, fps: int = 30) -> None:
        """Initialize learnable predictor.

        Args:
            model: Trained model with 'timesteps' attribute and __call__ method
            fps: Inference frame rate (default: 30)

        Raises:
            ValueError: If fps is not positive
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._inferData: Optional[InferenceData] = None
        self._inferResults: Optional[PredictionResult] = None
        self._model: ModelWrapper = model
        self._timesteps = self._model.timesteps
        self._prevSteps: List[Dict[str, Any]] = []
        self._fps = fps

    def __enter__(self) -> "LearnablePredictor":
        """Enter context manager: start background inference thread.

        Returns:
            Self for context manager protocol
        """
        self._thread = threading.Thread(target=self._loop, daemon=False)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: Any,  # type: ignore[unused-argument]
        exc_val: Any,  # type: ignore[unused-argument]
        exc_tb: Any,  # type: ignore[unused-argument]
    ) -> None:
        """Exit context manager: stop background inference thread.

        Args:
            exc_type: Exception type - unused, required by context manager protocol
            exc_val: Exception value - unused, required by context manager protocol
            exc_tb: Exception traceback - unused, required by context manager protocol
        """
        self._done.set()
        self._thread.join()

    def __call__(self, data: Optional[Dict[str, Any]]) -> Optional[PredictionResult]:
        """Make prediction by calling async_infer.

        Args:
            data: New frame data (Dict) or None to skip submission

        Returns:
            PredictionResult if inference completed, or None if no prediction available yet
        """
        return self.async_infer(data)

    def async_infer(self, data: Optional[Dict[str, Any]]) -> Optional[PredictionResult]:
        """Submit data for inference and retrieve latest results.

        Thread-safe method to queue new data and get previous inference results.
        Accumulates frames to match model timesteps.

        Args:
            data: New frame data (Dict) or None to skip submission

        Returns:
            PredictionResult if inference completed, or None if no prediction available yet
        """
        with self._lock:
            if data is not None:
                if self._timesteps:
                    arr = self._prevSteps + [data]
                    self._prevSteps = list(arr[-self._timesteps :])  # COPY of list
                    self._inferData = self._prevSteps  # same as self._prevSteps
                else:
                    self._inferData = data

            res = self._inferResults
            self._inferResults = None
        return res

    def _loop(self) -> None:
        """Background inference loop.

        Runs at fixed frame rate, pulling queued data and running model inference.
        """
        while not self._done.wait(1.0 / self._fps):
            self._infer()

    def _infer(self) -> None:
        """Execute single inference step.

        Retrieves queued data, prepares input tensors, runs model,
        and stores results for retrieval.
        """
        with self._lock:
            data: Optional[InferenceData] = self._inferData
            self._inferData = None
        if data is None:
            return

        if not isinstance(data, list):
            logger.warning("Inference data is not a list, expected list of frames")
            return

        if len(data) != self._timesteps:
            logger.warning(
                f"Inference data length {len(data)} does not match required timesteps {self._timesteps}"
            )
            return
        samples_list: List[Dict[str, Any]] = [
            Utils.tracked2sample(data=x) for x in data
        ]
        samples_dict: Dict[str, Any] = Utils.samples2inputs(samples=samples_list)
        T = np.diff(samples_dict["time"], n=1)
        T = np.insert(T, obj=0, values=0.0)
        samples_dict["time"] = T.reshape((self._timesteps, 1))
        X = {
            k: x[None] for k, x in samples_dict.items()
        }  # Add batch dimension: (timesteps, ...) => (1, timesteps, ...)

        # Use the most recent frame from the sequence as the current tracked state
        last_tracked: Dict[str, Any] = data[-1]
        res = self._model(X)

        with self._lock:
            self._inferResults = PredictionResult(prediction=res, tracked=last_tracked)

    @property
    def canPredict(self) -> bool:
        """Check if predictor can make predictions.

        Returns:
            True - this predictor can make predictions
        """
        return True
