"""Threaded eye tracker for non-blocking face tracking.

Wraps EyeTracker and runs tracking in a background thread at a fixed frame rate,
allowing non-blocking access to the latest tracking results.
"""

from typing import Any, Dict, Optional
from Core.logging_config import get_logger
from Core.tracking.EyeTracker import EyeTracker
import threading

logger = get_logger(__name__)


class ThreadedEyeTracker:
    """Threaded wrapper around eye tracker for asynchronous tracking.

    Runs face mesh tracking in a background thread at a fixed frame rate,
    allowing the main thread to query results without blocking.

    Attributes:
        _lock: Thread lock for synchronization
        _done: Event to signal thread termination
        _results: Latest tracking results
        _fps: Tracking frame rate
        _webcam: Webcam device index
        _tracker: Underlying eye tracker instance
        _thread: Background tracking thread
    """

    def __init__(self, fps: int = 30, webcam: int = 0) -> None:
        """Initialize threaded eye tracker.

        Args:
            fps: Tracking frame rate (default: 30)
            webcam: Webcam device index (default: 0)

        Raises:
            ValueError: If fps is not positive or webcam index is negative
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if webcam < 0:
            raise ValueError(f"webcam index must be non-negative, got {webcam}")
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._results: Optional[Dict[str, Any]] = None
        self._fps = fps
        self._webcam = webcam

    def __enter__(self) -> "ThreadedEyeTracker":
        """Enter context manager: initialize tracker and start thread.

        Initializes the underlying EyeTracker and starts the background
        tracking thread.

        Returns:
            Self for context manager protocol
        """
        # Note: We manually manage EyeTracker lifecycle here to keep it open
        # for the duration of this context manager, so we call __enter__ directly
        # rather than using 'with' statement
        self._tracker = EyeTracker(webcam=self._webcam)
        self._tracker = self._tracker.__enter__()  # type: ignore[assignment]

        self._thread = threading.Thread(target=self._track_loop, daemon=False)
        self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: Any,  # type: ignore[unused-argument]
        _exc_val: Any,  # type: ignore[unused-argument]
        _exc_tb: Any,  # type: ignore[unused-argument]
    ) -> None:
        """Exit context manager: stop thread and cleanup tracker.

        Signals the thread to stop, waits for it to finish, and closes
        the underlying tracker.

        Args:
            _exc_type: Exception type - unused, required by context manager protocol
            _exc_val: Exception value - unused, required by context manager protocol
            _exc_tb: Exception traceback - unused, required by context manager protocol
        """
        self._done.set()
        self._thread.join()
        self._tracker.__exit__(_exc_type, _exc_val, _exc_tb)

    def track(self) -> Optional[Dict[str, Any]]:
        """Retrieve latest tracking results.

        Thread-safe method to get the most recent tracking results and clear
        them so they're only retrieved once.

        Returns:
            Latest tracking results dictionary or None if no new results available.
            Dictionary contains keys like 'time', 'face points', 'left eye', etc.
        """
        with self._lock:
            res = self._results
            self._results = None
        return res

    def _track_loop(self) -> None:
        """Background tracking loop.

        Runs at fixed frame rate, continuously tracking face and updating
        results for retrieval by main thread.
        """
        # wait for the event to be set up to 1/fps seconds
        while not self._done.wait(timeout=1.0 / self._fps):
            res = self._tracker.track()
            with self._lock:
                self._results = res
