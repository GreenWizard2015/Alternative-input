"""Dummy predictor for testing and fallback purposes.

Implements the predictor interface but always returns None and reports
inability to predict. Useful for testing and fallback scenarios.
"""

from typing import Any
from Core.logging_config import get_logger

logger = get_logger(__name__)


class DummyPredictor:
    """Dummy predictor that doesn't perform actual predictions.

    Implements the predictor context manager interface but returns None
    for all prediction requests. Useful for testing and graceful fallbacks
    when actual predictor is unavailable.

    Attributes:
        canPredict: Always False - indicates this predictor cannot predict
    """

    def __init__(self) -> None:
        """Initialize dummy predictor.

        This is a no-op initializer for interface compatibility.
        """

    def __enter__(self) -> "DummyPredictor":
        """Enter context manager.

        Returns:
            Self for context manager protocol
        """
        return self

    def __exit__(
        self,
        exc_type: Any,  # type: ignore[unused-argument]
        exc_val: Any,  # type: ignore[unused-argument]
        exc_tb: Any,  # type: ignore[unused-argument]
    ) -> None:
        """Exit context manager.

        Args:
            exc_type: Exception type (if any, required by context manager protocol)
            exc_val: Exception value (if any, required by context manager protocol)
            exc_tb: Exception traceback (if any, required by context manager protocol)
        """

    def __call__(self, data: Any) -> None:  # type: ignore[unused-argument]
        """Make prediction (always returns None).

        Args:
            data: Input data (ignored, required for interface compatibility)

        Returns:
            None - this dummy predictor does not make predictions
        """
        self.async_infer(data=data)
        return None

    def async_infer(self, data: Any) -> None:  # type: ignore[unused-argument]
        """Dummy inference - no-op for dummy predictor.

        Args:
            data: Input data (ignored, required for interface compatibility)

        Returns:
            None - this dummy predictor does not make predictions
        """

    @property
    def canPredict(self) -> bool:
        """Check if predictor can make predictions.

        Returns:
            False - this dummy predictor cannot predict
        """
        return False
