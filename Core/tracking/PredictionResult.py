"""Prediction result type for learnable predictor.

Provides structured output for tracking predictions with associated frame data.
"""

from typing import Any, Dict, NamedTuple
from Core.models.PredictionOutputTypes import PredictionOutputNumpy


class PredictionResult(NamedTuple):
    """Result from prediction inference.

    Attributes:
        prediction: Predicted gaze output from model
        tracked: Last tracked frame data used for prediction
    """

    prediction: PredictionOutputNumpy
    tracked: Dict[str, Any]
