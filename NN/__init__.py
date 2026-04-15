"""Neural network model re-exports for backwards compatibility.

This module provides convenient access to all main neural network model classes
used for eye-tracking and gaze prediction. It re-exports models from the NN.models
subpackage.

Exported Models:
    - GazePredictionModel: Complete end-to-end model (Face2Step + Step2Latent)
"""

from NN.models.GazePredictionModel import GazePredictionModel

__all__ = [
    "GazePredictionModel",
]
