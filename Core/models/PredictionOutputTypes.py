"""Output types for gaze prediction pipeline.

Provides structured output classes for model predictions in both tensor and numpy formats.
"""

from typing import Dict, NamedTuple

import numpy as np
import tensorflow as tf


class PredictionOutput(NamedTuple):
    """Structured output from gaze prediction pipeline.

    Attributes:
        result: Predicted gaze coordinates of shape (batch_size, timesteps, 2)
        raw: Raw output dictionary from PredictorBlock containing all outputs
        latents: Final GazePredictionModel latent tensor of shape (batch_size, timesteps, latent_size)
        intermediate_latents: Intermediate latent from Face2Step of shape (batch_size, timesteps, latent_size)
    """

    result: tf.Tensor
    raw: Dict[str, tf.Tensor]
    latents: tf.Tensor
    intermediate_latents: tf.Tensor

    def slice(self, start_idx, end_idx):
        return PredictionOutput(
            intermediate_latents=self.intermediate_latents[start_idx:end_idx],
            latents=self.latents[start_idx:end_idx],
            result=self.result[start_idx:end_idx],
            raw=None,
        )


class PredictionOutputNumpy(NamedTuple):
    """Structured numpy output from gaze prediction pipeline.

    Numpy-based version of PredictionOutput for external API consumption.

    Attributes:
        result: Predicted gaze coordinates as numpy array of shape (batch_size, timesteps, 2)
    """

    result: np.ndarray
