"""Binary classification model for filter validation.

Processes left and right eye images and predicts whether they are valid
using a shared convolutional encoder followed by fusion and classification.
"""

from typing import Any, Dict
import tensorflow as tf
from NN.layers import sMLP
from NN.models import EyeEncoder
from NN.models.NpzModelMixin import NpzModelMixin
from Core.logging_config import get_logger

logger = get_logger(__name__)


class FilterModel(NpzModelMixin, tf.keras.Model):
    """Binary classification model for filter validation.

    Processes left and right eye images through a shared convolutional encoder,
    fuses the features, and predicts a sigmoid probability indicating whether
    the eye pair is valid.

    Architecture:
        - Eye preprocessing (normalization)
        - Shared convolutional encoder with 3 layers (32, 64, 128 filters)
        - Global average pooling (reduces spatial dimensions)
        - Feature fusion (concatenation of both eye features)
        - Dense layers with dropout for classification
        - Sigmoid output (binary classification)

    Attributes:
        latent_size: Dimension of latent feature representations
    """

    def __init__(
        self,
        latent_size: int = 64,
        **kwargs: Any,
    ) -> None:
        """Initialize FilterModel.

        Args:
            latent_size: Dimension of latent feature representations (default: 64).
            **kwargs: Additional Keras Model arguments (name, trainable, dtype, etc).

        Raises:
            ValueError: If latent_size not positive.
        """
        super().__init__(**kwargs)

        # Input validation
        if latent_size <= 0:
            raise ValueError(f"latent_size must be positive, got {latent_size}")

        self._latent_size = latent_size

    def build(self, input_shape: tuple) -> None:
        self._encoder = EyeEncoder(latent_size=self._latent_size, scale_mult=8.0)
        self._mlp = sMLP(sizes=[self._latent_size] * 4, activation="relu")
        self._output = tf.keras.layers.Dense(
            units=1, activation="sigmoid", name="output"
        )

        shape = (None, 48, 48, 1)
        for layer in [self._encoder, self._mlp, self._output]:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)

        super().build(input_shape)

    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """Process eye images through the filter classification pipeline.

        Args:
            inputs: Dictionary containing:
                - 'left eye': Left eye image of shape (batch, 48, 48, 1)
                - 'right eye': Right eye image of shape (batch, 48, 48, 1)
            training: Boolean indicating training or inference mode.

        Returns:
            Dictionary with:
                - 'predictions': Sigmoid predictions of shape (batch, 1)
                - 'features': Concatenated eye features of shape (batch, 2 * latent_size)
        """
        eyes = self._encoder(
            [inputs["left eye"], inputs["right eye"]],
            training=training,
        )
        eyes = self._mlp(eyes, training=training)
        predictions = self._output(eyes)
        return {
            "predictions": predictions,
        }
