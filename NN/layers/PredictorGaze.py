"""Layer for predicting gaze points from latent features."""

from typing import Any
import tensorflow as tf
import tensorflow.keras.layers as L
from NN.Constants import (
    PREDICTOR_BLOCK_OUTPUT_DIMS,
    PREDICTOR_BLOCK_SHIFT,
    PREDICTOR_BLOCK_ACTIVATION,
    PREDICTOR_BLOCK_MLP_SIZES,
)
from NN.layers.sMLP import sMLP


class PredictorGaze(tf.keras.layers.Layer):
    """Predicts gaze points (2D coordinates) from latent representations.

    Architecture:
        - sMLP: Processes latent features through multiple dense layers
        - Dense layer: Projects to 2D gaze coordinates
        - Shift: Applied offset for normalization (default: 0.5)

    Attributes:
        _shift: Scalar offset applied to predicted points for normalization
        _mlp: sMLP for feature processing
        _dense: Dense layer decoding to gaze points
    """

    def __init__(
        self,
        shift: float = PREDICTOR_BLOCK_SHIFT,
        **kwargs: Any,
    ) -> None:
        """Initialize the PredictorGaze layer.

        Args:
            shift: Scalar value to shift predicted points. Used to normalize output
                to desired range (e.g., [0.5, 0.5] centers points).
                Default: PREDICTOR_BLOCK_SHIFT.
            **kwargs: Additional keyword arguments passed to parent Layer class.
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)
        self._shift = shift

        # Store sublayer names directly

    def build(self, input_shape):
        """Build the layer with input shape.

        Args:
            input_shape: Input tensor shape (batch, seq_len, latent_dim).
        """
        # Create sMLP for feature processing
        self._mlp = sMLP(
            sizes=PREDICTOR_BLOCK_MLP_SIZES,
            activation=PREDICTOR_BLOCK_ACTIVATION,
            name="MLP",
        )
        self._mlp.build(input_shape)

        # Dense layer for gaze prediction
        self._dense = L.Dense(units=PREDICTOR_BLOCK_OUTPUT_DIMS, name="Dense")
        self._dense.build(
            (input_shape[0], input_shape[1], PREDICTOR_BLOCK_MLP_SIZES[-1])
        )

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Predict gaze points from latent features.

        Args:
            x: Input tensor of shape (batch, seq_len, latent_dim).
            training: Whether in training mode (default: False).
            **kwargs: Additional call arguments.

        Returns:
            Predicted gaze points tensor of shape (batch, seq_len, 2)
        """
        mlp_output = self._mlp(x, training=training)
        points = self._shift + self._dense(mlp_output, training=training)
        return points
