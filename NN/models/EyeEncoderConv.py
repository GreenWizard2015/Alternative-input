"""Convolutional encoder for extracting multi-scale eye features.

Builds a convolutional encoder that extracts features from stereo eye images
at multiple scales using EyeEncoderStage modules with attention-based mixing.
"""

from typing import Any, List
import tensorflow as tf

from NN.models.EyeEncoderStage import EyeEncoderStage
from NN.layers.EncodingLayers import ConvPE
from NN.Constants import (
    EYE_ENCODER_CONV_FILTERS,
    EYE_ENCODER_CONV2LATENT_CONV_PE_CHANNELS,
    STANDARD_NONNEGATIVE_ACTIVATION,
)


class EyeEncoderConv(tf.keras.Model):
    """Convolutional encoder for extracting multi-scale eye features.

    Builds a convolutional encoder that extracts features from stereo eye images
    at multiple scales using EyeEncoderStage modules with attention-based mixing.

    Attributes:
        _latent_size: Dimension of latent feature representations
        _scale_mult: Scaling multiplier for filter dimensions
        _conv_pe: ConvPE layer for positional encoding
        _stages: List of EyeEncoderStage instances for each scale
    """

    def __init__(
        self, latent_size: int, scale_mult: float = 1.0, **kwargs: Any
    ) -> None:
        """Initialize convolutional eye encoder.

        Args:
            latent_size: Base dimension of latent feature representations.
            scale_mult: Scaling multiplier for filter dimensions (default: 1.0).
                Filters are scaled: [int(32 * scale_mult), int(32 * scale_mult)]
            **kwargs: Additional Keras Model arguments (name, trainable, dtype, etc).

        Example:
            >>> encoder = EyeEncoderConv(latent_size=256, scale_mult=1.0)
        """
        super().__init__(**kwargs)
        if scale_mult <= 0:
            raise ValueError(f"scale_mult must be positive, got {scale_mult}")

        self._latent_size = latent_size
        self._scale_mult = scale_mult
        # Use scaled latent_size for output dimensions
        scaled_latent_size = int(latent_size * scale_mult)

        self._conv_pe = ConvPE(
            channels=EYE_ENCODER_CONV2LATENT_CONV_PE_CHANNELS,
            activation=STANDARD_NONNEGATIVE_ACTIVATION,
            name="conv_pe",
        )

        # Build EyeEncoderStage instances for each scale
        scaled_filters = [int(f * scale_mult) for f in EYE_ENCODER_CONV_FILTERS]
        self._stages = []
        for scale_idx, num_filters in enumerate(scaled_filters):
            stage = EyeEncoderStage(
                latent_size=scaled_latent_size,  # Use scaled latent_size
                num_filters=num_filters,
                name=f"stage_{scale_idx}",
            )
            self._stages.append(stage)

    def build(self, input_shape):
        """Build the EyeEncoderConv layer.

        Args:
            input_shape: Shape of input tensor (batch, height, width, channels)
        """
        # Build the ConvPE layer first to ensure it's properly initialized
        self._conv_pe.build(input_shape)
        shape = self._conv_pe.compute_output_shape(input_shape)

        # Build all EyeEncoderStage instances
        for stage in self._stages:
            stage.build(shape)
            # compute_output_shape returns (feature_map_shape, latent_shape)
            # Use feature_map_shape for next stage input
            shape, _ = stage.compute_output_shape(shape)

        super().build(input_shape)

    def call(
        self, inputs: tf.Tensor, training: bool = False, **kwargs: Any
    ) -> List[tf.Tensor]:
        """Process stereo eye images through multi-scale encoder.

        Args:
            inputs: Stereo eye images with shape (batch, 32, 32, 2).
            **kwargs: Additional call arguments (e.g., training=True/False).

        Returns:
            List of attention-mixed latent features, one per scale: [(batch, latent_size), ...].

        Example:
            >>> encoder = EyeEncoderConv(latent_size=256)
            >>> eyes = tf.random.normal((32, 32, 32, 2))
            >>> features = encoder(eyes)
            >>> len(features)  # 2 scales
        """
        # Apply positional encoding once at input
        x = self._conv_pe(inputs, training=training)

        # For each stage, pass both the original input and ConvPE output
        multi_scale_features = []
        for stage in self._stages:
            # Process through EyeEncoderStage with original input
            x, latent = stage(x)
            multi_scale_features.append(latent)

        return multi_scale_features
