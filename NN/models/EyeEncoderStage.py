"""Single-stage encoder for processing eye images at one resolution level.

Encodes eye images using convolutional layers, downsampling, and attention-based
mixing to produce latent features at a specific resolution scale.
"""

from typing import Any
import tensorflow as tf
import tensorflow.keras.layers as L

from NN.layers.LinearAttentionMixer import LinearAttentionMixer
from NN.Constants import (
    EYE_ENCODER_CONV_KERNEL,
    EYE_ENCODER_CONV_PADDING,
    STANDARD_NONNEGATIVE_ACTIVATION,
)


class EyeEncoderStage(tf.keras.Model):
    """Single-stage encoder for processing eye images at one resolution level.

    Encodes eye images using convolutional layers, downsampling, and attention-based
    mixing to produce latent features at a specific resolution scale.

    Attributes:
        _latent_size: Dimension of latent feature representations
        _num_filters: Number of filters for convolutional layers
        _conv1: First convolutional layer
        _conv2: Second convolutional layer
        _conv_latent: Latent space projection layer
        _downsample: Downsampling layer
        _mixer: Linear attention mixer for feature combination
    """

    def __init__(self, latent_size: int, num_filters: int, **kwargs: Any) -> None:
        """Initialize single stage of eye encoder.

        Args:
            latent_size: Dimension of latent feature representations (already scaled).
            num_filters: Number of filters for convolutional layers.
            **kwargs: Additional Keras Model arguments (name, trainable, dtype, etc).

        Example:
            >>> stage = EyeEncoderStage(latent_size=512, num_filters=64)  # scaled version
        """
        super().__init__(**kwargs)
        if num_filters <= 0:
            raise ValueError(f"num_filters must be positive, got {num_filters}")

        self._latent_size = latent_size
        self._num_filters = num_filters

        # Create scale-specific layers
        self._conv1 = L.Conv2D(
            filters=num_filters,
            kernel_size=EYE_ENCODER_CONV_KERNEL,
            padding=EYE_ENCODER_CONV_PADDING,
            activation=STANDARD_NONNEGATIVE_ACTIVATION,
            name="stage_conv1",
        )

        self._conv2 = L.Conv2D(
            filters=num_filters,
            kernel_size=EYE_ENCODER_CONV_KERNEL,
            padding=EYE_ENCODER_CONV_PADDING,
            activation=STANDARD_NONNEGATIVE_ACTIVATION,
            name="stage_conv2",
        )

        self._conv_latent = L.Conv2D(
            filters=latent_size,
            kernel_size=1,
            padding=EYE_ENCODER_CONV_PADDING,
            activation=STANDARD_NONNEGATIVE_ACTIVATION,
            name="stage_conv_latent",
        )

        self._downsample = L.Conv2D(
            filters=num_filters,
            kernel_size=2,
            strides=2,
            padding=EYE_ENCODER_CONV_PADDING,
            activation=STANDARD_NONNEGATIVE_ACTIVATION,
            name="stage_downsample",
        )

        # Use max_dim=num_filters to get n_heads=1 via for loop calculation
        self._mixer = LinearAttentionMixer(max_dim=num_filters, name="stage_mixer")

    def build(self, input_shape):
        """Build the EyeEncoderStage layer.

        Args:
            input_shape: Shape of input tensor (batch, height, width, channels)
        """
        # Build all layers to ensure proper initialization
        self._conv1.build(input_shape)

        self._conv2.build(self._conv1.compute_output_shape(input_shape))
        self._downsample.build(
            self._conv2.compute_output_shape(
                self._conv1.compute_output_shape(input_shape)
            )
        )

        self._conv_latent.build(
            self._downsample.compute_output_shape(
                self._conv2.compute_output_shape(
                    self._conv1.compute_output_shape(input_shape)
                )
            )
        )

        # Mixer input shape: (batch, num_pixels, latent_size)
        self._mixer.build((None, None, self._latent_size))

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the EyeEncoderStage.

        Args:
            input_shape: Shape of input tensor (batch, height, width, channels)

        Returns:
            Tuple of (feature_map_shape, latent_shape) where:
            - feature_map_shape: (batch, height/2, width/2, num_filters) for next stage
            - latent_shape: (batch, latent_size) for collection
        """
        # After downsampling with stride 2, height and width are halved
        feature_map_shape = tf.TensorShape(
            [
                input_shape[0],
                input_shape[1] // 2 if input_shape[1] is not None else None,
                input_shape[2] // 2 if input_shape[2] is not None else None,
                self._num_filters,
            ]
        )
        latent_shape = tf.TensorShape([input_shape[0], self._latent_size])
        return (feature_map_shape, latent_shape)

    def call(
        self, inputs: tf.Tensor, training: bool = False, **kwargs: Any
    ) -> tf.Tensor:
        """Process eye images through single-stage encoder.

        Args:
            inputs: Eye images with shape (batch, height, width, channels).
            **kwargs: Additional call arguments (e.g., training=True/False).

        Returns:
            Latent feature tensor with shape (batch, latent_size) after attention mixing.

        Example:
            >>> stage = EyeEncoderStage(latent_size=256, scale_idx=0)
            >>> eyes = tf.random.normal((3, 32, 32, 2))
            >>> features = stage(eyes)
            >>> features.shape  # (3, 256)
        """
        # Apply convolutions
        x = self._conv1(inputs, training=training)
        x = self._conv2(x, training=training)
        x = self._downsample(x, training=training)

        # Project to latent space
        latent = self._conv_latent(x)
        B = tf.shape(latent)[0]
        N = tf.shape(latent)[-1]

        # Flatten spatial dimensions: (batch, h, w, c) -> (batch, num_pixels, c)
        latent = tf.reshape(latent, (B, -1, N))

        # Apply attention mixer: output shape (batch, 1, N) from n_outputs=1
        latent = self._mixer(latent)

        # Squeeze n_outputs dimension: (batch, 1, N) -> (batch, N)
        latent = tf.squeeze(latent, axis=1)

        return x, latent
