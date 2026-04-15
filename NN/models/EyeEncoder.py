"""Encoder for stereo eye images with attention-based multi-scale fusion.

Encodes left and right eye images using a shared convolutional encoder that extracts
multi-scale features, then combines them using attention-weighted pooling for robust
feature representation.
"""

from typing import Any, List
import tensorflow as tf
import tensorflow.keras.layers as L

from NN.models.EyeEncoderConv import EyeEncoderConv
from NN.Constants import STANDARD_NONNEGATIVE_ACTIVATION


class EyeEncoder(tf.keras.Model):
    """Encoder for stereo eye images with attention-based multi-scale fusion.

    Encodes left and right eye images using a shared convolutional encoder that extracts
    multi-scale features, then combines them using attention-weighted pooling for robust
    feature representation.

    Attributes:
        _latent_size: Base dimension of latent representations
        _scale_mult: Scaling multiplier for filter dimensions
        _encoder: Internal convolutional encoder model that outputs multi-scale features
        _eye_mixer: Mixer layer for combining scale features
    """

    def __init__(
        self, latent_size: int, scale_mult: float = 1.0, **kwargs: Any
    ) -> None:
        """Initialize eye encoder.

        Creates an encoder that processes stereo eye images (left and right)
        and outputs latent features with attention-based mixing.

        Args:
            latent_size: Base dimension of latent feature representations.
            scale_mult: Scaling multiplier for filter dimensions (default: 1.0).
                Filters are scaled: [int(32 * scale_mult), int(32 * scale_mult)]
            **kwargs: Additional Keras Model arguments (name, trainable, dtype, etc).

        Example:
            >>> encoder = EyeEncoder(latent_size=256, scale_mult=1.0)
        """
        super().__init__(**kwargs)
        if scale_mult <= 0:
            raise ValueError(f"scale_mult must be positive, got {scale_mult}")

        self._latent_size = latent_size
        self._scale_mult = scale_mult
        # Use scaled latent_size for mixer output dimension
        scaled_latent_size = int(latent_size * scale_mult)
        self._mixer = L.Dense(
            scaled_latent_size,
            activation=STANDARD_NONNEGATIVE_ACTIVATION,
            name="eye_mixer",
        )
        # Define the shared encoder, passing scale_mult for filter scaling
        self._encoder = EyeEncoderConv(
            latent_size=latent_size,
            scale_mult=scale_mult,
            name="eye_encoder_conv",
        )

    def build(self, input_shape):
        """Build the EyeEncoder model.

        Args:
            input_shape: Shape of input tensor(s). For eye encoder, expects list of shapes
                [(batch, height, width, channels), (batch, height, width, channels)] for
                left and right eyes respectively.
        """
        # Default shape if input_shape is invalid - stereo shape with batch dimension
        default_shape = list(input_shape[:3]) + [2]
        self._encoder.build(default_shape)

        # Build the mixer layer based on expected output dimensions
        # EyeEncoderConv returns list of scaled_latent_size features per stage
        # With 2 stages, total features = 2 * scaled_latent_size
        scaled_latent_size = int(self._latent_size * self._scale_mult)
        self._mixer.build((1, 2 * scaled_latent_size))

        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """Encode stereo eye images with attention-based scale mixing.

        Concatenates left and right eye images, applies dual-scale convolutional encoding
        where each scale is individually mixed via attention mechanism, then combines
        the scale features for final representation.

        Args:
            inputs: List of two tensors [left_eye, right_eye], each with shape
                (batch, height, width, channels).
            training: Boolean flag for training mode (default: False).

        Returns:
            Mixed latent feature tensor with shape (batch, latent_size),
            combining attention-mixed scale features.

        Example:
            >>> encoder = EyeEncoder(latent_size=256)
            >>> left_eye = tf.random.normal((32, 32, 32, 1))
            >>> right_eye = tf.random.normal((32, 32, 32, 1))
            >>> features = encoder([left_eye, right_eye])
            >>> features.shape  # (32, 256)
        """
        left_eye_image, right_eye_image = inputs
        B = tf.shape(left_eye_image)[0]

        # Concatenate left and right eyes: (batch, height, width, 2*channels)
        stereo_eye_images = tf.concat([left_eye_image, right_eye_image], -1)
        # Get attention-mixed scale features from encoder
        eye_features_list = self._encoder(stereo_eye_images, training=training)
        # Stack scale features and apply final mixing
        eye_features_stacked = tf.concat(eye_features_list, axis=-1)
        features = self._mixer(eye_features_stacked)
        scaled_latent_size = int(self._latent_size * self._scale_mult)
        tf.debugging.assert_equal(
            tf.shape(features),
            (B, scaled_latent_size),
            message=f"Expected features shape (batch={B}, latent={scaled_latent_size}), got {tf.shape(features)}",
        )

        return features

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the EyeEncoder.

        Args:
            input_shape: Shape of input tensor(s). Expected to be list of shapes
                [(batch, height, width, channels), (batch, height, width, channels)]
                for left and right eyes.

        Returns:
            Output shape as tuple (batch, scaled_latent_size).
        """
        # Input shape should be a list of two eye shapes
        # We just need the batch size from the input
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0:
            batch_size = (
                input_shape[0][0] if hasattr(input_shape[0], "__getitem__") else None
            )
        else:
            batch_size = None

        scaled_latent_size = int(self._latent_size * self._scale_mult)
        return (batch_size, scaled_latent_size)
