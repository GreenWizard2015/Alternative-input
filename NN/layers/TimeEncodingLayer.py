"""Time encoding layer using shallow coordinate-based encoding."""

from typing import Any
import tensorflow as tf
from NN.layers.ShallowEncoderLayer import ShallowEncoderLayer
from NN.layers.LinearAttentionMixer import LinearAttentionMixer
from NN.Constants import TIME_NORMALIZATION_EPSILON


class TimeEncodingLayer(tf.keras.layers.Layer):
    """Encodes temporal information using shallow coordinate-based encoding.

    Wraps ShallowEncoderLayer for efficient time encoding with learnable
    coordinate encoding scheme.

    Attributes:
        _shallow_encoder: ShallowEncoderLayer for coordinate-based time encoding
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the time encoding layer.

        Args:
            **kwargs: Additional keyword arguments passed to parent Layer class.
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build the layer with input shape.

        Args:
            input_shape: Input tensor shape (batch, timesteps, 1)
        """
        self._encoder = ShallowEncoderLayer(name="encoder")
        self._attention_mixer = LinearAttentionMixer(name="AttentionMixer")

        self._encoder.build((None, input_shape[1], 1))
        # Create attention mixer sublayer
        self._attention_mixer.build(
            (
                None,
                input_shape[1],
                4,
                self._encoder.output_shape[-1],
            )
        )

        super().build(input_shape)

    def _relative_times(self, times: tf.Tensor) -> tf.Tensor:
        """Compute relative times by subtracting minimum.

        Args:
            times: Input times of shape (batch_size, timesteps, 1).

        Returns:
            Relative times with minimum subtracted, shape (batch_size, timesteps, 1).
        """
        times_min = tf.reduce_min(times, axis=1, keepdims=True)
        return times - times_min

    def _normalize_times(self, times: tf.Tensor) -> tf.Tensor:
        """Normalize times to relative [0, 1] range per batch.

        Args:
            times: Input times of shape (batch_size, timesteps, 1).

        Returns:
            Normalized times of shape (batch_size, timesteps, 1).
        """
        times_min = tf.reduce_min(times, axis=1, keepdims=True)
        times_max = tf.reduce_max(times, axis=1, keepdims=True)
        times_range = times_max - times_min + TIME_NORMALIZATION_EPSILON
        return (times - times_min) / times_range

    def _diff_times(self, times: tf.Tensor) -> tf.Tensor:
        """Compute time differences with zero-padding for first timestep.

        Args:
            times: Input times of shape (batch_size, timesteps, 1)

        Returns:
            Time differences of shape (batch_size, timesteps, 1) with first element as 0
        """
        diffs = times[:, 1:] - times[:, :-1]
        # Zero-pad the first timestep to maintain original shape
        zero_pad = tf.zeros(tf.shape(times[:, :1]), dtype=times.dtype)
        return tf.concat([zero_pad, diffs], axis=1)

    def call(self, times: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Encode temporal input through shallow encoder.

        Args:
            times: Input tensor of shape (batch_size, timesteps, 1) containing time values
                in normalized range [0, 1].
            training: Boolean indicating training or inference mode.

        Returns:
            Encoded times tensor of shape (batch_size, timesteps, 32).

        Example:
            >>> time_encoder = TimeEncodingLayer()
            >>> times = tf.constant([[[0.0], [0.5], [1.0]]], dtype=tf.float32)
            >>> encoded = time_encoder(times)
            >>> assert encoded.shape == (1, 3, 32)
        """
        tf.debugging.assert_rank(times, 3, message="rank must be 3")
        tf.debugging.assert_equal(
            tf.shape(times)[-1], 1, message="Last dimension must be 1"
        )
        B = tf.shape(times)[0]
        T = tf.shape(times)[1]

        times = self._relative_times(times)
        norm_times = self._normalize_times(times)

        rel_encoded = self._encoder(times, training=training)
        norm_encoded = self._encoder(norm_times, training=training)
        diff_encoded = self._encoder(self._diff_times(times), training=training)
        norm_diff_encoded = self._encoder(
            self._diff_times(norm_times), training=training
        )

        # Stack all encoded representations and mix them using linear attention
        stacked_encodings = tf.stack(
            [rel_encoded, norm_encoded, diff_encoded, norm_diff_encoded], axis=2
        )
        tf.debugging.assert_equal(tf.shape(stacked_encodings)[:-1], (B, T, 4))

        # Output shape: (B, T, 1, encoding_dim) from n_outputs=1
        res = self._attention_mixer(stacked_encodings, training=training)
        # Squeeze n_outputs dimension: (B, T, 1, encoding_dim) -> (B, T, encoding_dim)
        res = tf.squeeze(res, axis=2)
        tf.debugging.assert_equal(tf.shape(res)[:-1], (B, T))
        return res
