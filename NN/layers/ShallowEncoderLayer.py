"""Shallow temporal encoding layer with learnable coordinate-based time encoding."""

from typing import Any
import tensorflow as tf
from NN.layers.RolloutTimesteps import RolloutTimesteps
from NN.layers.CoordsEncodingLayer import CoordsEncodingLayer


class ShallowEncoderLayer(tf.keras.layers.Layer):
    """Encodes temporal information using coordinate encoding and rollout timesteps.

    This shallow layer takes timestep values and encodes them using a coordinate encoding
    scheme, then rolls them out across the sequence dimension.

    Attributes:
        _encoder: RolloutTimesteps wrapper around CoordsEncodingLayer
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the time encoder layer.

        Args:
            **kwargs: Additional keyword arguments passed to parent Layer class.
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)
        self._encoder = RolloutTimesteps(
            lambda: CoordsEncodingLayer(N=32, raw=False, name="CoordsEncoding"),
            name="Encoder",
        )

    def build(self, input_shape):
        """Build the layer.

        Args:
            input_shape: Input shape tuple (batch, time_steps, features).
        """
        assert 1 == input_shape[-1]
        assert 3 == len(input_shape)
        batch_size = input_shape[0]
        timesteps_count = input_shape[1]
        last = input_shape[-1]

        dummy_input_shape = (batch_size, timesteps_count, 1, last)
        self._encoder.build(dummy_input_shape)
        super().build(input_shape)

    def call(self, data: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Encode temporal input.

        Args:
            data: Input tensor of shape (batch, time_steps, 1) containing time values.
            training: Boolean indicating training or inference mode.

        Returns:
            Encoded tensor of shape (batch, time_steps, encoding_dim).
        """
        batch_size = tf.shape(data)[0]
        timesteps_count = tf.shape(data)[1]
        last = tf.shape(data)[-1]
        # timesteps is (batch, timesteps, 1) - expand dims is handled by encoder
        # Add a dimension for the coordinates
        expanded = tf.expand_dims(data, axis=-2)
        tf.debugging.assert_equal(
            tf.shape(expanded), (batch_size, timesteps_count, 1, last)
        )

        # Process through rollout encoder
        encoded_timesteps = self._encoder(expanded, training=training)
        encoded_timesteps = tf.reshape(
            encoded_timesteps, [batch_size, timesteps_count, -1]
        )

        return encoded_timesteps

    @property
    def output_shape(self):
        """Get the output shape of the encoder layer.

        Returns:
            Output shape tuple (batch, timesteps, encoding_dim) where encoding_dim
            is the output dimension of the internal CoordsEncodingLayer.
        """
        return tf.TensorShape([None, None, 32])
