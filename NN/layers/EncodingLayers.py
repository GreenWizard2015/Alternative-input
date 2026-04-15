"""Encoding layers for positional and coordinate encoding."""

from typing import Optional, Any, List, Union
import tensorflow as tf


class LearnablePositionalEncoding(tf.keras.layers.Layer):
    """Learnable positional encoding layer with multi-dimensional support.

    Generates learnable positional encodings based on specified input dimensions
    and concatenates with input along the channel axis.

    Attributes:
        _channels: Number of PE channels to generate
        _embeddings: Learnable position embeddings
        _activation: Activation function for PE
        _axis: Axes used to build embeddings (positive indices after normalization)
        _tile_axis: Axes that need to be tiled/broadcast
    """

    def __init__(
        self,
        channels: int = 32,
        activation: Optional[str] = None,
        axis: Union[int, List[int]] = -2,
        **kwargs: Any,
    ) -> None:
        """Initialize learnable positional encoding layer.

        Args:
            channels: Number of positional encoding channels (default: 32)
            activation: Optional activation function (default: None)
            axis: Axes used to build embeddings. Can be a single int or list of ints.
                Negative indices refer to dimensions from the end. For input (B, H, W, C),
                -2 uses width dimension (default: -2)
            **kwargs: Additional Keras layer arguments
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)
        self._channels = channels
        self._activation_string = activation  # Store for serialization
        self._activation = tf.keras.activations.get(activation)
        self._axis = [axis] if isinstance(axis, int) else axis

    def build(self, input_shape: tuple) -> None:
        """Build learnable embeddings based on input shape.

        Normalizes negative axis indices to positive indices and creates embeddings
        with dimensions corresponding to specified axes, plus channel dimension.

        Args:
            input_shape: Input tensor shape (rank-4 by default: batch, height, width, channels)
        """
        real_axis = list(range(len(input_shape)))
        self._axis = sorted([real_axis[axis] for axis in self._axis])
        assert not (
            real_axis[-1] in self._axis
        ), f"Last axis are for channels. Input shape: {input_shape}. Axis: {self._axis}"
        assert not (
            0 in self._axis
        ), f"First axis are for batch dim. Input shape: {input_shape}"
        self._tile_axis = sorted(
            [axis for axis in real_axis if axis not in self._axis]
        )[:-1]

        spatial_dims = [1] * len(input_shape)
        for axis in self._axis:
            spatial_dims[axis] = input_shape[axis]
        spatial_dims[-1] = self._channels

        self._embeddings = self.add_weight(
            name="spatial_embeddings",
            shape=spatial_dims,
            initializer=tf.keras.initializers.RandomUniform(),
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Apply positional encoding and concatenate with input.

        Expands embeddings to match all input dimensions via tiling, applies optional
        activation, then concatenates along the channel dimension.

        Args:
            x: Input feature map of shape (B, H, W, C) or (B, T, H, W, C) or any shape matching build().
            **kwargs: Additional call arguments (e.g., training for dropout layers).

        Returns:
            Concatenated tensor with expanded channel dimension from PE concatenation.
        """
        pe_tensor: tf.Tensor = self._embeddings
        if self._activation is not None:
            pe_tensor = self._activation(pe_tensor)

        x_shape = tf.shape(x)
        # Repeat to match full input shape
        for axis in self._tile_axis:
            pe_tensor = tf.repeat(pe_tensor, x_shape[axis], axis=axis)
        return tf.concat([x, pe_tensor], axis=-1)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Input tensor shape (tuple or tf.TensorShape)

        Returns:
            Output tensor shape with additional channels from positional encoding
        """
        # Concatenation happens along the last dimension, so we add _channels to the channel dimension
        if hasattr(input_shape, "as_list"):
            # tf.TensorShape
            input_list = input_shape.as_list()
        else:
            # tuple
            input_list = list(input_shape)

        return tf.TensorShape(input_list[:-1] + [input_list[-1] + self._channels])


class ConvPE(LearnablePositionalEncoding):
    """Convolutional positional encoding shortcut.

    Convenience alias for LearnablePositionalEncoding with spatial axes (H, W).
    Creates positional encodings based on height and width dimensions.
    """

    def __init__(
        self, channels: int = 32, activation: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize convolutional PE layer.

        Args:
            channels: Number of positional encoding channels (default: 32)
            activation: Optional activation function (default: None)
            **kwargs: Additional Keras layer arguments
        """
        # Remove 'axis' from kwargs if present (ConvPE always uses [-3, -2])
        kwargs.pop("axis", None)
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(
            channels=channels, activation=activation, axis=[-3, -2], **kwargs
        )
