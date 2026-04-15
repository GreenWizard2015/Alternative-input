"""Layer that applies a function to rolled-out timesteps.

Provides a layer for processing temporal sequences by applying a function
per-timestep, useful for reducing computation when processing time-distributed data.
"""

from typing import Any, Dict, List, Tuple, Union, Callable

import tensorflow as tf


class RolloutTimesteps(tf.keras.layers.Layer):
    """Applies a function to rolled-out timesteps.

    This layer takes temporal sequences and applies a function per-timestep,
    then reshapes back to temporal format. Supports tensors, lists, and dicts.

    Attributes:
        _F: Function/layer to apply to each timestep
    """

    def __init__(self, F: Callable, **kwargs: Any) -> None:
        """Initialize rollout layer.

        Args:
            F: Function that returns a Keras layer. Must be a callable that when
                called returns a layer instance (e.g., lambda: tf.keras.layers.Dense(256)).
            **kwargs: Additional Keras layer arguments (name, trainable, etc).

        Example:
            >>> rollout = RolloutTimesteps(lambda: tf.keras.layers.Dense(256))
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)
        # F must be a function that returns a layer
        if not callable(F):
            raise ValueError("F must be a callable that returns a layer")

        # Call the function to get the actual layer
        self._F = F

    def build(self, input_shape):
        """Build the layer.

        Args:
            input_shape: Input shape tuple (batch, time_steps, features).
        """
        self._F = self._F()

        reshaped_input_shape = None
        # Handle different input shape formats
        if isinstance(input_shape, list):
            reshaped_input_shape = list(map(lambda shp: (None, *shp[2:]), input_shape))
        if isinstance(input_shape, tuple):
            reshaped_input_shape = (None, *input_shape[2:])

        self._F.build(reshaped_input_shape)
        super().build(input_shape)

    def _reshape_all(
        self,
        input_data: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]],
        target_shape_prefix: Tuple,
        preserve_axis: int,
    ) -> Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]:
        """Reshape single tensor, list of tensors, or dict of tensors.

        Preserves the input structure while reshaping. Lists remain lists,
        dicts remain dicts, tensors remain tensors.

        Args:
            input_data: Input data - either a tensor, list of tensors, or dict mapping
                strings to tensors.
            target_shape_prefix: Prefix shape tuple to prepend to each tensor.
            preserve_axis: Axis from which to preserve original shape dimensions.

        Returns:
            Reshaped data with identical structure as input.

        Example:
            >>> layer = RolloutTimesteps(...)
            >>> tensor = tf.random.normal((16, 100, 256))
            >>> reshaped = layer._reshape_all(tensor, (1600,), preserve_axis=2)
            >>> assert reshaped.shape == (1600, 256)
        """
        prefix_list = list(target_shape_prefix)
        # Convert prefix to tensor form
        prefix_tensor = tf.stack(prefix_list)

        def compute_new_shape(tensor: tf.Tensor) -> tf.Tensor:
            """Compute new shape by concatenating prefix with remaining dimensions."""
            tensor_shape = tf.shape(tensor)
            remaining_shape = tf.stack(tf.unstack(tensor_shape)[preserve_axis:])
            new_shape = tf.concat([prefix_tensor, remaining_shape], axis=-1)
            return new_shape

        if isinstance(input_data, tf.Tensor):
            return tf.reshape(input_data, compute_new_shape(input_data))

        if isinstance(input_data, list):
            return [tf.reshape(v, compute_new_shape(v)) for v in input_data]

        if isinstance(input_data, dict):
            return {
                k: tf.reshape(v, compute_new_shape(v)) for k, v in input_data.items()
            }

        return input_data

    def call(
        self, x: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], **kwargs: Any
    ) -> Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]:
        """Apply function to rolled-out timesteps.

        Reshapes temporal sequences by collapsing batch and timesteps dimensions,
        applies the pre-initialized layer function, then reshapes back to temporal format.

        Args:
            x: Input with shape (batch, timesteps, ...) - can be tensor, list of
                tensors, or dict of tensors.
            **kwargs: Additional arguments to pass to the internal layer.

        Returns:
            Output with same shape (batch, timesteps, ...) and structure as input.

        Example:
            >>> layer = RolloutTimesteps(lambda: tf.keras.layers.Dense(256))
            >>> x = tf.random.normal((32, 10, 64))  # batch=32, timesteps=10, features=64
            >>> out = layer.call(x)
            >>> assert out.shape == (32, 10, 256)
        """
        # Extract reference tensor to determine batch and timestep dimensions
        reference_tensor: tf.Tensor
        if isinstance(x, tf.Tensor):
            reference_tensor = x
        elif isinstance(x, list):
            reference_tensor = x[0]
        elif isinstance(x, dict):
            reference_tensor = x[next(iter(x.keys()))]

        batch_size = tf.shape(reference_tensor)[0]
        timesteps_count: Union[int, tf.Tensor] = tf.shape(reference_tensor)[1]
        if timesteps_count is None:
            # Dynamic shape - get from tf.shape
            timesteps_count = reference_tensor.shape[1]

        # Reshape to (batch * timesteps, ...)
        flattened_input = self._reshape_all(
            x, (batch_size * timesteps_count,), preserve_axis=2
        )
        rollout_result = self._F(flattened_input, **kwargs)

        # Reshape back to (batch, timesteps, ...)
        return self._reshape_all(
            rollout_result, (batch_size, timesteps_count), preserve_axis=1
        )
