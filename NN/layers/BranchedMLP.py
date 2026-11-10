from numbers import Real
from typing import Any, List, Optional, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow.keras.layers as L

from NN.Constants import SMLP_GLOBAL_DROPOUT


FractionArg = Union[float, Sequence[float]]


class BranchedDense(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        branches: int,
        activation: Any = "linear",
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)

        if units < 1:
            raise ValueError("units must be >= 1.")
        if branches < 1:
            raise ValueError("branches must be >= 1.")

        self.units = int(units)
        self.branches = int(branches)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = bool(use_bias)
        self._input_dim = None
        self._kernel = None
        self._bias = 0.0

    def build(self, input_shape: Tuple) -> None:
        shape = tf.TensorShape(input_shape)
        if shape.rank != 3:
            raise ValueError("Dense input must have shape [branches, batch, features].")
        if shape[-1] is None:
            raise ValueError("Input feature dimension must be statically known.")
        if shape[0] is not None and int(shape[0]) != self.branches:
            raise ValueError("Input branch dimension does not match branches.")

        self._input_dim = int(shape[-1])
        limit = (6.0 / float(self._input_dim + self.units)) ** 0.5
        self._kernel = self.add_weight(
            name="kernel",
            shape=[self.branches, self._input_dim, self.units],
            initializer=tf.keras.initializers.RandomUniform(
                minval=-limit, maxval=limit
            ),
            trainable=True,
        )
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=[self.branches, self.units],
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, x: tf.Tensor, output_units: Optional[int] = None) -> tf.Tensor:
        tf.debugging.assert_rank(x, 3)
        tf.debugging.assert_equal(tf.shape(x)[0], self.branches)
        tf.debugging.assert_less_equal(tf.shape(x)[-1], self._input_dim)

        output_units = self.units if output_units is None else output_units
        kernel = self._kernel[:, : tf.shape(x)[-1], :output_units]
        x = tf.matmul(x, kernel)
        x = x + self._bias[:, None, :output_units] if self.use_bias else x
        return self.activation(x)


class BranchedMLP(tf.keras.layers.Layer):
    """Branched prefix MLP with scalar or static-list fractions.

    Single fraction -> original shape [..., branches, output_dim], with the
    branches axis removed when branches == 1.
    Fraction list -> [F, ... , branches, output_dim], likewise removing the
    branches axis when branches == 1. F follows the input list's order.

    Each fraction uses prefixes of its input and hidden widths, but ALWAYS
    uses the full final output width, exactly as the original implementation.

    The list must be a Python sequence (not a symbolic tf.Tensor) to keep
    layer widths and XLA tracing static.
    """

    def __init__(
        self,
        sizes: List[int],
        branches: int = 1,
        activation: Any = "linear",
        dropout: Optional[float] = None,
        last_activation: Optional[Any] = None,
        use_bias: bool = True,
        use_last_bias: bool = True,
        branched_input: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)

        if not sizes or any(size < 1 for size in sizes):
            raise ValueError("sizes must contain positive output dimensions.")
        if branches < 1:
            raise ValueError("branches must be >= 1.")

        self.sizes = [int(size) for size in sizes]
        self.branches = int(branches)
        self.activation = activation
        self.dropout = SMLP_GLOBAL_DROPOUT if dropout is None else float(dropout)
        self.last_activation = (
            activation if last_activation is None else last_activation
        )
        self.use_bias = bool(use_bias)
        self.use_last_bias = bool(use_last_bias)
        self.branched_input = bool(branched_input)
        self._input_dim = None
        self._dropout_layers = []
        self._dense_layers = []

        for i, size in enumerate(self.sizes):
            is_last = i == len(self.sizes) - 1
            self._dropout_layers.append(
                L.Dropout(self.dropout, name=f"dropout-{i}")
                if self.dropout > 0.0
                else None
            )
            self._dense_layers.append(
                BranchedDense(
                    units=size,
                    branches=self.branches,
                    activation=self.last_activation if is_last else self.activation,
                    use_bias=self.use_last_bias if is_last else self.use_bias,
                    name=f"dense-{i}",
                )
            )

    def build(self, input_shape: Tuple) -> None:
        shape = tf.TensorShape(input_shape)
        if shape.rank is None or shape.rank < 2:
            raise ValueError("Input rank must be >= 2.")
        if shape[-1] is None:
            raise ValueError("Input feature dimension must be statically known.")
        if self.branched_input and (
            shape.rank < 3
            or (shape[-2] is not None and int(shape[-2]) != self.branches)
        ):
            raise ValueError("Input branch dimension does not match branches.")

        self._input_dim = int(shape[-1])
        self._dense_layers[0].build(
            tf.TensorShape([self.branches, None, self._input_dim])
        )
        for i in range(1, len(self._dense_layers)):
            self._dense_layers[i].build(
                tf.TensorShape([self.branches, None, self.sizes[i - 1]])
            )
        super().build(input_shape)

    @staticmethod
    def _fraction_size(size: int, fraction: float) -> int:
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction values must be in (0, 1].")
        return max(1, min(size, round(size * fraction)))

    def _dropout(self, x: tf.Tensor, index: int, training: Optional[bool]) -> tf.Tensor:
        layer = self._dropout_layers[index]
        return x if layer is None else layer(x, training=training)

    def _to_branch_major(
        self, x: tf.Tensor, input_dim: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        tf.debugging.assert_greater_equal(tf.shape(x)[-1], input_dim)
        tf.debugging.assert_less_equal(tf.shape(x)[-1], self._input_dim)
        feature_dim = tf.shape(x)[-1]

        if self.branched_input:
            tf.debugging.assert_equal(tf.shape(x)[-2], self.branches)
            leading_shape = tf.shape(x)[:-2]
            x = tf.reshape(x, [-1, self.branches, feature_dim])
            return tf.transpose(x[..., :input_dim], [1, 0, 2]), leading_shape

        leading_shape = tf.shape(x)[:-1]
        x = tf.reshape(x, [-1, feature_dim])
        return (
            tf.broadcast_to(
                x[None, :, :input_dim], [self.branches, tf.shape(x)[0], input_dim]
            ),
            leading_shape,
        )

    def _from_branch_major(self, x: tf.Tensor, leading_shape: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_rank(x, 3)
        tf.debugging.assert_equal(tf.shape(x)[0], self.branches)
        output_units = tf.shape(x)[-1]
        if self.branches == 1:
            return tf.reshape(x[0], tf.concat([leading_shape, [output_units]], axis=0))
        x = tf.transpose(x, [1, 0, 2])
        return tf.reshape(
            x, tf.concat([leading_shape, [self.branches, output_units]], axis=0)
        )

    def _widths(self, fraction: float) -> Tuple[int, ...]:
        # The last layer is always full-width; for a single layer this is
        # intentional and matches the original BranchedMLP semantics.
        return (
            self._fraction_size(self._input_dim, fraction),
            *(self._fraction_size(s, fraction) for s in self.sizes[:-1]),
            self.sizes[-1],
        )

    def _call_single(
        self, x: tf.Tensor, training: Optional[bool], fraction: float
    ) -> tf.Tensor:
        widths = self._widths(fraction)
        x = self._dropout(x, 0, training)
        x, leading_shape = self._to_branch_major(x, widths[0])
        for i, dense in enumerate(self._dense_layers):
            if i:
                x = self._dropout(x, i, training)
            x = dense(x, output_units=widths[i + 1])
        return self._from_branch_major(x, leading_shape)

    def _call_many_vectorized(
        self,
        x: tf.Tensor,
        training: Optional[bool],
        fractions: Tuple[float, ...],
        aligned_fraction_inputs: bool = False,
    ) -> tf.Tensor:
        """One batched matmul per dense layer for all fractions.

        Masks before/after each layer reproduce prefix slicing, including for
        activations with nonzero f(0) and for nonzero bias terms. The fraction
        dimension is folded into the batch dimension; kernels are shared.
        """
        widths = tuple(self._widths(fraction) for fraction in fractions)
        count = len(fractions)
        input_dims = [w[0] for w in widths]
        max_input = max(input_dims)

        # Optional aligned inputs: x[f] supplies the feature vector for fraction[f].
        # Otherwise preserve the original shared-input broadcasting behavior.
        if aligned_fraction_inputs:
            tf.debugging.assert_equal(tf.shape(x)[0], count)
        else:
            x = tf.broadcast_to(x[None, ...], tf.concat([[count], tf.shape(x)], axis=0))
        x = self._dropout(x, 0, training)
        x, leading_shape = self._to_branch_major(x, max_input)
        batch = tf.shape(x)[1] // count
        x = tf.reshape(x, [self.branches, count, batch, max_input])
        x = x * tf.sequence_mask(input_dims, max_input, dtype=x.dtype)[None, :, None, :]

        for i, dense in enumerate(self._dense_layers):
            if i:
                x = self._dropout(x, i, training)
            output_dims = [w[i + 1] for w in widths]
            max_output = max(output_dims)
            x = tf.reshape(x, [self.branches, count * batch, tf.shape(x)[-1]])
            x = dense(x, output_units=max_output)
            x = tf.reshape(x, [self.branches, count, batch, max_output])
            if i < len(self._dense_layers) - 1:
                x = (
                    x
                    * tf.sequence_mask(output_dims, max_output, dtype=x.dtype)[
                        None, :, None, :
                    ]
                )

        x = tf.reshape(x, [self.branches, count * batch, self.sizes[-1]])
        return self._from_branch_major(x, leading_shape)

    def _fraction_strategy(self, fractions: Tuple[float, ...]) -> str:
        """Conservative estimate: pad only when wasted matmul FLOPs are modest."""
        widths = tuple(self._widths(f) for f in fractions)
        sliced = sum(sum(a * b for a, b in zip(w[:-1], w[1:])) for w in widths)
        padded = len(widths) * sum(
            max(w[i] for w in widths) * max(w[i + 1] for w in widths)
            for i in range(len(self.sizes))
        )
        return "vectorized" if padded <= 1.75 * sliced else "sliced"

    def call(
        self,
        x: tf.Tensor,
        training: Optional[bool] = None,
        fraction: FractionArg = 1.0,
        fractions: Optional[Sequence[float]] = None,
        fraction_strategy: str = "auto",
        aligned_fraction_inputs: bool = False,
        **kwargs: Any,
    ) -> tf.Tensor:
        if fractions is not None:
            if not isinstance(fraction, Real) or float(fraction) != 1.0:
                raise ValueError("Use either fraction or fractions, not both.")
            fraction = fractions
        if isinstance(fraction, Real):
            return self._call_single(x, training, float(fraction))

        if isinstance(fraction, (str, bytes)):
            raise TypeError(
                "fraction must be a number or a Python sequence of numbers."
            )
        try:
            fractions = tuple(float(f) for f in fraction)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "fraction must be a number or a Python sequence of numbers."
            ) from exc
        if not fractions:
            raise ValueError("fraction sequence must not be empty.")
        for f in fractions:
            self._fraction_size(1, f)

        if fraction_strategy not in ("auto", "vectorized", "sliced"):
            raise ValueError(
                "fraction_strategy must be 'auto', 'vectorized' or 'sliced'."
            )
        if aligned_fraction_inputs:
            tf.debugging.assert_equal(tf.shape(x)[0], len(fractions))
        if len(fractions) == 1:
            return self._call_single(
                x[0] if aligned_fraction_inputs else x, training, fractions[0]
            )[None, ...]
        strategy = (
            self._fraction_strategy(fractions)
            if fraction_strategy == "auto"
            else fraction_strategy
        )
        if strategy == "vectorized":
            return self._call_many_vectorized(
                x, training, fractions, aligned_fraction_inputs
            )
        return tf.stack(
            [
                self._call_single(x[i] if aligned_fraction_inputs else x, training, f)
                for i, f in enumerate(fractions)
            ],
            axis=0,
        )

    def compute_output_shape(
        self,
        input_shape: Tuple,
        fraction: FractionArg = 1.0,
        fractions: Optional[Sequence[float]] = None,
        aligned_fraction_inputs: bool = False,
    ) -> tf.TensorShape:
        shape = tf.TensorShape(input_shape)
        shape = (
            shape[:-1].concatenate([self.sizes[-1]])
            if self.branched_input
            else shape[:-1].concatenate([self.branches, self.sizes[-1]])
        )
        shape = shape[:-2].concatenate([shape[-1]]) if self.branches == 1 else shape
        if fractions is not None:
            fraction = fractions
        if isinstance(fraction, Real):
            return shape
        return (
            shape
            if aligned_fraction_inputs
            else tf.TensorShape([len(fraction)]).concatenate(shape)
        )
