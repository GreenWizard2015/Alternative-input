"""Simple Multi-Layer Perceptron layer with dropout."""

from typing import Any, List, Optional, Tuple

import tensorflow as tf
import tensorflow.keras.layers as L
from NN.Constants import SMLP_GLOBAL_DROPOUT


class sMLP(tf.keras.layers.Layer):
    """Simple Multi-Layer Perceptron layer with dropout.

    Stacks dense layers with optional dropout in between.

    Attributes:
        _F: Sequential model containing the MLP layers
    """

    def __init__(
        self,
        sizes: List[int],
        activation: str = "linear",
        dropout: Optional[float] = None,
        use_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize MLP layer.

        Args:
            sizes: List of layer sizes (neurons per layer)
            activation: Activation function name (default: "linear")
            dropout: Dropout rate (uses global default if None)
            use_norm: Whether to use LayerNormalization (default: False)
            **kwargs: Additional Keras layer arguments
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)
        self.sizes = sizes
        self.activation = activation
        self.use_norm = use_norm
        dropout = SMLP_GLOBAL_DROPOUT if dropout is None else dropout
        self.dropout = dropout

    def build(self, input_shape: Tuple) -> None:
        """Build the layer.

        Args:
            input_shape: Input tensor shape
        """
        layers: List[L.Layer] = []
        for i, sz in enumerate(self.sizes):
            if self.dropout > 0.0:
                layers.append(L.Dropout(self.dropout, name=f"dropout-{i}"))
            layers.append(L.Dense(sz, activation=self.activation, name=f"dense-{i}"))
            if self.use_norm:
                layers.append(L.LayerNormalization(name=f"norm-{i}"))

        self._F = tf.keras.Sequential(layers, name="_F")
        self._F.build(input_shape)
        return super().build(input_shape)

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Forward pass.

        Args:
            x: Input tensor
            **kwargs: Additional call arguments

        Returns:
            Output from MLP
        """
        return self._F(x, **kwargs)

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """Compute the output shape of the sMLP layer.

        Args:
            input_shape: Input tensor shape

        Returns:
            Output shape as tuple. The last dimension will be the size of the final dense layer.
        """
        return self._F.compute_output_shape(input_shape)
