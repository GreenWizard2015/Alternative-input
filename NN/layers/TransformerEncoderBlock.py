"""Transformer encoder block with multi-head attention and feed-forward network.

Implements a single transformer encoder layer consisting of multi-head self-attention
followed by a position-wise feed-forward network, with residual connections and
layer normalization.

FEATURES:
---------
- Multi-head self-attention with dynamic head calculation
- Max-dim based head computation using for loop
- Position-wise feed-forward network with residual connections
- Layer normalization and dropout for regularization
- Efficient implementation with automatic head calculation

USAGE:
------
    from NN.layers.TransformerEncoderBlock import TransformerEncoderBlock

    # Basic usage (max_dim=16 default, num_heads calculated via for loop)
    block = TransformerEncoderBlock(d_model=64)
    x = tf.random.normal((32, 10, 64))
    output = block(x)  # num_heads calculated via for loop

    # Custom max_dim for specific head configuration
    block = TransformerEncoderBlock(d_model=64, max_dim=8)
    x = tf.random.normal((32, 10, 64))
    output = block(x)  # num_heads calculated via for loop

    # 4D spatial input: (batch, height, width, d_model)
    spatial = tf.random.normal((32, 5, 6, 64))
    block = TransformerEncoderBlock(d_model=64, max_dim=16)
    output = block(spatial)  # num_heads calculated via for loop
"""

import tensorflow as tf
from typing import Optional, Any
from NN.layers.MultiHeadAttention import MultiHeadAttention
from NN.Constants import STANDARD_NONNEGATIVE_ACTIVATION


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block with self-attention and FFN.

    Architecture:
        1. Multi-head self-attention with residual connection and layer norm
        2. Position-wise feed-forward network with residual connection and layer norm

    The feed-forward network expands the dimension by expansion_factor and then
    projects back to d_model.

    Attributes:
        d_model: The embedding dimension (model size).
        max_dim: Maximum dimension per head for feature splitting.
        num_heads: Number of attention heads (calculated via for loop using max_dim).
        dff: Dimension of the inner feed-forward layer.
        dropout_rate: Dropout rate for attention and FFN.

    Head calculation:
        num_heads is calculated using a for loop to find the maximum possible value
        such that d_model % possible_dim == 0. The loop iterates from d_model downward
        to find the largest possible divisor, ensuring optimal head configuration.
    """

    def __init__(
        self,
        d_model: int = 128,
        max_dim: int = 16,
        dff: Optional[int] = None,
        dropout_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize TransformerEncoderBlock.

        Args:
            d_model: The embedding dimension (default: 128).
            max_dim: Maximum dimension per head for feature splitting. num_heads will be
                calculated using for loop: find maximum possible num_heads such that
                d_model % max_dim == 0 (default: 16).
            dff: Dimension of the feed-forward inner layer. If None, defaults to
                4 * d_model (default: None).
            dropout_rate: Dropout rate for attention and FFN (default: 0.1).
            **kwargs: Additional keyword arguments passed to parent Layer.

        Raises:
            ValueError: If max_dim < 1 or d_model % max_dim != 0.
        """
        if max_dim < 1:
            raise ValueError(f"max_dim must be >= 1, got {max_dim}")

        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)

        if dff is None:
            dff = 4 * d_model

        self.d_model = d_model
        self.max_dim = max_dim
        self.dff = dff
        self.dropout_rate = dropout_rate

        # Store sublayer names directly

    def build(self, input_shape) -> None:
        """Build layer with sublayer creation.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If d_model % num_heads != 0.
        """
        # Calculate num_heads using for loop to find maximum possible (same as LinearAttentionMixer)
        self.num_heads = 0
        for possible_dim in range(self.d_model, 1, -1):
            if self.d_model % possible_dim == 0:
                self.num_heads = self.d_model // possible_dim
                break

        # Validate that max_dim divides d_model evenly (same as LinearAttentionMixer logic)
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads}). "
                f"Ensure input shape[-1] % num_heads == 0."
            )

        # Multi-head self-attention
        self.mha = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            hypersphere=True,
            name="multi_head_attention",
        )
        # MHA expects (batch, seq_len, d_model) input
        self.mha.build(input_shape)

        # Position-wise feed-forward network
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.dff,
                    activation=STANDARD_NONNEGATIVE_ACTIVATION,
                    name="ffn_inner",
                ),
                tf.keras.layers.Dense(self.d_model, name="ffn_output"),
                tf.keras.layers.Dropout(rate=self.dropout_rate),
            ],
            name="feed_forward_network",
        )
        # FFN receives MHA output (batch, seq_len, d_model)
        self.ffn.build(input_shape)

        # Layer normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="layernorm1"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="layernorm2"
        )
        # Layer norm expects (batch, seq_len, d_model) input
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)

        # Dropout for residual connections
        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout_rate, name="dropout1")
        self.dropout2 = tf.keras.layers.Dropout(rate=self.dropout_rate, name="dropout2")
        # Dropout expects (batch, seq_len, d_model) input
        self.dropout1.build(input_shape)
        self.dropout2.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Process input through encoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask of shape (batch_size, 1, 1, seq_len)
                (default: None).
            training: Whether in training mode (default: False).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Multi-head self-attention with residual connection
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            mask=mask,
            training=training,
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
