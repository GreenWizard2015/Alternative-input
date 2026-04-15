"""Linear attention mixer layer for multi-output feature pooling.

This module implements a linear attention mechanism that pools variable-length
sequences into N fixed-size representations using attention-weighted aggregation.
Each output has independent attention weights. Supports multi-head attention for
attending to different representation subspaces within each output.

FEATURES:
---------
- Multi-output attention: Each of N outputs learns independent attention weights
- Optimized computation: Single Dense(N × n_heads) layer computes all weights efficiently
- Multi-head attention: Split features across multiple heads per output (n_heads = feature_dim // max_dim)
- Learns which positions/features are most important for each pooled representation
- Supports variable sequence lengths and feature dimensions
- Minimal parameters per output - O(feature_dim * n_heads) per output
- Automatic head calculation based on max_dim parameter

USAGE:
------
    from NN.layers.LinearAttentionMixer import LinearAttentionMixer

    # Single output (n_outputs=1, max_dim=16 default)
    layer = LinearAttentionMixer()
    encoded_features = tf.random.normal((32, 478, 64))
    pooled = layer(encoded_features)  # Shape: (32, 1, 64) - n_heads calculated via for loop

    # Multiple independent outputs with custom max_dim
    layer = LinearAttentionMixer(n_outputs=4, max_dim=16)
    encoded_features = tf.random.normal((32, 478, 64))
    pooled = layer(encoded_features)  # Shape: (32, 4, 64) - n_heads calculated via for loop

    # Each output has independent attention weights with smaller max_dim
    layer = LinearAttentionMixer(n_outputs=2, max_dim=8)
    encoded_features = tf.random.normal((32, 478, 64))
    pooled = layer(encoded_features)  # Shape: (32, 2, 64) - n_heads calculated via for loop
"""

from typing import Any
import tensorflow as tf
import tensorflow.keras.layers as L
from NN.Constants import STANDARD_NONNEGATIVE_ACTIVATION


class LinearAttentionMixer(tf.keras.layers.Layer):
    """Linear attention-weighted mixer layer for multi-output feature pooling.

    This layer computes independent attention weights for N outputs, each using weighted sum
    (pooling) of an input sequence. Supports multi-head attention within each output.

    Architecture (OPTIMIZED for N outputs):
        1. Reshape input to (batch, spatial, feature_dim)
        2. Split features into multi-head format: (batch, n_heads, spatial, head_dim)
        3. Compute all N×n_heads attention logits: Dense(N×n_heads) with activation
        4. Reshape logits: (batch, spatial, N, n_heads)
        5. For each output i:
           - Extract logits[:, :, i, :] (batch, spatial, n_heads)
           - Softmax normalize across spatial dimension
           - Apply attention weights and pool: sum across spatial → (batch, head_dim)
           - Reshape back to feature_dim
        6. Stack outputs: (batch, N, feature_dim)
        7. Reshape to original leading dimensions: (..., N, feature_dim)

    Example:
        >>> # Single output (n_outputs=1, max_dim=16 default)
        >>> layer = LinearAttentionMixer()
        >>> encoded = tf.random.normal((batch_size, 478, 64))
        >>> pooled = layer(encoded)  # Shape: (batch_size, 1, 64), n_heads calculated via for loop

        >>> # Multiple outputs with independent attention
        >>> layer = LinearAttentionMixer(n_outputs=4, max_dim=16)
        >>> encoded = tf.random.normal((batch_size, 478, 64))
        >>> pooled = layer(encoded)  # Shape: (batch_size, 4, 64), n_heads calculated via for loop

        >>> # 4D spatial input: (batch, height, width, feature_dim)
        >>> spatial = tf.random.normal((batch_size, 5, 6, 64))
        >>> layer = LinearAttentionMixer(n_outputs=4, max_dim=16)
        >>> pooled = layer(spatial)  # Shape: (batch_size, 5, 4, 64) - pools width, preserves height
    """

    def __init__(
        self,
        n_outputs: int = 1,
        max_dim: int = 16,
        activation: str = STANDARD_NONNEGATIVE_ACTIVATION,
        dropout_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize the linear attention pooling layer.

        Args:
            n_outputs: Number of independent output representations. Default is 1.
                      Each output learns independent attention weights.
            max_dim: Maximum dimension per head for feature splitting. Default is 16.
                     n_heads will be calculated as feature_dim // max_dim.
                     Feature dimension must be divisible by max_dim.
            activation: Activation function for attention weight computation.
                       Default is STANDARD_NONNEGATIVE_ACTIVATION. Can be any Keras activation.
            dropout_rate: Dropout rate for regularization. Default is 0.1.
            **kwargs: Additional keyword arguments passed to Layer.__init__

        Raises:
            ValueError: If n_outputs < 1 or max_dim < 1.
        """
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")
        if max_dim < 1:
            raise ValueError(f"max_dim must be >= 1, got {max_dim}")

        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)

        self.n_outputs = n_outputs
        self.max_dim = max_dim
        self.activation = activation
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        """Build the layer with input shape.

        Args:
            input_shape: Tuple with shape (..., spatial_dim, feature_dim)

        Raises:
            ValueError: If input rank < 2 or feature_dim not divisible by max_dim
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"Input shape must have at least 2 dimensions, got {len(input_shape)}"
            )

        # Feature dimension is the last dimension
        self.feature_dim = input_shape[-1]

        # Calculate n_heads using for loop to find maximum possible
        self.n_heads = 0
        for possible_dim in range(self.feature_dim, 1, -1):
            if self.feature_dim % possible_dim == 0:
                self.n_heads = self.feature_dim // possible_dim
                break

        # Validate that max_dim divides feature_dim evenly
        if self.feature_dim % self.n_heads != 0:
            raise ValueError(
                f"feature_dim ({self.feature_dim}) must be divisible by n_heads ({self.n_heads}). "
                f"Ensure input shape[-1] % n_heads == 0."
            )

        # Create scale factor in build()
        self.scale_factor = self.add_weight(
            name="attention_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
            dtype=tf.float32,
        )

        # Create sublayers directly
        self.attention_layer = L.Dense(
            units=self.n_outputs * self.n_heads,
            activation=self.activation,
            name="AttentionWeights",
        )
        self.dropout = L.Dropout(rate=self.dropout_rate, name="AttentionDropout")

        # Build the single attention layer
        # Input to attention_layer: (..., spatial_dim, feature_dim)
        # Output: (..., spatial_dim, n_outputs * n_heads)
        self.attention_layer.build(input_shape)

        super().build(input_shape)

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Generate N independent attention-pooled outputs.

        Pools along the second-to-last axis for each of N outputs independently,
        preserving all other dimensions.

        Args:
            features: Tensor of shape (..., spatial_dim, feature_dim) where spatial_dim
                     is the axis to pool over. Supports:
                     - 3D: (batch, sequence_length, feature_dim) → (batch, n_outputs, feature_dim)
                     - 4D: (batch, height, width, feature_dim) → (batch, height, n_outputs, feature_dim)
                     - 5D: (batch, d1, d2, d3, feature_dim) → (batch, d1, d2, n_outputs, feature_dim)
            training: Whether in training mode (passed to sublayers)

        Returns:
            Tensor of shape (..., n_outputs, feature_dim) with spatial_dim removed,
            N independent outputs stacked in the new dimension.

        Example:
            >>> # 3D sequence input: generate 2 independent pooled outputs
            >>> encoded = tf.random.normal((4, 478, 64))
            >>> layer = LinearAttentionMixer(n_outputs=2, max_dim=16)
            >>> pooled = layer(encoded)  # Output shape: (4, 2, 64), n_heads calculated via for loop

            >>> # 4D spatial input: pool width, preserve height, generate 3 outputs
            >>> spatial = tf.random.normal((4, 5, 6, 64))
            >>> layer = LinearAttentionMixer(n_outputs=3, max_dim=16)
            >>> pooled = layer(spatial)  # Output shape: (4, 5, 3, 64), n_heads calculated via for loop

        Note:
            Dropout is applied to combined attention logits (all N outputs)
            before per-output softmax, creating correlated regularization across outputs.
        """
        # Get original shape and dimensions
        input_shape = tf.shape(features)

        # Extract leading dims, spatial dim, and feature dim
        leading_dims = input_shape[:-2]  # All dims except spatial and feature
        spatial_dim = input_shape[-2]
        feature_dim = input_shape[-1]

        # Compute flattened batch size
        batch_size = tf.reduce_prod(leading_dims)

        # Reshape to 3D: (batch, spatial, feature)
        features_3d = tf.reshape(features, [batch_size, spatial_dim, feature_dim])

        # Split features into heads: (batch, spatial, n_heads, head_dim)
        features_reshaped = tf.reshape(
            features_3d, [batch_size, spatial_dim, self.n_heads, -1]
        )
        # Transpose to (batch, n_heads, spatial, head_dim)
        features_heads = tf.transpose(features_reshaped, [0, 2, 1, 3])

        # Compute all attention logits at once: (batch, spatial, n_outputs*n_heads)
        attn_logits_all = self.attention_layer(features_3d, training=training)
        attn_logits_all = self.dropout(attn_logits_all, training=training)

        # Reshape to separate outputs: (batch, spatial, n_outputs, n_heads)
        attn_logits_reshaped = tf.reshape(
            attn_logits_all,
            [batch_size, spatial_dim, self.n_outputs, self.n_heads],
        )

        # Fully vectorized processing: eliminate Python loop entirely
        # Apply learnable scale factor to attention logits (like MultiHeadAttention)
        scaled_attn_logits = attn_logits_reshaped * self.scale_factor

        # Apply softmax across spatial dimension for all outputs simultaneously
        # attn_logits_reshaped: (batch, spatial, n_outputs, n_heads)
        # attn_weights_all: (batch, spatial, n_outputs, n_heads)
        attn_weights_all = tf.nn.softmax(scaled_attn_logits, axis=1)

        # Reshape features for vectorized operations
        # features_heads: (batch, n_heads, spatial, head_dim)
        # Transpose to: (batch, spatial, n_heads, head_dim)
        features_spatial = tf.transpose(features_heads, [0, 2, 1, 3])

        # Reshape attention weights for broadcasting
        # attn_weights_all: (batch, spatial, n_outputs, n_heads)
        # Reshape to: (batch, spatial, n_outputs * n_heads, 1)
        attn_weights_flat = tf.reshape(
            attn_weights_all,
            [batch_size, spatial_dim, self.n_outputs * self.n_heads, 1],
        )

        # Calculate head_dim and reshape features for batched operations
        head_dim = self.feature_dim // self.n_heads
        features_flat = tf.tile(
            tf.reshape(
                features_spatial, [batch_size, spatial_dim, self.n_heads, head_dim]
            ),
            [1, 1, self.n_outputs, 1],
        )

        # Apply attention in fully vectorized way
        # (batch, spatial, n_outputs*n_heads, head_dim) * (batch, spatial, n_outputs*n_heads, 1)
        weighted_features = features_flat * attn_weights_flat

        # Pool across spatial: (batch, n_outputs*n_heads, head_dim)
        pooled_flat = tf.reduce_sum(weighted_features, axis=1)

        # Calculate head_dim and reshape to separate outputs and heads: (batch, n_outputs, n_heads, head_dim)
        head_dim = self.feature_dim // self.n_heads
        pooled_4d = tf.reshape(
            pooled_flat, [batch_size, self.n_outputs, self.n_heads, head_dim]
        )

        # Pool across heads for each output: (batch, n_outputs, feature_dim)
        pooled_3d = tf.reshape(
            pooled_4d, [batch_size, self.n_outputs, self.feature_dim]
        )

        # Reshape to original leading dims: (..., n_outputs, feature_dim)
        output_shape = tf.concat([leading_dims, [self.n_outputs, feature_dim]], axis=0)
        pooled = tf.reshape(pooled_3d, output_shape)

        return pooled
