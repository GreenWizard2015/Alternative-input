"""Multi-head attention layer implementation.

Implements scaled dot-product attention with multiple attention heads,
allowing the model to attend to information from different representation
subspaces at different positions.
"""

import tensorflow as tf
from typing import Optional, Tuple, Any, Union
from NN.Constants import ATTENTION_MASK_VALUE


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head scaled dot-product attention layer with learnable scale factor.

    Splits the input into multiple heads, applies scaled dot-product attention
    independently on each head, and concatenates the results. This allows the
    model to attend to information from different representation subspaces.

    The scale factor is learnable and wrapped with softplus for numerical stability,
    allowing the model to adaptively adjust attention scaling during training.

    Attributes:
        d_model: The embedding dimension (model size).
        num_heads: Number of attention heads.
        d_k: Dimension of key/query vectors per head (d_model // num_heads).
        d_v: Dimension of value vectors per head (d_model // num_heads).
        scale_factor: Learnable scale factor (wrapped with softplus).
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        hypersphere: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize MultiHeadAttention layer.

        Args:
            d_model: The embedding dimension (default: 512).
            num_heads: Number of attention heads (default: 8).
                d_model must be divisible by num_heads.
            dropout_rate: Dropout rate for attention weights (default: 0.1).
            hypersphere: If True, normalize query and key vectors to unit norm
                (default: False).
            **kwargs: Additional keyword arguments passed to parent Layer.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.hypersphere = hypersphere
        self.dropout_rate = dropout_rate

        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        # Store configuration for build() method
        self._d_model = d_model
        self._dropout_rate = dropout_rate

    def build(self, input_shape):
        """Build layer by creating and configuring sublayers.

        Args:
            input_shape: Shape of input (not used, but required by Keras API).
                         For multi-head attention, typically a tuple of
                         (query_shape, key_shape, value_shape).
        """
        # Linear transformations for Q, K, V
        self.W_q = tf.keras.layers.Dense(self._d_model, name="query_projection")
        self.W_k = tf.keras.layers.Dense(self._d_model, name="key_projection")
        self.W_v = tf.keras.layers.Dense(self._d_model, name="value_projection")

        # Output projection
        self.W_o = tf.keras.layers.Dense(self._d_model, name="output_projection")

        # Dropout for attention weights
        self.dropout = tf.keras.layers.Dropout(
            rate=self._dropout_rate, name="attention_dropout"
        )

        # Learnable scale factor for attention
        # Learnable scale factor for attention (initialized so softplus ≈ 1/sqrt(d_k))
        # softplus(x) = log(1 + exp(x)), so we solve: softplus(init_val) ≈ 1/sqrt(d_k)
        target_scale = 1.0 / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        init_val = tf.math.log(tf.exp(target_scale) - 1.0).numpy()
        self.scale_factor = self.add_weight(
            name="attention_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(init_val),
            trainable=self.hypersphere,
            dtype=tf.float32,
        )

        # Build all Dense layers with standard input shape (batch, seq_len, d_model)
        # The actual batch and sequence lengths don't matter for Dense layers
        dense_input_shape = (None, None, self._d_model)

        self.W_q.build(dense_input_shape)
        self.W_k.build(dense_input_shape)
        self.W_v.build(dense_input_shape)
        self.W_o.build(dense_input_shape)

        # Also build the dropout layer
        self.dropout.build(dense_input_shape)

        # Mark as built
        super().build(input_shape)

    def _split_heads(
        self, x: tf.Tensor, batch_size: Union[int, tf.Tensor]
    ) -> tf.Tensor:
        """Split the last dimension into (num_heads, d_k/d_v).

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            batch_size: Batch size (int or Tensor).

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k/d_v).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _scaled_dot_product_attention(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute scaled dot-product attention with learnable scale factor.

        Applies scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T * scale)V
        where scale is a learnable parameter wrapped with softplus for numerical stability.
        Optionally normalizes query and key vectors to unit norm on hypersphere.

        Args:
            query: Query tensor of shape (batch_size, num_heads, q_len, d_k).
            key: Key tensor of shape (batch_size, num_heads, k_len, d_k).
            value: Value tensor of shape (batch_size, num_heads, v_len, d_v).
            mask: Optional mask tensor to apply to attention weights (default: None).
            training: Whether in training mode for dropout (default: False).

        Returns:
            Tuple of:
                - output: Attention output of shape (batch_size, num_heads, q_len, d_v).
                - attention_weights: Attention weights of shape
                  (batch_size, num_heads, q_len, k_len).
        """
        # Normalize query and key to unit norm if hypersphere is enabled
        if self.hypersphere:
            query = tf.nn.l2_normalize(query, axis=-1)
            key = tf.nn.l2_normalize(key, axis=-1)

        # Compute attention scores: QK^T * scale_factor
        # scale_factor is learnable and wrapped with softplus for stability (always positive)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        learnable_scale = tf.nn.softplus(self.scale_factor)
        scaled_attention_logits = matmul_qk * learnable_scale

        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += mask * ATTENTION_MASK_VALUE

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)

        # Multiply by values
        output = tf.matmul(attention_weights, value)

        return output, attention_weights

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Compute multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_v, d_model).
            mask: Optional attention mask of shape (batch_size, 1, seq_len_q, seq_len_k)
                (default: None).
            training: Whether in training mode (default: False).

        Returns:
            - output: Tensor of shape (batch_size, seq_len_q, d_model).
        """
        batch_size = tf.shape(query)[0]

        # Linear projections in batch from d_model => num_heads x d_k
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # Split into multiple heads
        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        # Apply scaled dot-product attention
        scaled_attention, attention_weights = self._scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            training=training,
        )

        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear projection
        output = self.W_o(concat_attention)
        return output
