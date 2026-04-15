"""EmbeddingsProcessor component for processing and mixing embeddings.

This component handles the mixing/processing of embeddings from EmbeddingsTable
with eager loading of all processing layers. Stats and default IDs are passed
dynamically at runtime rather than being stored in the constructor.

Usage Example:
    from NN.models.EmbeddingsProcessor import EmbeddingsProcessor

    # Create processor with eager loading
    processor = EmbeddingsProcessor(
        embedding_size=64,
        mixing_method="attention"
    )

    # Process embeddings with dynamic config
    mixed_embeddings = processor.call(
        concatenated_embeddings=embeddings_tensor,  # From EmbeddingsTable
        shape=batch_shape
    )

Note:
    This is a Keras Layer and can be used as part of a Keras model.
    ALWAYS loads resources on initialization (eager loading).
    Stats and default_ids are passed dynamically at runtime via call().
"""

import tensorflow as tf
from typing import Any, Optional
from NN.layers.LinearAttentionMixer import LinearAttentionMixer
from NN.models.NpzModelMixin import NpzModelMixin
from NN.Constants import STANDARD_NONNEGATIVE_ACTIVATION
from Core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingsProcessor(NpzModelMixin, tf.keras.Model):
    """Component for processing and mixing embeddings with eager loading.

    This component handles the mixing/processing of embeddings from EmbeddingsTable
    with eager loading of all processing layers. Stats and default IDs are passed
    dynamically at runtime rather than being stored in the constructor.

    Processes individual embeddings by:
    1. Requiring embeddings dictionary with all required keys
    2. Using selected mixing method (dense or attention) to combine embeddings
    3. Applying final dense transformations

    Eager loading ensures all resources are immediately available when the component
    is created, avoiding any lazy initialization dependencies.

    No dependency on vocabulary stats or default IDs during initialization.
    Focuses purely on embedding mixing/processing operations.
    """

    def __init__(
        self,
        embedding_size: int = 64,
        mixing_method: str = "attention",
        **kwargs: Any,
    ) -> None:
        """Initialize EmbeddingsProcessor with eager loading.

        Args:
            embedding_size: Dimension of embedding vectors
            mixing_method: Method to mix embeddings. Options: 'dense', 'attention'
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If embedding_size is not positive or mixing_method is invalid.

        Note:
            ALWAYS loads resources on initialization (eager loading).
            Stats and default_ids are passed dynamically at runtime via call().
        """
        # Input validation
        if embedding_size <= 0:
            raise ValueError(f"embedding_size must be positive, got {embedding_size}")
        if mixing_method not in ("dense", "attention"):
            raise ValueError(
                f"mixing_method must be 'dense' or 'attention', got {mixing_method}"
            )

        super().__init__(**kwargs)
        self._embedding_size = embedding_size
        self._mixing_method = mixing_method

        # Initialize mixing layers lazily (will be built in build() method)
        self._mixer: Optional[tf.keras.layers.Layer] = None
        self._final_dense: Optional[tf.keras.layers.Layer] = None

        # Define layer structure immediately (eager definition)
        self._initialize_mixing_layers()

        # Build the model (creates actual weights in sublayers)
        self.build(input_shape=None)

    def build(self, input_shape=None) -> None:
        """Build the model - initialize all sublayer weights.

        Creates weights in mixing layers and final dense layer by calling them
        with dummy inputs. This follows the Keras pattern where build() actually
        initializes weights, while __init__() just defines the layer structure.

        Args:
            input_shape: Input shape (not used for dict-based models, optional)
        """

        """Build layers by calling them with dummy input to mark them as built."""
        if self._final_dense is None:
            raise RuntimeError("Final dense layer not initialized")

        batch_size, timesteps = 1, 1

        if self._mixing_method == "dense":
            # Dense mode: final dense layer directly processes concatenated embeddings
            # Input: [batch, timesteps, 5 * embedding_size]
            dense_input = tf.zeros((batch_size, timesteps, self._embedding_size * 5))
            self._final_dense(dense_input)
        elif self._mixing_method == "attention":
            # Attention mode: mixer pools embeddings, then final dense transforms
            if self._mixer is None:
                raise RuntimeError("Mixer not initialized for attention mode")
            # LinearAttentionMixer expects [batch, spatial_dim, feature_dim]
            attention_input = tf.zeros((batch_size, 5, self._embedding_size))
            # Output shape: (batch, 1, embedding_size) from n_outputs=1
            attention_output = self._mixer(attention_input)
            # Squeeze n_outputs dimension: (batch, 1, embedding_size) -> (batch, embedding_size)
            attention_output = tf.squeeze(attention_output, axis=1)
            # Final dense expects [batch, embedding_size] from mixer output
            self._final_dense(attention_output)

        # Mark model as built for Keras' serialization machinery
        # Use super().build() instead of direct assignment to maintain Keras state
        super().build(input_shape or {})

    def _initialize_mixing_layers(self) -> None:
        """Initialize mixing layers immediately on creation."""
        if self._mixing_method == "dense":
            # Dense mode: no mixer, final dense layer handles mixing
            self._mixer = None
        elif self._mixing_method == "attention":
            # Attention mode: use LinearAttentionMixer
            self._mixer = LinearAttentionMixer(
                activation=STANDARD_NONNEGATIVE_ACTIVATION,
                name="mixer_attention",
            )
        else:
            raise ValueError(
                f"mixing_method must be one of 'dense', 'attention', got: {self._mixing_method}"
            )

        # Final dense layer for output transformation
        self._final_dense = tf.keras.layers.Dense(
            units=self._embedding_size, activation="tanh", name="final_dense"
        )

    def call(self, **kwargs: Any) -> tf.Tensor:
        """Process concatenated embeddings with dynamic configuration.

        Takes concatenated embeddings and processes them using the configured
        mixing method. Expands embeddings from (B, 1, 5*embedding_size) to (B, T, embedding_size).

        Args:
            concatenated_embeddings: Concatenated embedding tensor of shape (batch, 1, 5*embedding_size)
            shape: Shape tensor of shape [2] indicating [batch_size, timesteps]
            training: Whether in training mode (default: False)

        Returns:
            Processed and expanded embedding tensor of shape [batch_size, timesteps, embedding_size]
        """
        # Extract and validate required parameters
        required_params = ["concatenated_embeddings", "shape"]
        for param in required_params:
            if param not in kwargs:
                raise KeyError(f"'{param}' parameter is required")

        concatenated = kwargs["concatenated_embeddings"]
        shape = kwargs["shape"]
        training = kwargs.get("training", False)

        # Validate concatenated embeddings tensor
        if len(concatenated.shape) != 3:
            raise ValueError(
                f"Concatenated embeddings must be rank 3 (batch, 1, features), got rank {len(concatenated.shape)}"
            )

        # Validate feature dimension is 5*embedding_size
        expected_feature_dim = 5 * self._embedding_size
        if concatenated.shape[-1] != expected_feature_dim:
            raise ValueError(
                f"Concatenated embeddings should have feature dimension {expected_feature_dim}, got {concatenated.shape[-1]}"
            )

        if self._final_dense is None:
            raise RuntimeError("Final dense layer not initialized")
        if self._mixing_method == "attention" and self._mixer is None:
            raise RuntimeError("Mixer not initialized for attention mode")

        # Apply mixing method
        mixed = self._apply_mixing(concatenated, training)

        # Expand timesteps if needed
        target_timesteps = shape[1]
        if mixed.shape[1] == 1 and target_timesteps > 1:
            mixed = tf.tile(mixed, [1, target_timesteps, 1])

        # Apply final dense transformation
        return self._final_dense(mixed)

    def _apply_mixing(
        self, concatenated: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Apply the configured mixing method to concatenated embeddings.

        Args:
            concatenated: Concatenated embedding tensor, shape [batch, timesteps, 5*embedding_size]
            training: Whether in training mode

        Returns:
            Mixed embeddings, shape [batch, timesteps, embedding_size] (dense mode)
                            or [batch, timesteps, embedding_size] (attention mode)
        """
        if self._mixing_method == "dense":
            # Dense mode: no mixing, pass directly to final dense layer
            return concatenated

        if self._mixing_method == "attention":
            # Attention mode: LinearAttentionMixer pools the 5 embedding types
            # Reshape [batch, timesteps, 5*embedding_size] to [batch*timesteps, 5, embedding_size]
            shape = tf.shape(concatenated)
            batch_size, timesteps = shape[0], shape[1]
            reshaped = tf.reshape(
                concatenated, [batch_size * timesteps, 5, self._embedding_size]
            )
            # Apply mixing and reshape back to [batch, timesteps, embedding_size]
            if self._mixer is None:
                raise RuntimeError("Mixer not initialized for attention mode")
            # Output shape: (batch_size*timesteps, 1, embedding_size) from n_outputs=1
            mixed = self._mixer(reshaped, training=training)
            # Squeeze n_outputs dimension: (batch*timesteps, 1, embedding_size) -> (batch*timesteps, embedding_size)
            mixed = tf.squeeze(mixed, axis=1)
            return tf.reshape(mixed, [batch_size, timesteps, self._embedding_size])

        raise ValueError(f"Unknown mixing method: {self._mixing_method}")
