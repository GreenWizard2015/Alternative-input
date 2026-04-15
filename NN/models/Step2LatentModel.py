"""Model for processing temporal sequences of latent features.

Second stage of the Face2Latent pipeline that processes temporal sequences
through Transformer encoder blocks and fusion blocks to produce final latent representations.
"""

from typing import Any, Dict
import tensorflow as tf
from NN.layers.sMLP import sMLP
from NN.layers.TimeEncodingLayer import TimeEncodingLayer
from NN.layers.TransformerEncoderBlock import TransformerEncoderBlock
from NN.layers.LinearAttentionMixer import LinearAttentionMixer
from NN.models.NpzModelMixin import NpzModelMixin
from NN.Constants import (
    STEP2LATENT_TRANSFORMER_BLOCKS,
    STEP2LATENT_MLP_ACTIVATION,
    STEP2LATENT_TRANSFORMER_DROPOUT_RATE,
    STANDARD_NONNEGATIVE_ACTIVATION,
)


class Step2LatentModel(NpzModelMixin, tf.keras.Model):
    """Processes sequence of latent steps to produce final latent representation.

    This model processes temporal sequences of latent vectors combined with
    time and embedding information through Transformer encoder blocks and fusion blocks
    to produce a final latent representation. It's the second stage in the Face2Latent
    pipeline.

    Architecture:
        - Time encoder: Encodes temporal information
        - Transformer blocks: Processes temporal dependencies with self-attention
        - Fusion blocks: Combines temporal and spatial features

    Attributes:
        latent_size: Dimension of latent representations
    """

    def __init__(
        self,
        latent_size: int,
        num_sink_tokens: int = 1,
        num_heads: int = None,
        dff_multiplier: int = None,
        scale_mult: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize Step2LatentModel.

        Args:
            latent_size: Base dimension of the latent representation.
            num_sink_tokens: Number of learnable sink tokens to prepend to sequence.
            num_heads: Number of attention heads in transformer blocks.
                If None, computed from scale_mult: max(1, int(16 * scale_mult)).
                Must divide scaled_latent_size evenly for multi-head attention.
            dff_multiplier: Multiplier for feed-forward dimension.
                If None, computed from scale_mult: int(16 * scale_mult).
                Feed-forward dimension = scaled_latent_size * dff_multiplier.
            scale_mult: Scaling multiplier for all dimensions (default: 1.0).
                Must be positive. Scaled dimensions:
                - scaled_latent_size = int(latent_size * scale_mult)
                - num_heads scaled from 16 to max(1, int(16 * scale_mult)) if not explicit
                - dff_multiplier scaled from 16 to int(16 * scale_mult) if not explicit
            **kwargs: Additional keyword arguments passed to parent Model class.

        Raises:
            ValueError: If any parameter is invalid or if num_heads doesn't divide scaled_latent_size.
        """
        super().__init__(**kwargs)

        # Input validation
        if latent_size <= 0:
            raise ValueError(f"latent_size must be positive, got {latent_size}")
        if num_sink_tokens <= 0:
            raise ValueError(f"num_sink_tokens must be positive, got {num_sink_tokens}")
        if scale_mult <= 0:
            raise ValueError(f"scale_mult must be positive, got {scale_mult}")

        # Compute scaled latent_size
        scaled_latent_size = int(latent_size * scale_mult)

        # Compute num_heads: use scaled value if not explicitly provided
        if num_heads is None:
            target_num_heads = max(1, int(16 * scale_mult))
            # Find largest divisor of scaled_latent_size that doesn't exceed target
            valid_heads = 1
            for h in range(target_num_heads, 0, -1):
                if scaled_latent_size % h == 0:
                    valid_heads = h
                    break
            num_heads = valid_heads
        else:
            # Explicit num_heads provided: validate divisibility with scaled latent_size
            if num_heads <= 0:
                raise ValueError(f"num_heads must be positive, got {num_heads}")
            if scaled_latent_size % num_heads != 0:
                raise ValueError(
                    f"scaled_latent_size ({scaled_latent_size}) must be divisible by num_heads ({num_heads})"
                )

        # Compute dff_multiplier: use scaled value if not explicitly provided
        if dff_multiplier is None:
            dff_multiplier = int(16 * scale_mult)

        if dff_multiplier <= 0:
            raise ValueError(f"dff_multiplier must be positive, got {dff_multiplier}")

        self.latent_size = latent_size
        self._scaled_latent_size = scaled_latent_size
        self._scale_mult = scale_mult
        self._num_sink_tokens = num_sink_tokens
        self._num_heads = num_heads
        self._dff_multiplier = dff_multiplier

        # Create encoder and processing layers
        self.time_encoder = TimeEncodingLayer(name="time_encoder")
        self.mlp_init = sMLP(
            sizes=[scaled_latent_size],
            activation=STANDARD_NONNEGATIVE_ACTIVATION,
            name="mlp_init",
        )

        # Transformer blocks for temporal processing with explicit unique names
        self.transformer_blocks = [
            TransformerEncoderBlock(
                d_model=scaled_latent_size,
                dff=scaled_latent_size * self._dff_multiplier,
                dropout_rate=STEP2LATENT_TRANSFORMER_DROPOUT_RATE,
                name=f"transformer_encoder_block_{i}",
            )
            for i in range(STEP2LATENT_TRANSFORMER_BLOCKS)
        ]

        # Adaptive low-rank transformation layers after transformer blocks
        # Uses sample-dependent low-rank factorization for parameter efficiency
        self.mlps_transformer = [
            tf.keras.layers.Dense(
                units=scaled_latent_size,
                activation=STEP2LATENT_MLP_ACTIVATION,
                name=f"adaptive_transform_{i}",
            )
            for i in range(STEP2LATENT_TRANSFORMER_BLOCKS)
        ]

        # Single multi-output mixer to generate all sink tokens at once
        # Uses optimized Dense(N × n_heads) for efficient computation, n_heads calculated via for loop
        self.sink_mixer = LinearAttentionMixer(
            n_outputs=num_sink_tokens,
            max_dim=16,
            name="sink_tokens_mixer",
        )

        # Final layers
        self.mlp_final = sMLP(
            sizes=[4 * scaled_latent_size, 2 * scaled_latent_size, scaled_latent_size],
            activation=STANDARD_NONNEGATIVE_ACTIVATION,
            name="mlp_final",
        )
        self.final_dense = tf.keras.layers.Dense(
            scaled_latent_size, activation="linear", name="final_dense"
        )

    def build(self, input_shape: Dict[str, tuple]) -> None:
        """Build model for given input shape.

        Args:
            input_shape: Dictionary with input shapes for each key in the model.
        """
        # All sublayers are built in __init__, just mark as built
        super().build(input_shape)

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Process temporal latent sequence to final latent representation.

        Learnable sink tokens are internally prepended before transformer blocks
        and removed after processing. This is transparent to callers.

        Args:
            inputs: Dictionary containing:
                - 'latent': Latent vectors of shape (batch, seq_len, latent_size)
                - 'time': Time values of shape (batch, seq_len, 1)
                - 'embeddings': Embeddings of shape (batch, seq_len, embeddings_size)
            training: Boolean indicating training or inference mode.

        Returns:
            Final latent representation of shape (batch, seq_len, latent_size).
            Note: Output shape matches input sequence length; sink tokens removed.
        """
        steps_data = inputs["latent"]
        batch_size = tf.shape(steps_data)[0]
        sequence_length = tf.shape(steps_data)[1]
        embeddings = tf.reshape(inputs["embeddings"], (batch_size, sequence_length, -1))
        encoded_time = self.time_encoder(inputs["time"], training=training)

        # Initial temporal processing
        emb = self.mlp_init(
            tf.concat([encoded_time, embeddings], axis=-1),
            training=training,
        )
        temporal = steps_data

        # Handle sink tokens - efficient skip when num_sink_tokens == 0
        if self._num_sink_tokens > 0:
            # Generate all sink tokens at once: (batch, num_sink_tokens, latent_size)
            batch_sink_tokens = self.sink_mixer(steps_data, training=training)

            # Create dummy embeddings for sink tokens (zeros)
            sink_embeddings = tf.zeros(
                (batch_size, self._num_sink_tokens, tf.shape(emb)[-1]), dtype=emb.dtype
            )
            emb = tf.concat([sink_embeddings, emb], axis=1)
            temporal = tf.concat([batch_sink_tokens, temporal], axis=1)

        # Transformer blocks with temporal modeling
        for block_id in range(STEP2LATENT_TRANSFORMER_BLOCKS):
            # Concatenate all inputs for transformer
            combined = tf.concat([temporal, emb], axis=-1)
            combined = self.mlps_transformer[block_id](combined, training=training)

            temporal = self.transformer_blocks[block_id](combined, training=training)

        # Remove sink tokens before returning
        if self._num_sink_tokens > 0:
            temporal = temporal[:, self._num_sink_tokens :, :]
            emb = emb[:, self._num_sink_tokens :, :]

        # # Final output processing
        final = self.mlp_final(
            tf.concat([steps_data, temporal, emb], axis=-1),
            training=training,
        )
        final = self.final_dense(final, training=training)
        return steps_data + final
