"""Encoder for face mesh landmarks - Attention-weighted pooling.

Encodes 478 facial landmarks into a latent vector using attention-weighted
pooling to learn which landmarks are most important for the task.

FEATURES:
---------
- Attention-weighted pooling: Dense(1) → Softmax → Weighted Sum
- Learns which landmarks are most important for the task
- Simple, efficient, and interpretable
- Lightweight implementation with minimal parameters

USAGE:
------
    from NN.layers.FaceMeshEncoder import FaceMeshEncoder
    encoder = FaceMeshEncoder(latent_size=64)
    landmarks = tf.random.normal((batch_size, 478, 2))
    latent = encoder(landmarks)
"""

import tensorflow as tf
from typing import Any

from NN.layers.CoordsEncodingLayer import CoordsEncodingLayer
from NN.layers.LinearAttentionMixer import LinearAttentionMixer
from Core.Utils import FACE_MESH_INVALID_VALUE, FACE_MESH_POINTS
from NN.Constants import STANDARD_NONNEGATIVE_ACTIVATION


class FaceMeshEncoder(tf.keras.Model):
    """Face mesh landmark encoder with attention-weighted pooling.

    Encodes 478 facial landmarks into a latent vector using:
    1. Coordinate encoding with normalized indices
    2. Attention-weighted pooling to aggregate landmarks
    3. Final projection and normalization

    Example:
        >>> encoder = FaceMeshEncoder(latent_size=64)
        >>> landmarks = tf.random.normal((batch_size, 478, 2))
        >>> latent = encoder(landmarks)  # Shape: (batch_size, 64)

    Attributes:
        latent_size: Dimension of final output latent vector
        internal_latent_size: Dimension of intermediate latent representation
    """

    def __init__(
        self,
        latent_size: int,
        scale_mult: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize attention-weighted pooling encoder.

        Args:
            latent_size: Base dimension of final output latent vector
            internal_latent_size: Dimension of intermediate latent representation (default: 32)
            scale_mult: Scaling multiplier for dimensions (default: 1.0).
                Scales the output latent_size: scaled_latent_size = int(latent_size * scale_mult)
            **kwargs: Additional Keras Model arguments

        Example:
            >>> encoder = FaceMeshEncoder(latent_size=64, scale_mult=1.0)
            >>> encoder = FaceMeshEncoder(latent_size=64, internal_latent_size=16, scale_mult=2.0)
        """
        super().__init__(**kwargs)
        if scale_mult <= 0:
            raise ValueError(f"scale_mult must be positive, got {scale_mult}")

        # Compute scaled latent_size for output dimensions
        scaled_latent_size = int(latent_size * scale_mult)

        self.latent_size = latent_size
        self._scale_mult = scale_mult
        self._scaled_latent_size = scaled_latent_size
        self.internal_latent_size = 8 + int(2 * scale_mult)

        # Common coordinate encoding - match internal_latent_size
        self.coord_encoding = CoordsEncodingLayer(
            self.internal_latent_size,
            raw=True,  # Keep raw coordinates for better localization
            name="coord_encoding",
        )

        # Project to internal latent_size
        self.coord_projection = tf.keras.layers.Dense(
            self.internal_latent_size, name="CoordProjection"
        )

        # Final output projection and normalization
        self.final_dense = tf.keras.layers.Dense(
            self._scaled_latent_size,
            activation=STANDARD_NONNEGATIVE_ACTIVATION,
            name="Final",
        )

        # Attention pooling: learns which landmarks are most important
        # Use max_dim=16 for optimal head calculation (internal_latent_size % 16 == 0)
        max_dim = 16
        self.attention_pool = LinearAttentionMixer(
            max_dim=max_dim, name="attention_pool"
        )

        # Learnable embedding for invalid points
        self.invalid_embedding = self.add_weight(
            name="InvalidEmbedding",
            shape=(self.internal_latent_size,),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
        )

    def build(self, input_shape):
        """Build the FaceMeshEncoder model.

        Args:
            input_shape: Shape of input tensor (batch, num_points, coord_dim)
                where num_points is typically 478 for face mesh and coord_dim is 2.
        """
        # Build the coordinate encoding layer
        # Input shape for coord_encoding: (batch, num_points, 3) with normalized indices
        dummy_coord_shape = (1, FACE_MESH_POINTS, 3)
        self.coord_encoding.build(dummy_coord_shape)

        # Build the coordinate projection layer
        # Input shape: (batch, num_points, internal_latent_size)
        dummy_projection_shape = (1, FACE_MESH_POINTS, self.internal_latent_size)
        self.coord_projection.build(dummy_projection_shape)

        # Build the attention pooling layer
        # Input shape: (batch, num_points, internal_latent_size)
        dummy_attention_shape = (1, FACE_MESH_POINTS, self.internal_latent_size)
        self.attention_pool.build(dummy_attention_shape)

        # Build the final dense layer
        # Input shape: (batch, internal_latent_size)
        dummy_final_shape = (1, self.internal_latent_size)
        self.final_dense.build(dummy_final_shape)

        super().build(input_shape)

    def add_normalized_indices(self, points: tf.Tensor) -> tf.Tensor:
        """Add normalized landmark indices to coordinates.

        This helps the model learn position-dependent patterns by providing
        explicit index information for each landmark.

        Args:
            points: Landmark coordinates of shape (batch, num_points, 2)

        Returns:
            Points with indices of shape (batch, num_points, 3)
        """
        B = tf.shape(points)[0]
        N = tf.shape(points)[1]

        # Create normalized indices [0, 1]
        norm_idx = tf.cast(tf.range(N), tf.float32) / tf.cast(N - 1, tf.float32)
        norm_idx = tf.repeat(norm_idx[None, :], B, axis=0)  # (B, N)
        norm_idx = tf.expand_dims(norm_idx, axis=-1)  # (B, N, 1)

        # Concatenate with coordinates
        points_with_idx = tf.concat([points, norm_idx], axis=-1)  # (B, N, 3)
        return points_with_idx

    def get_valid_points_mask(self, points: tf.Tensor) -> tf.Tensor:
        """Compute mask for valid (non-invalid) landmarks.

        Args:
            points: Landmark coordinates of shape (batch, num_points, 2)

        Returns:
            Boolean mask of shape (batch, num_points) where True = valid point
        """
        return tf.reduce_all(FACE_MESH_INVALID_VALUE != points, axis=-1, keepdims=True)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Orchestrate the encoding pipeline.

        Pipeline:
        1. Add normalized indices to coordinates
        2. Encode with CoordsEncodingLayer (32 dims)
        3. Project to internal_latent_size
        4. Replace invalid points with learnable embedding
        5. Apply attention-weighted pooling
        6. Final projection to latent_size with layer normalization

        Args:
            inputs: Facial landmark points of shape (batch, FACE_MESH_POINTS, 2)
            training: Whether in training mode

        Returns:
            Latent representation of shape (batch, latent_size)
        """
        B = tf.shape(inputs)[0]
        N = tf.shape(inputs)[1]
        tf.debugging.assert_equal(
            tf.shape(inputs),
            (B, FACE_MESH_POINTS, 2),
            message=f"Expected (batch, {FACE_MESH_POINTS}, 2) landmarks",
        )

        valid_mask = self.get_valid_points_mask(inputs)

        points = self.add_normalized_indices(inputs)
        tf.debugging.assert_equal(tf.shape(points), (B, N, 3))
        # Encode coordinates
        x = self.coord_encoding(points, training=training)
        x = self.coord_projection(x, training=training)
        tf.debugging.assert_equal(tf.shape(x), (B, N, self.internal_latent_size))

        # Broadcast invalid_embedding to match x shape and apply mask
        invalid_embedding_broadcast = tf.expand_dims(
            tf.expand_dims(self.invalid_embedding, axis=0), axis=0
        )

        x = tf.where(valid_mask, x, invalid_embedding_broadcast)
        tf.debugging.assert_equal(tf.shape(x), (B, N, self.internal_latent_size))

        # Attention-weighted pooling: output shape (B, 1, internal_latent_size) from n_outputs=1
        pooled = self.attention_pool(x, training=training)
        # Squeeze n_outputs dimension: (B, 1, internal_latent_size) -> (B, internal_latent_size)
        pooled = tf.squeeze(pooled, axis=1)
        tf.debugging.assert_equal(tf.shape(pooled), (B, self.internal_latent_size))

        # Final projection
        output = self.final_dense(pooled, training=training)
        tf.debugging.assert_equal(tf.shape(output), (B, self._scaled_latent_size))
        return output
