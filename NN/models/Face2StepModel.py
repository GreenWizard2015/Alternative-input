"""Model for encoding face mesh and eye images to latent representation.

First stage of the Face2Latent pipeline that encodes facial landmarks and eye
images through convolutional and coordinate encoding layers, then combines
them using iterative fusion blocks.
"""

from typing import Any, Dict
import tensorflow as tf
import tensorflow.keras.layers as L
from NN.layers.RolloutTimesteps import RolloutTimesteps
from NN.layers.sMLP import sMLP
from NN.models.EyeEncoder import EyeEncoder
from NN.models.FaceMeshEncoder import FaceMeshEncoder
from NN.models.NpzModelMixin import NpzModelMixin
from NN.Constants import (
    FACE2STEP_MLP_ACTIVATION,
    FACE2STEP_MLP_MULTIPLIER_LARGE,
    FACE2STEP_MLP_MULTIPLIER_MEDIUM,
    FACE2STEP_MLP_MULTIPLIER_SMALL,
)


class Face2StepModel(NpzModelMixin, tf.keras.Model):
    """Encodes face mesh points and eye images to a latent representation.

    This model processes facial landmarks (points), left and right eye images,
    and embeddings to produce a combined latent representation through multiple
    fusion blocks. It's the first stage in the Face2Latent pipeline.

    Architecture:
        - Eye encoder: Processes left and right eye images independently
        - Face mesh encoder: Processes facial landmark points
        - Fusion blocks: Combines eye features with face features iteratively

    Attributes:
        latent_size: Dimension of latent representations
    """

    def __init__(
        self,
        latent_size: int,
        scale_mult: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize Face2StepModel.

        Args:
            latent_size: Base dimension of the latent representation.
            scale_mult: Scaling multiplier for all dimensions (default: 1.0).
                Must be positive. All numeric dimensions are scaled:
                - scaled_latent_size = int(latent_size * scale_mult)
                - eye filters scaled via EyeEncoder
                - MLP sizes scaled: [3, 2, 1] × scaled_latent_size
            **kwargs: Additional keyword arguments passed to parent Model class.

        Raises:
            ValueError: If any numeric parameter is not positive.
        """
        super().__init__(**kwargs)

        # Input validation
        if latent_size <= 0:
            raise ValueError(f"latent_size must be positive, got {latent_size}")
        if scale_mult <= 0:
            raise ValueError(f"scale_mult must be positive, got {scale_mult}")

        # Compute scaled latent_size (single application of scale_mult)
        scaled_latent_size = int(latent_size * scale_mult)

        self.latent_size = latent_size
        self._scaled_latent_size = scaled_latent_size
        self._scale_mult = scale_mult

        # Create encoder layers with explicit unique names to avoid Keras name collisions
        # Pass base latent_size and scale_mult; encoders compute scaled values independently
        self.eye_encoder = RolloutTimesteps(
            lambda: EyeEncoder(
                latent_size=latent_size, scale_mult=scale_mult, name="EyeEncoder"
            ),
            name="Eyes",
        )
        self.face_encoder = RolloutTimesteps(
            lambda: FaceMeshEncoder(
                latent_size=latent_size, scale_mult=scale_mult, name="FaceMeshEncoder"
            ),
            name="FaceMesh",
        )

        self.pre_mlp_dropout = L.Dropout(0.005)
        # Create fusion blocks and MLPs for composition
        # CRITICAL: Use scaled_latent_size (not latent_size * scale_mult) to avoid double-scaling
        base_mlp_multipliers = [
            FACE2STEP_MLP_MULTIPLIER_LARGE,  # 3
            FACE2STEP_MLP_MULTIPLIER_MEDIUM,  # 2
            FACE2STEP_MLP_MULTIPLIER_SMALL,  # 1
        ]
        scaled_mlp_sizes = [int(m * scaled_latent_size) for m in base_mlp_multipliers]

        self.mlp = sMLP(
            sizes=scaled_mlp_sizes,
            activation=FACE2STEP_MLP_ACTIVATION,
            name="MLP",
        )

        self.final_dense = L.Dense(scaled_latent_size, name="Combine")
        self.final_norm = L.LayerNormalization(name="FinalNorm")

    def build(self, input_shape: Dict[str, tuple]) -> None:
        """Build model for given input shape.

        Args:
            input_shape: Dictionary with input shapes for each key in the model.
        """
        left_eye, right_eye = input_shape["left eye"], input_shape["right eye"]
        combined_eyes = (*left_eye[:-1], left_eye[-1] + right_eye[-1])
        self.eye_encoder.build(combined_eyes)
        self.face_encoder.build(input_shape["points"])
        super().build(input_shape)

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Process facial data to produce latent representation.

        Args:
            inputs: Dictionary containing:
                - 'points': Facial mesh points of shape (batch, seq_len, FACE_MESH_POINTS, 2)
                - 'left eye': Left eye image of shape (batch, seq_len, EYE_SIZE, EYE_SIZE, 1)
                - 'right eye': Right eye image of shape (batch, seq_len, EYE_SIZE, EYE_SIZE, 1)
                - 'embeddings': User/place/screen embeddings of shape (batch, seq_len, emb_size)
            training: Whether in training mode.

        Returns:
            Combined latent representation of shape (batch, seq_len, latent_size)
        """
        # Encode eyes and face mesh points
        # Eye encoder now returns mixed features directly
        mixed_eye_features = self.eye_encoder(
            [inputs["left eye"], inputs["right eye"]], training=training
        )
        encoded_face_points = self.face_encoder(inputs["points"], training=training)

        # Ensure 3D shape by flattening any extra dimensions
        batch_size = tf.shape(encoded_face_points)[0]
        sequence_length = tf.shape(encoded_face_points)[1]
        target_shape = tf.stack([batch_size, sequence_length, -1])
        encoded_face_points = tf.reshape(encoded_face_points, target_shape)

        emb = inputs["embeddings"]
        emb = tf.reshape(emb, target_shape)

        combined = tf.concat([encoded_face_points, mixed_eye_features, emb], axis=-1)
        combined = self.pre_mlp_dropout(combined, training=training)
        combined = self.mlp(combined, training=training)
        combined = self.final_dense(combined, training=training)
        combined = self.final_norm(combined, training=training)

        return combined
