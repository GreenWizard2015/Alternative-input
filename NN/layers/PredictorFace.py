"""Layer for predicting facial mesh points from latent features."""

from typing import Any
import tensorflow as tf
import tensorflow.keras.layers as L
from Core.Utils import FACE_MESH_POINTS, FACE_MESH_INVALID_VALUE
from NN.layers.sMLP import sMLP
from NN.Constants import (
    PREDICTOR_BLOCK_ACTIVATION,
    PREDICTOR_BLOCK_MLP_SIZES,
    FACE_VALIDITY_THRESHOLD,
)


class PredictorFace(tf.keras.layers.Layer):
    """Predicts facial mesh points from latent representations.

    Architecture:
        - sMLP: Processes latent features through multiple dense layers
        - Additional sMLP: Deeper processing for face mesh prediction
        - Dense layer: Projects to facial mesh point coordinates
        - sMLP validity branch: Deeper processing for validity prediction
        - Validity head: Predicts if face is valid

    Attributes:
        _feature_mlp: sMLP for initial feature processing
        _face_mlp: sMLP for face mesh prediction processing
        _dense: Dense layer for face mesh prediction
        _validity_mlp: sMLP for validity feature processing
        _validity_head: Dense layer for validity prediction
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the PredictorFace layer.

        Args:
            **kwargs: Additional keyword arguments passed to parent Layer class.
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)

        # Store sublayer names directly

    def build(self, input_shape):
        """Build the layer with input shape.

        Args:
            input_shape: Input tensor shape (batch, seq_len, latent_dim).
        """
        # Create sMLPs for feature processing
        self._feature_mlp = sMLP(
            sizes=PREDICTOR_BLOCK_MLP_SIZES,
            activation=PREDICTOR_BLOCK_ACTIVATION,
            name="FeatureMLP",
        )
        self._feature_mlp.build(input_shape)

        # Face MLP receives output from feature MLP
        mlp_output_size = PREDICTOR_BLOCK_MLP_SIZES[-1]
        self._face_mlp = sMLP(
            sizes=PREDICTOR_BLOCK_MLP_SIZES,
            activation=PREDICTOR_BLOCK_ACTIVATION,
            name="FaceMLP",
        )
        self._face_mlp.build((input_shape[0], input_shape[1], mlp_output_size))

        # Dense layer for face mesh prediction
        self._points_pred = L.Dense(
            units=FACE_MESH_POINTS * 2, activation="linear", name="Dense"
        )
        self._points_pred.build((input_shape[0], input_shape[1], mlp_output_size))

        # Validity MLP
        self._validity_mlp = sMLP(
            sizes=PREDICTOR_BLOCK_MLP_SIZES,
            activation=PREDICTOR_BLOCK_ACTIVATION,
            name="ValidityMLP",
        )
        self._validity_mlp.build((input_shape[0], input_shape[1], mlp_output_size))

        # Validity head
        self._validity_head = L.Dense(
            units=FACE_MESH_POINTS, activation="sigmoid", name="ValidityHead"
        )
        self._validity_head.build((input_shape[0], input_shape[1], mlp_output_size))

        super().build(input_shape)

    def call(
        self, latent_features: tf.Tensor, training: bool = False, **kwargs: Any
    ) -> tf.Tensor:
        """Predict facial mesh points from latent features.

        Args:
            latent_features: Input tensor of shape (batch, seq_len, latent_dim).
            training: Whether in training mode (default: False).
            **kwargs: Additional call arguments.

        Returns:
            Face mesh points of shape (batch, seq_len, face_points, 2).
            Invalid points are set to FACE_MESH_INVALID_VALUE based on validity predictions.

        Example:
            >>> predictor = PredictorFace()
            >>> latent = tf.random.normal((4, 10, 256))
            >>> face_points = predictor(latent)
            >>> assert face_points.shape == (4, 10, 478, 2)
        """
        batch_size = tf.shape(latent_features)[0]
        sequence_length = tf.shape(latent_features)[1]

        # Initial feature processing
        feature_output = self._feature_mlp(latent_features, training=training)

        # Process for face points prediction
        face_mlp_output = self._face_mlp(feature_output, training=training)
        face_points_pred = self._points_pred(face_mlp_output, training=training)

        # Process for validity prediction
        validity_mlp_output = self._validity_mlp(feature_output, training=training)
        validity_scores = self._validity_head(validity_mlp_output, training=training)
        # Round to 0/1 while preserving gradients through original sigmoid output
        rounded_validity = tf.where(FACE_VALIDITY_THRESHOLD < validity_scores, 1.0, 0.0)
        validity_scores = validity_scores + tf.stop_gradient(
            rounded_validity - validity_scores
        )

        # Reshape: (batch, seq_len, face_points*2) -> (batch, seq_len, face_points, 2)
        face_points_pred = tf.reshape(
            face_points_pred, (batch_size, sequence_length, FACE_MESH_POINTS, 2)
        )
        # Reshape: (batch, seq_len, face_points) -> (batch, seq_len, face_points, 1)
        validity_scores = tf.expand_dims(validity_scores, axis=-1)

        # Blend between FACE_MESH_INVALID_VALUE (invalid) and actual face points (valid)
        face_points_pred = (
            face_points_pred * validity_scores
            + FACE_MESH_INVALID_VALUE * (1.0 - validity_scores)
        )

        return face_points_pred
