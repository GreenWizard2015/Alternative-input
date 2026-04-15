"""Layer for predicting outputs (gaze points or full features) from latent features."""

from typing import Any, Dict
import tensorflow as tf
import tensorflow.keras.layers as L
from NN.layers.sMLP import sMLP
from NN.layers.PredictorGaze import PredictorGaze
from NN.layers.PredictorEyes import PredictorEyes
from NN.layers.PredictorFace import PredictorFace
from NN.models.NpzModelMixin import NpzModelMixin
from NN.Constants import (
    PREDICTOR_BLOCK_ACTIVATION,
    PREDICTOR_BLOCK_SHIFT,
)
from Core.logging_config import get_logger

logger = get_logger(__name__)


class PredictorBlock(NpzModelMixin, tf.keras.Model):
    """Predicts outputs (gaze points or full features) from latent representations.

    Supports two modes with different architectures:
    - 'result': Single head for gaze points (2D coordinates)
    - 'full': Three separate heads for gaze points, eyes, and face mesh
      - Wider shared MLP (256x2, 128x2, 64x2) for richer processing
      - Separate head MLPs and dense layers for each output (eyes, face, points)

    Architecture:
        - Shared MLP: Wider feature processing [256, 128, 64]
        - Result head: Dense layer to 2D gaze points
        - Eyes head (full mode): MLP + dense layer for left/right eye images
        - Face head (full mode): MLP + dense layer for face mesh points

    Attributes:
        mode: 'result' or 'full' prediction mode
        _mlp: Shared feature extraction MLP
        _predictor_gaze: PredictorGaze layer for gaze points
        _predictor_eyes: PredictorEyes layer (full mode only)
        _predictor_face: PredictorFace layer (full mode only)
    """

    def __init__(
        self,
        mode: str = "result",
        shift: float = PREDICTOR_BLOCK_SHIFT,
        **kwargs: Any,
    ) -> None:
        """Initialize the PredictorBlock.

        Args:
            mode: Prediction mode ('result' for gaze points only, 'full' for all features).
                Default: 'result'.
            shift: Scalar value to shift predicted points. Used to normalize output
                to desired range (e.g., [0.5, 0.5] centers points).
                Default: PREDICTOR_BLOCK_SHIFT.
            **kwargs: Additional keyword arguments passed to parent Layer class.

        Raises:
            ValueError: If mode is not 'result' or 'full', or if shift is not a valid number.
        """
        if mode not in ("result", "full"):
            raise ValueError(f"mode must be 'result' or 'full', got {mode}")

        # Validate shift is a valid number (not NaN or Inf)
        try:
            shift_float = float(shift)
        except (TypeError, ValueError) as e:
            raise ValueError(f"shift must be a valid number, got {shift}") from e

        if not (-float("inf") < shift_float < float("inf")):
            raise ValueError(f"shift must be a finite number, got {shift}")

        # Ensure explicit name is set for layer tracking
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}"

        super().__init__(**kwargs)
        self.mode = mode
        self.shift = shift

        # Shared MLP for processing
        mlp_sizes = [128, 128, 128]
        if mode == "full":  # Wider for full mode
            mlp_sizes = [256, 256, 256]
        self._mlp = sMLP(
            sizes=mlp_sizes,
            activation=PREDICTOR_BLOCK_ACTIVATION,
            name="MLP",
        )
        self._mlp_norm = L.BatchNormalization(name="MLPNorm")

        # Gaze predictor (used in both modes)
        self._predictor_gaze = PredictorGaze(shift=shift, name="PredictorGaze")

        # Eyes and face predictors (full mode only)
        if self.mode == "full":
            self._predictor_eyes = PredictorEyes(name="PredictorEyes")
            self._predictor_face = PredictorFace(name="PredictorFace")

    def call(self, x: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        """Predict points from latent features.

        Args:
            x: Input tensor of shape (batch, seq_len, latent_dim).
            training: Boolean indicating training or inference mode.

        Returns:
            Dictionary containing predicted outputs. Keys depend on mode:
            If mode='result': Dictionary with 'result' key containing predicted gaze points
                of shape (batch, seq_len, 2).
            If mode='full': Dictionary with keys:
                - 'points': Face mesh points of shape (batch, seq_len, face_points, 2)
                - 'left eye': Left eye image of shape (batch, seq_len, eye_size, eye_size)
                - 'right eye': Right eye image of shape (batch, seq_len, eye_size, eye_size)
                - 'result': Gaze points of shape (batch, seq_len, 2)
        """
        # Process through shared MLP
        x = self._mlp(x, training=training)
        x = self._mlp_norm(x, training=training)

        # Predict gaze points
        points = self._predictor_gaze(x, training=training)
        # Verify gaze points shape: (batch, seq_len, 2) for gaze coordinates
        tf.debugging.assert_rank(points, 3, message="Gaze points must be rank-3 tensor")
        assert (
            points.shape[-1] == 2
        ), f"Expected 2 gaze coordinates, got {points.shape[-1]}"

        if self.mode == "result":
            return {"result": points}

        # mode == "full" - predict eyes and face
        eyes = self._predictor_eyes(x, training=training)
        face = self._predictor_face(x, training=training)

        return {
            "points": face,
            "left eye": eyes[..., 0],
            "right eye": eyes[..., 1],
            "result": points,
        }

    # NPZ save/load functionality inherited from NpzModelMixin
    # Use save_npz() and load_npz() methods from the mixin
