"""Orchestrator for gaze prediction pipeline.

Coordinates the gaze prediction workflow by:
1. Managing EmbeddingsTable and EmbeddingsProcessor for converting categorical IDs (userId, screenId, cameraId, monitorId, placeId) to embeddings
2. Orchestrating GazePredictionModel for encoding facial features and generating latent representations
3. Applying PredictorBlock(s) to convert latent vectors to gaze point predictions
4. Supporting both shared and per-layer predictor configurations

Provides both tensor-based (call) and numpy-based (__call__) interfaces for prediction.
Handles model building, weight saving/loading, and predictions.
"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from Core.dataset_loading import ensure_sorted_stats
from NN.models.EmbeddingsTable import EmbeddingsTable
from NN.models.EmbeddingsProcessor import EmbeddingsProcessor
from Core.logging_config import get_logger
from NN.models.GazePredictionModel import GazePredictionModel
from NN.models.PredictorBlock import PredictorBlock

from Core import Utils
from Core.Constants import (
    EYE_REGION_SIZE,
    FACEMESH_LANDMARK_COUNT,
    PREDICTOR_SHIFT,
    HIERARCHY_LEVELS,
)
from Core.models.PredictionOutputTypes import (
    PredictionOutput,
    PredictionOutputNumpy,
)
from Core.utils import validate_data_dict

logger = get_logger(__name__)


class ModelWrapper(tf.keras.Model):
    """Orchestrator that coordinates submodels for gaze prediction.

    Orchestrates:
    - EmbeddingsTable and EmbeddingsProcessor: Convert categorical IDs to embedding vectors
    - GazePredictionModel: Performs actual gaze prediction computation
    - PredictorBlock: Converts final latent representation to gaze predictions

    Attributes:
        _timesteps: Number of timesteps in model
        _model_name: Name identifier for the model
        _predictor_mode: Mode for PredictorBlock instance (e.g., "full", "result")
        _stats: Dictionary with lists of available user/screen/camera/monitor/place IDs
        _default_ids: Dictionary with default IDs computed from user strings
        _embedding_size: Dimension of embedding vectors
        _model: GazePredictionModel instance
        _predictor: PredictorBlock for generating gaze predictions from latent
    """

    # Default configuration values
    DEFAULT_EMBEDDING_SIZE = 64
    DEFAULT_LATENT_SIZE = 128

    def __init__(
        self,
        timesteps: int,
        stats: Dict[str, List[str]],
        model: str = "simple",
        user: Optional[Dict[str, str]] = None,
        predictor_mode: str = "result",
        **kwargs: Any,
    ) -> None:
        """Initialize model wrapper.

        Args:
            timesteps: Number of timesteps for temporal processing
            stats: Dictionary with lists of available user/screen/camera/monitor/place IDs for indexing (required)
            model: Model identifier string
            user: Dictionary with 'userId', 'screenId', 'cameraId', 'monitorId', 'placeId' strings to compute default IDs (optional)
            predictor_mode: Mode for PredictorBlock instance (default: "result").
                Controls the output format of predictor block.
            **kwargs: Additional arguments including:
                - embeddingSize: Dimension of embedding vectors (default: 64)
                - latent_size: Dimension of latent space (default: 64)
                - mode: Training mode "full" (two-stage, default) or "encoder" (Face2Step only)
                - weights: Dictionary with folder/postfix for loading pre-trained weights

        Raises:
            ValueError: If timesteps <= 0 or stats is missing.
        """
        super().__init__()

        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}")

        if not stats:
            raise ValueError("stats is required to initialize ModelWrapper")

        self._timesteps = timesteps
        self._model_name = model
        self._predictor_mode = predictor_mode

        # Build embeddings configuration with vocab sizes and embedding dimension
        embedding_size = kwargs.get("embeddingSize", self.DEFAULT_EMBEDDING_SIZE)

        # Ensure stats are sorted for deterministic indices
        stats = ensure_sorted_stats(stats)

        # Compute default_ids from user strings if provided
        self._default_ids = None
        if user is not None:
            self._default_ids = {
                "userId": stats["userId"].index(user["userId"]),
                "screenId": stats["screenId"].index(user["screenId"]),
                "cameraId": stats["cameraId"].index(user["cameraId"]),
                "monitorId": stats["monitorId"].index(user["monitorId"]),
                "placeId": stats["placeId"].index(user["placeId"]),
            }

        # Store stats for dynamic passing
        self._stats = stats

        # Compute vocab from stats (moved from _get_embeddings)
        # Only include HIERARCHY_LEVELS keys to avoid passing extra keys like "blacklist"
        vocab = {k: len(stats[k]) for k in HIERARCHY_LEVELS if k in stats}

        # Store embedding size for use in call method
        self._embedding_size = embedding_size
        # Store latent size for use in predictor
        self._latent_size = kwargs.get("latent_size", self.DEFAULT_LATENT_SIZE)

        # Extract scale_mult for model scaling (defaults to 1.0 for backward compatibility)
        self._scale_mult = kwargs.get("scale_mult", 1.0)
        if self._scale_mult <= 0:
            raise ValueError(f"scale_mult must be positive, got {self._scale_mult}")

        # Store training mode
        self._mode = kwargs.get("mode", "full")

        # Create GazePredictionModel with scale_mult support
        self._model: GazePredictionModel = GazePredictionModel(
            latent_size=self._latent_size,
            mode=self._mode,
            scale_mult=self._scale_mult,
            name="gaze_prediction_model",
        )

        # Initialize predictor block
        self._predictor: PredictorBlock = PredictorBlock(
            mode=self._predictor_mode,
            shift=PREDICTOR_SHIFT,
            name="PredictorBlock",
        )

        # Initialize EmbeddingsTable and EmbeddingsProcessor once (eager loading)
        self._table: EmbeddingsTable = EmbeddingsTable(
            vocab=vocab,
            embedding_size=self._embedding_size,
            name="embeddings_table",
        )
        self._processor: EmbeddingsProcessor = EmbeddingsProcessor(
            embedding_size=self._embedding_size,
            mixing_method="attention",
            name="embeddings_processor",
        )

        # Build model with fake data
        self._build_model()

        # Load weights if provided
        if "weights" in kwargs:
            self.load(**kwargs["weights"])

    def build(self, input_shape) -> None:
        """Build the model - mark as ready after sublayers are created.

        For Dict-based models, we build sublayers during __init__() using fake data.
        This method just needs to mark the model as built so Keras knows it's ready.

        Args:
            input_shape: Typically None for Dict-based models
        """
        # All sublayers are already built during __init__ via _build_model()
        # This just marks the model as built for Keras' serialization machinery
        # Use a dummy shape to satisfy Keras' build requirements
        if input_shape is None:
            input_shape = {}
        super().build(input_shape)

    def create_fake_data(
        self, batch_size: int = 1, include_result: bool = True
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """Create synthetic data for model building or loss key initialization.

        Generates fake tensors with correct shapes and types for all required input fields.
        Used for model weight initialization and loss structure determination.

        Args:
            batch_size: Batch size for fake data (default: 1)
            include_result: Whether to include 'result' key in output dict (default: True).
                Set False for loss key initialization, True for model building.

        Returns:
            Tuple of (fake_data, fake_y) dictionaries with properly shaped tensors.
            fake_data: Input features (points, eyes, time, IDs)
            fake_y: Ground truth targets (result) - only includes result if include_result=True
        """
        fake_data = {
            "points": tf.random.normal(
                (batch_size, self._timesteps, FACEMESH_LANDMARK_COUNT, 2),
                dtype=tf.float32,
            ),
            "left eye": tf.random.normal(
                (batch_size, self._timesteps, EYE_REGION_SIZE, EYE_REGION_SIZE, 1),
                dtype=tf.float32,
            ),
            "right eye": tf.random.normal(
                (batch_size, self._timesteps, EYE_REGION_SIZE, EYE_REGION_SIZE, 1),
                dtype=tf.float32,
            ),
            "time": tf.random.uniform(
                (batch_size, self._timesteps, 1), 0, 1, dtype=tf.float32
            ),
            "userId": tf.zeros((batch_size, self._timesteps, 1), dtype=tf.int32),
            "placeId": tf.zeros((batch_size, self._timesteps, 1), dtype=tf.int32),
            "screenId": tf.zeros((batch_size, self._timesteps, 1), dtype=tf.int32),
            "cameraId": tf.zeros((batch_size, self._timesteps, 1), dtype=tf.int32),
            "monitorId": tf.zeros((batch_size, self._timesteps, 1), dtype=tf.int32),
        }

        fake_y = {
            "result": tf.random.uniform(
                (batch_size, self._timesteps, 2), 0, 1, dtype=tf.float32
            ),
        }

        return fake_data, fake_y

    def _build_model(self) -> None:
        """Build the model by calling it with fake data.

        This initializes all weights and layers in the neural network,
        allowing the model to be saved immediately after construction.

        Creates and initializes:
        1. PredictorBlock for converting latent to gaze points
        2. GazePredictionModel with fake input batch

        If building fails, the model will be built lazily on first real call.
        This is non-critical; only affects when weights can be saved.
        """
        # Create fake input data with correct shapes, ensuring float32 precision
        fake_inputs, fake_targets = self.create_fake_data(
            batch_size=1, include_result=True
        )

        # Include result in inputs for model building
        fake_inputs.update(fake_targets)

        # Call the model to populate weights
        # This triggers weight creation in sublayers
        _ = self.call(inputs=fake_inputs, training=False)

    def _get_embeddings(
        self, inputs: Dict[str, tf.Tensor], shape: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Get processed embeddings from raw inputs.

        Encapsulates the EmbeddingsTable → EmbeddingsProcessor pipeline,
        eliminating redundant parameter passing.

        Args:
            inputs: Raw input dict with ID tensors (userId, placeId, etc.)
            shape: Tensor of shape [batch_size, timesteps]
            training: Training mode flag

        Returns:
            Processed embeddings of shape (batch, timesteps, embedding_size)
        """
        # Get concatenated embeddings from EmbeddingsTable
        concatenated_embeddings = self._table.call(
            **{id_key: inputs.get(id_key) for id_key in HIERARCHY_LEVELS},
            default_ids=self._default_ids,
            shape=shape,
            training=training,
        )

        # Expand embeddings to full timesteps
        return self._processor.call(
            concatenated_embeddings=concatenated_embeddings,
            shape=shape,
            training=training,
        )

    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> PredictionOutput:
        """Orchestrate submodels: embeddings -> gaze prediction.

        Delegates embedding generation to EmbeddingBlock, then passes embeddings
        and facial data to GazePredictionModel for actual prediction. Finally,
        applies PredictorBlock to generate gaze point predictions.

        Args:
            inputs: Dictionary containing facial data and optional categorical IDs:
                - 'points': Face mesh landmarks of shape (batch, timesteps, FACEMESH_LANDMARK_COUNT, 2)
                - 'left eye': Left eye region of shape (batch, timesteps, EYE_REGION_SIZE, EYE_REGION_SIZE, 1)
                - 'right eye': Right eye region of shape (batch, timesteps, EYE_REGION_SIZE, EYE_REGION_SIZE, 1)
                - 'time': Time values of shape (batch, timesteps, 1)
                - 'userId': Optional user IDs of shape (batch, timesteps, 1) or (batch, timesteps)
                - 'placeId': Optional place IDs of shape (batch, timesteps, 1) or (batch, timesteps)
                - 'screenId': Optional screen IDs of shape (batch, timesteps, 1) or (batch, timesteps)
            training: Whether in training mode (default: False). Controls dropout, batch norm behavior.

        Returns:
            PredictionOutput with:
            - result: Predicted gaze points of shape (batch, timesteps, 2)
        """
        # Validate input points are in valid range
        inputs = validate_data_dict(inputs)
        points = inputs["points"]

        # Get processed embeddings (Table lookup + Processor expansion)
        embeddings = self._get_embeddings(
            inputs, tf.shape(points)[:2], training=training
        )

        # GazePredictionModel expects embeddings in inputs dict
        model_inputs = {**inputs}
        model_inputs["points"] = points
        model_inputs["embeddings"] = embeddings
        model_output = self._model(model_inputs, training=training)

        # Extract both latents from GazePredictionModel output
        latent_intermediate = model_output["intermediate_latent"]
        latent_final = model_output["final_latent"]

        # Use final latent to generate gaze predictions
        predictor_output = self._predictor(latent_final, training=training)
        # PredictorBlock returns {"result": points, ...other outputs}
        result = predictor_output["result"]

        # Convert dictionary output to structured NamedTuple
        return PredictionOutput(
            result=result,
            raw=predictor_output,
            latents=latent_final,
            intermediate_latents=latent_intermediate,
        )

    def __call__(
        self, data: Dict[str, np.ndarray], training: bool = False
    ) -> PredictionOutputNumpy:
        """Make predictions on input data with numpy interface.

        Converts numpy input to tensors, runs orchestration pipeline (embeddings + gaze
        prediction), and converts output back to numpy arrays for external consumption.

        Args:
            data: Dictionary containing input data with numpy arrays:
                - 'points': Face mesh landmarks of shape (batch, timesteps, FACEMESH_LANDMARK_COUNT, 2)
                - 'left eye': Left eye region of shape (batch, timesteps, EYE_REGION_SIZE, EYE_REGION_SIZE, 1)
                - 'right eye': Right eye region of shape (batch, timesteps, EYE_REGION_SIZE, EYE_REGION_SIZE, 1)
                - 'time': Time values of shape (batch, timesteps, 1)
                - 'userId': Optional user IDs of shape (batch, timesteps)
                - 'placeId': Optional place IDs of shape (batch, timesteps)
                - 'screenId': Optional screen IDs of shape (batch, timesteps)

        Returns:
            PredictionOutputNumpy with numpy-based predictions

        Raises:
            KeyError: If required keys missing from data
            TypeError: If data values are not numpy arrays or tensors
        """
        # Convert numpy arrays to tensors for model input
        tensor_data: Dict[str, tf.Tensor] = Utils.to_tensor(data)
        prediction = self.call(inputs=tensor_data, training=training)

        return PredictionOutputNumpy(
            result=Utils.to_numpy(prediction.result),
        )

    @property
    def timesteps(self) -> int:
        """Get number of timesteps for temporal processing.

        Returns:
            Number of timesteps configured in this model instance.
        """
        return self._timesteps

    def get_predictor(self) -> PredictorBlock:
        """Get the PredictorBlock instance.

        Returns:
            PredictorBlock instance for generating gaze predictions from latent representation.
        """
        return self._predictor

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable variables from all submodels.

        Aggregates trainable variables from:
        - _model (GazePredictionModel)
        - _predictor (PredictorBlock)
        - _table (EmbeddingsTable)
        - _processor (EmbeddingsProcessor)

        Returns:
            List of all trainable tf.Variable objects from all submodels.
            Returns empty list if no submodels have trainable variables.

        Note:
            This property ensures unified access to all model weights for:
            - Gradient computation during training
            - Weight inspection and debugging
            - Optimizer state management
        """
        variables = []

        # Aggregate trainable variables from all submodels
        if self._model is not None:
            variables.extend(self._model.trainable_variables)
        if self._predictor is not None:
            variables.extend(self._predictor.trainable_variables)
        if self._table is not None:
            variables.extend(self._table.trainable_variables)
        if self._processor is not None:
            variables.extend(self._processor.trainable_variables)

        return variables

    def save(self, folder: str = "models", postfix: str = "") -> None:
        """Save model with full serialization (.keras format).

        Creates the following files in the specified folder:
        - {model_name}{postfix}/model.keras: Full GazePredictionModel (architecture + weights)
        - {model_name}{postfix}/embeddings_table.keras: Full EmbeddingsTable (architecture + weights)
        - {model_name}{postfix}/embeddings_processor.keras: Full EmbeddingsProcessor (architecture + weights)
        - {model_name}{postfix}/predictor.keras: Full PredictorBlock (architecture + weights)

        Args:
            folder: Directory to save model files (must exist, default: "models")
            postfix: Optional suffix for filenames to distinguish different saves (default: "")

        Raises:
            OSError: If folder does not exist or write permission denied
        """
        # Get model path using helper method
        model_path = self.get_checkpoint_path(folder, postfix)

        # Remove existing model folder to ensure clean save
        shutil.rmtree(model_path, ignore_errors=True)

        # Create parent directories
        os.makedirs(model_path, exist_ok=True)

        # Save models using NPZ format (path without extension - methods add .npz)
        self._model.save_npz(f"{model_path}/model")
        self._predictor.save_npz(f"{model_path}/predictor")
        self._table.save_npz(f"{model_path}/embeddings_table")
        self._processor.save_npz(f"{model_path}/embeddings_processor")

    def load(
        self,
        folder: str,
        postfix: str = "",
        embeddings: bool = True,
        force: bool = False,
    ) -> None:
        """Load model weights from HDF5 format with force support for missing weights.

        Loads weights into existing model instances from:
        - {folder}/{model_name}/{postfix}/model: GazePredictionModel weights
        - {folder}/{model_name}/{postfix}/predictor: PredictorBlock weights
        - {folder}/{model_name}/{postfix}/embeddings_processor: EmbeddingsProcessor weights
        - {folder}/{model_name}/{postfix}/embeddings_table: EmbeddingsTable weights (if embeddings=True)

        Args:
            folder: Directory containing model files to load from
            postfix: Suffix used when files were saved (default: "")
            embeddings: Whether to load EmbeddingsTable (optional, default: True)
            force: Whether to continue with random initialization if weights are missing (default: False)
        """
        # Get model path using helper method
        model_path = self.get_checkpoint_path(folder, postfix)

        # Load each component with individual error handling
        self._model.load_npz(f"{model_path}/model", force=force)
        self._predictor.load_npz(f"{model_path}/predictor", force=force)
        self._processor.load_npz(f"{model_path}/embeddings_processor", force=force)

        if embeddings:
            self._table.load_npz(f"{model_path}/embeddings_table", force=force)

    # Public methods to avoid private field access violations
    def get_model(self) -> "GazePredictionModel":
        """Get the internal GazePredictionModel instance.

        Returns:
            GazePredictionModel instance for direct access when needed.
        """
        return self._model

    def get_embeddings_table(self) -> "EmbeddingsTable":
        """Get the EmbeddingsTable instance.

        Returns:
            EmbeddingsTable instance for managing embedding lookups.
        """
        return self._table

    def get_embeddings_processor(self) -> "EmbeddingsProcessor":
        """Get the EmbeddingsProcessor instance.

        Returns:
            EmbeddingsProcessor instance for embedding processing and expansion.
        """
        return self._processor

    def get_latent_size(self) -> int:
        """Get the latent space dimension.

        Returns:
            Dimension of latent space.
        """
        return self._latent_size * self._scale_mult

    def get_embedding_size(self) -> int:
        """Get the embedding vector dimension.

        Returns:
            Dimension of embedding vectors.
        """
        return self._embedding_size

    def get_timesteps(self) -> int:
        """Get the number of timesteps.

        Returns:
            Number of timesteps configured for temporal processing.
        """
        return self._timesteps

    def get_mode(self) -> str:
        """Get the training mode.

        Returns:
            Current training mode ('full' or 'encoder').
        """
        return self._mode

    def get_model_summary(self) -> str:
        """Get detailed model architecture summary.

        Returns:
            String representation of model architecture including submodels.
        """
        import io
        from contextlib import redirect_stdout

        summary_buffer = io.StringIO()
        with redirect_stdout(summary_buffer):
            self._model.summary(expand_nested=True)
            self._predictor.summary(expand_nested=True)
        return summary_buffer.getvalue()

    def get_checkpoint_path(self, folder: str, postfix: str = "") -> str:
        """Get full checkpoint path for this model.

        Encapsulates the path construction logic used by save() and load()
        for cleaner API access from other classes.

        Args:
            folder: Base directory for checkpoints
            postfix: Optional suffix to distinguish different saves (default: "")

        Returns:
            Full path: folder/{model_name}/{postfix} or folder/{model_name}
        """
        model_path = f"{folder}/{self._model_name}"
        if postfix:
            model_path = f"{model_path}/{postfix}"
        return model_path

    @tf.function
    def evaluate(
        self, xy: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:
        """Evaluate model on input data with loss and distance calculations.

        Uses ModelWrapper.call() to orchestrate embeddings and gaze predictions.
        Computes loss and distance metrics for gaze point accuracy.

        Args:
            xy: Tuple of (X_dict, Y_dict) where:
                - X_dict: Input features (points, eyes, embeddings, IDs)
                - Y_dict: Ground truth with 'result' key containing gaze points

        Returns:
            Dictionary containing per-sample arrays for each metric:
            - 'loss': Point prediction loss array (one value per sample)
            - 'distance': Euclidean distance from ground truth array (one value per sample)
            - Other loss components from calculate_losses() as arrays
        """
        import NN.Utils as NNU
        from Core.losses import calculate_losses

        x, y = xy

        # Validate ground truth points are in valid range
        y_validated = validate_data_dict(y)

        # Use ModelWrapper.call() to orchestrate embeddings and predictions
        # (input validation happens inside call())
        prediction_output = self.call(inputs=x, training=False)
        # Evaluate only on last frame.
        y_validated = {k: v[:, -1:] for k, v in y_validated.items()}
        predictions = {k: v[:, -1:] for k, v in prediction_output.raw.items()}
        points = predictions["result"]

        losses = calculate_losses(predictions, y_validated, training=False)
        gt = y_validated["result"]
        tf.debugging.assert_equal(
            tf.shape(points),
            tf.shape(gt),
            message="Predicted points shape must match ground truth shape",
        )
        dist = NNU.norm_vec(points - gt).length
        return {**losses, "distance": dist}

    def weights_dict(self) -> Dict[str, np.ndarray]:
        """Get model weights as a dictionary mapping layer names to weight arrays.

        Returns:
            Dictionary where keys are layer names and values are numpy arrays
            containing the weights for each layer.
        """
        weights = self.get_weights()
        layer_names = [layer.name for layer in self.layers]
        return dict(zip(layer_names, weights))

    def eval(
        self,
        data: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]],
    ) -> Dict[str, float]:
        losses = self.evaluate(data)
        return Utils.to_numpy(losses)

    def reset_embeddings(self):
        self._table.reset_embeddings()
