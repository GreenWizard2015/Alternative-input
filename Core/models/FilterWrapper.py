"""Orchestrator for filter classification pipeline.

Coordinates the filter classification workflow by:
1. Managing FilterModel for binary classification of eye image validity
2. Handling weight saving/loading operations
3. Providing evaluation capabilities for model performance

Provides a unified interface for filter model operations with proper serialization.
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import shutil

from Core import Utils
from NN.models import FilterModel
import NN.Utils as NNU


class FilterWrapper:
    """Orchestrator that manages FilterModel for binary classification.

    Manages:
    - FilterModel: Performs binary classification on left and right eye images
    - Weight saving/loading operations
    - Model evaluation and prediction capabilities

    Attributes:
        _model_name: Name identifier for the model
        _latent_size: Dimension of latent feature representations
        _model: FilterModel instance for binary classification
    """

    def __init__(
        self,
        model: str = "filter",
        **kwargs: Any,
    ) -> None:
        """Initialize filter model wrapper.

        Args:
            model: Model identifier string (default: "filter")
            latent_size: Dimension of latent feature representations (default: 64)
            **kwargs: Additional arguments including:
                - weights: Dictionary with folder/postfix for loading pre-trained weights
        """
        super().__init__()

        latent_size = 64 * 2

        self._model_name = model
        self._latent_size = latent_size
        self._micro_batch_size = 16

        # Create FilterModel with proper naming
        self._model: FilterModel = FilterModel(
            latent_size=latent_size,
            name="filter_model",
        )

        self.compile()
        self._build_model()
        if "weights" in kwargs and kwargs["weights"]:
            self.load(**kwargs["weights"])

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
            fake_data: Input features (left eye, right eye)
            fake_y: Ground truth targets (validity) - only includes validity if include_result=True
        """
        fake_data = {
            "left eye": tf.random.normal(
                (batch_size, 48, 48, 1),
                dtype=tf.float32,
            ),
            "right eye": tf.random.normal(
                (batch_size, 48, 48, 1),
                dtype=tf.float32,
            ),
        }

        fake_y = {
            "is_valid": tf.random.uniform((batch_size,), 0, 1, dtype=tf.float32),
        }

        return fake_data, fake_y if include_result else fake_data

    def _build_model(self) -> None:
        """Build the model by calling it with fake data.

        This initializes all weights and layers in the neural network,
        allowing the model to be saved immediately after construction.

        Creates and initializes:
        1. FilterModel for binary classification of eye images

        If building fails, the model will be built lazily on first real call.
        This is non-critical; only affects when weights can be saved.
        """
        # Create fake input data for model building
        fake_inputs, fake_targets = self.create_fake_data(
            batch_size=1, include_result=True
        )

        temp_losses = self._train_on(fake_inputs, fake_targets)
        self._loss_keys: List[str] = list(temp_losses.keys())

    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """Process eye images through the filter classification pipeline.

        Args:
            inputs: Dictionary containing:
                - 'left eye': Left eye image of shape (batch, 48, 48, 1)
                - 'right eye': Right eye image of shape (batch, 48, 48, 1)
                - 'validity': Binary labels (batch, 1) - optional during inference
                OR nested structure with 'clean'/'augmented' keys
            training: Whether in training mode (default: False). Controls dropout behavior.

        Returns:
            Dictionary with:
                - 'predictions': Sigmoid predictions of shape (batch, 1)
                - 'features': Concatenated eye features of shape (batch, 2 * latent_size)
        """
        outputs = self._model(inputs, training=training)
        return outputs

    def __call__(self, data, training: bool = False) -> Dict[str, np.ndarray]:
        """Make predictions on input data.

        Args:
            data: Dictionary with 'left eye', 'right eye' keys containing numpy arrays

        Returns:
            Dictionary with numpy-based predictions and features
        """
        prediction = self.call(inputs=data, training=training)
        return prediction

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Get trainable variables from FilterModel."""
        return self._model.trainable_variables

    def save(self, folder: str = "models", postfix: str = "") -> None:
        """Save model with full serialization (.npz format).

        Creates the following files in the specified folder:
        - {model_name}{postfix}/filter_model.npz: Full FilterModel (weights only)

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

        # Save model using NPZ format (FilterModel implements NpzModelMixin)
        self._model.save_npz(f"{model_path}/filter_model")

    def load(
        self,
        folder: str,
        postfix: str = "",
        force: bool = False,
    ) -> None:
        """Load model weights from NPZ format with force support for missing weights.

        Loads weights into existing FilterModel instance from:
        - {folder}/{model_name}/{postfix}/filter_model: FilterModel weights

        Args:
            folder: Directory containing model files to load from
            postfix: Suffix used when files were saved (default: "")
            force: Whether to continue with random initialization if weights are missing (default: False)
        """
        # Get model path using helper method
        model_path = self.get_checkpoint_path(folder, postfix)

        # Load FilterModel weights
        self._model.load_npz(f"{model_path}/filter_model", force=force)

    def get_checkpoint_path(self, folder: str, postfix: str = "") -> str:
        """Get full checkpoint path for this model.

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
        """Evaluate model on input data with binary classification metrics.

        Args:
            xy: Tuple of (X_dict, Y_dict) where:
                - X_dict: Input features (left eye, right eye)
                - Y_dict: Ground truth with 'validity' key containing binary labels

        Returns:
            Dictionary with:
            - 'result': Binary cross-entropy loss array
            - 'accuracy': Binary classification accuracy array
        """
        x, y = xy

        # Run classification pipeline
        predictions_dict = self.call(inputs=x, training=False)
        predictions = predictions_dict["predictions"]

        # Compute binary cross-entropy loss
        loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")
        loss_values = loss_fn(y["is_valid"][:, None], predictions)
        losses = {
            "result": loss_values,
        }

        return self._with_accuracy(predictions, y["is_valid"], losses)

    def compile(self) -> None:
        """Create and configure optimizer for training.

        Initializes AdamW optimizer with default configuration for filter training.
        """
        self._optimizer = NNU.create_optimizer()

    def _build_loss_keys(self) -> None:
        """Initialize loss keys by running _train_on with fake data.

        This determines the structure of losses returned by _train_on
        during training, allowing the training loop to pre-allocate
        TensorArray with the correct size.
        """
        fake_data, fake_y = self.create_fake_data(batch_size=1, include_result=False)
        temp_losses = self._train_on(fake_data, fake_y)
        self._loss_keys: List[str] = list(temp_losses.keys())

    def _calc_accuracy_per_label(self, pred, ytrue, label, threshold=0.1):
        pred = pred[:, 0]
        mask = tf.equal(ytrue, label)
        pred = tf.boolean_mask(pred, mask)
        ytrue = tf.boolean_mask(ytrue, mask)

        pred = tf.concat([[label], pred], axis=-1)
        ytrue = tf.concat([[label], ytrue], axis=-1)

        tf.debugging.assert_equal(tf.shape(pred), tf.shape(ytrue))
        pred_threshold_low = pred < threshold
        pred_threshold_high = (1.0 - pred) < threshold
        pred = tf.where(
            pred_threshold_low, 0.0, tf.where(pred_threshold_high, 1.0, 0.5)
        )

        tf.debugging.assert_equal(tf.shape(pred), tf.shape(ytrue))
        eq_true = tf.cast(tf.equal(ytrue, pred), tf.float32)
        return tf.stop_gradient(eq_true)

    def _with_accuracy(self, pred, ytrue, res):
        return {
            **res,
            "accuracy_0_0.1": tf.reduce_mean(
                self._calc_accuracy_per_label(pred, ytrue, label=0.0, threshold=0.1)
            ),
            "accuracy_1_0.1": tf.reduce_mean(
                self._calc_accuracy_per_label(pred, ytrue, label=1.0, threshold=0.1)
            ),
            "accuracy_0_0.5": tf.reduce_mean(
                self._calc_accuracy_per_label(pred, ytrue, label=0.0, threshold=0.5)
            ),
            "accuracy_1_0.5": tf.reduce_mean(
                self._calc_accuracy_per_label(pred, ytrue, label=1.0, threshold=0.5)
            ),
        }

    @tf.function
    def _train_on(
        self,
        data: Dict[str, tf.Tensor],
        y: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Single training step for filter classification.

        Args:
            data: Input data dictionary with left and right eye images
            y: Ground truth target dictionary with 'validity' key containing binary labels

        Returns:
            Dictionary mapping loss names to scalar tensor values.
        """
        # Get filter predictions
        predictions_dict = self.call(inputs=data, training=True)
        predictions = predictions_dict["predictions"]

        # Compute binary cross-entropy loss
        loss_values = tf.losses.binary_crossentropy(y["is_valid"], predictions[:, 0])
        losses = {
            "result": tf.reduce_mean(loss_values),
        }

        return self._with_accuracy(predictions, y["is_valid"], losses)

    @tf.function
    def _train_step(
        self, data: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:
        """Single training step with gradient updates.

        Args:
            data: Tuple of (X_dict, Y_dict) where:
                - X_dict: Input data with left and right eye images
                - Y_dict: Ground truth with 'validity' key containing binary labels

        Returns:
            Dictionary of scalar loss tensor values.
        """
        x, y = data
        x = x["clean"]

        batch_size = tf.shape(list(x.values())[0])[0]

        # Determine batch size
        micro_batch_size = (
            self._micro_batch_size if self._micro_batch_size is not None else batch_size
        )

        # Calculate number of micro-batches
        num_micro_batches = tf.cast(
            tf.math.ceil(
                tf.cast(batch_size, tf.float32) / tf.cast(micro_batch_size, tf.float32)
            ),
            tf.int32,
        )

        # Initialize TensorArray for storing losses per micro-batch
        accumulated_losses = tf.TensorArray(
            dtype=tf.float32, size=num_micro_batches, dynamic_size=False
        )

        for idx in tf.range(num_micro_batches):
            start_idx = idx * micro_batch_size
            end_idx = tf.minimum(start_idx + micro_batch_size, batch_size)

            # Extract micro-batch from full batch
            micro_x = {k: v[start_idx:end_idx] for k, v in x.items()}
            micro_y = {k: v[start_idx:end_idx] for k, v in y.items()}

            with tf.GradientTape() as tape:
                batch_losses = self._train_on(micro_x, micro_y)

                # Store losses as 1D tensor for this micro-batch
                loss_values = tf.stack(
                    [batch_losses[k] for k in self._loss_keys], axis=0
                )
                accumulated_losses = accumulated_losses.write(idx, loss_values)
                total_loss = sum(batch_losses.values(), tf.constant(0.0))

            # Compute and apply gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)
            self._optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Average losses across accumulated micro-batches
        stacked_losses = accumulated_losses.stack()
        avg_losses = tf.reduce_mean(stacked_losses, axis=0)

        return {k: avg_losses[idx] for idx, k in enumerate(self._loss_keys)}

    def eval(
        self,
        data: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]],
    ) -> Dict[str, float]:
        """Evaluate model and return average metrics.

        Args:
            data: Tuple of (X_dict, Y_dict) for evaluation

        Returns:
            Dictionary with averaged metrics:
            - 'result': Average binary cross-entropy loss
            - 'accuracy': Average binary classification accuracy
        """
        losses = self.evaluate(data)
        return Utils.to_numpy(losses)

    def fit(
        self, data: Tuple[Dict[str, Dict[str, tf.Tensor]], Tuple[tf.Tensor]]
    ) -> Dict[str, Any]:
        """Perform one training step.

        Args:
            data: Training data tuple containing input tensors and targets, or generator that yields such data

        Returns:
            Dictionary of loss values as numpy arrays
        """
        # Handle generator data by unwrapping if needed
        if not isinstance(data, tuple):
            # Generator case: unwrap the first batch
            data = next(data)

        losses = self._train_step(data)
        return Utils.to_numpy(losses)
