"""Student model trainer for knowledge distillation.

Handles student model training with teacher guidance. Manages student model
training with teacher supervision for knowledge distillation, including
multi-level loss computation and gradient computation.

Key responsibilities:
- Student model training with teacher guidance
- Multi-level loss computation with teacher supervision
- Gradient computation and optimization
- Uses provided teacher model (no recursive creation)
"""

from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
import NN.Utils as NNU

from Core import Utils
from Core.logging_config import get_logger
from Core.losses import calculate_losses
from Core.models.ModelWrapper import ModelWrapper
from Core.utils import validate_data_dict
from Core.models.TrainerAdapter import TrainerAdapter
from Core.models.PredictionOutputTypes import PredictionOutput

logger = get_logger(__name__)


class ModelStudentTrainer:
    """Handles student model training with teacher guidance.

    Specialized trainer for student models that learn from teacher models
    through knowledge distillation. Manages student model training with
    teacher supervision and multi-level loss computation.

    Attributes:
        _model_wrapper: ModelWrapper instance for student model predictions
        _teacher_model: Teacher model for knowledge distillation
        _optimizer: AdamW optimizer instance
        _train_step: TF graph-compiled training step
        _eval: TF graph-compiled evaluation
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        teachers_models: List[ModelWrapper],
        micro_batch_size: Optional[int] = 16,
        feature_match_loss_weight: float = 0.5,
        weights: Optional[Dict[str, Any]] = None,
        exclude=None,
    ) -> None:
        """Initialize student model trainer with teacher guidance.

        Args:
            model_wrapper: ModelWrapper instance for student model
            micro_batch_size: Size of micro-batches for memory efficiency (default: 16)
            feature_match_loss_weight: Weight for feature matching loss [0.0-1.0] (default: 0.5)
                Controls balance between task loss and latent feature matching loss
            weights: Optional dictionary with checkpoint loading parameters: {'folder': ..., 'postfix': ...}
                If provided, loads both model wrapper and adapters after initialization (default: None)

        Raises:
            ValueError: If model_wrapper is None or feature_match_loss_weight outside [0, 1]
        """
        if model_wrapper is None:
            raise ValueError("model_wrapper is required for ModelStudentTrainer")

        if not teachers_models:
            raise ValueError("teachers_models is required for ModelStudentTrainer")

        if not (0.0 <= feature_match_loss_weight):
            raise ValueError(
                f"feature_match_loss_weight must be 0.0 <= {feature_match_loss_weight}"
            )

        self._exclude = exclude or []
        self._model_wrapper = model_wrapper
        self._teachers_models = teachers_models
        self._micro_batch_size = micro_batch_size
        self._feature_match_loss_weight = feature_match_loss_weight

        # Create adapters for latent space adaptation (will be created during first call)
        self._adapters = {}
        if "final" not in self._exclude:
            self._adapters["final"] = TrainerAdapter(name="final")

        if "intermediate" not in self._exclude:
            self._adapters["intermediate"] = TrainerAdapter(name="intermediate")

        # Create optimizer for gradient descent
        self.compile()
        # Build model to initialize loss keys
        self._build_loss_keys()

        # Load checkpoint weights if provided via weight parameter
        if weights is not None:
            self.load(**weights)

    def compile(self) -> None:
        """Create and configure optimizer.

        Initializes AdamW optimizer with default configuration for training.
        """
        self._optimizer = NNU.create_optimizer()

    def _gather_teacher_outputs(self, x, ids) -> List[PredictionOutput]:
        """Gather predictions from all teacher models."""
        by_cache_id = {}
        for teacher, cache_id in zip(self._teachers_models, ids):
            if cache_id not in by_cache_id:
                by_cache_id[cache_id] = teacher.call(inputs=x, training=False)

        res = [by_cache_id[id] for id in ids]
        return [
            PredictionOutput(
                intermediate_latents=teacher_output.intermediate_latents,
                latents=teacher_output.latents,
                result=teacher_output.result,
                raw=None,
            ).stop_gradient()
            for teacher_output in res
        ]

    def _build_loss_keys(self) -> None:
        """Initialize loss keys by running _train_on with fake data.

        This determines the structure of losses returned by _train_on
        during training, allowing the training loop to pre-allocate
        TensorArray with the correct size.
        """
        fake_data, fake_y = self._model_wrapper.create_fake_data(
            batch_size=1, include_result=False
        )
        ids = [x.cache_id for x in self._teachers_models]
        teacher_outputs = self._gather_teacher_outputs(fake_data, ids=ids)
        temp_losses = self._train_on(fake_data, fake_y, teacher_outputs)
        self._loss_keys: List[str] = list(temp_losses.keys())

    def _calc_teachers_losses(
        self,
        y: Dict[str, tf.Tensor],
        teacher_outputs: List[PredictionOutput],
    ) -> tf.Tensor:
        losses = []
        for teacher in teacher_outputs:
            predictions = {"result": teacher.result}
            loss = tf.stop_gradient(calculate_losses(predictions, y)["result"])
            losses.append(loss[..., None])

        return losses

    def _best_teacher_result(
        self,
        teacher_outputs: List[PredictionOutput],
        teachers_losses: List[tf.Tensor],
    ) -> tf.Tensor:
        losses = tf.concat(teachers_losses, axis=-1)
        best_idx = tf.argmin(losses, axis=-1, output_type=tf.int32)
        teacher_results = tf.stack(
            [teacher.result for teacher in teacher_outputs], axis=2
        )
        return tf.gather(teacher_results, best_idx, axis=2, batch_dims=2)

    def _specialist_weights(self, teachers_losses, student_loss):
        tf.debugging.assert_rank(teachers_losses, 3)
        advantage = tf.nn.relu(student_loss - teachers_losses) * 1e3
        return advantage

    def _normalize_specialist_weights(self, advantage, delta=1e-2):
        """Energy scale must be in [delta, 1]"""
        useful = advantage > 0.0

        bad_count = tf.reduce_sum(
            tf.cast(~useful, advantage.dtype), axis=-1, keepdims=True
        )
        remaining = 1.0 - delta * bad_count
        scale = tf.math.divide_no_nan(
            remaining, tf.reduce_sum(advantage, axis=-1, keepdims=True)
        )

        weights = tf.where(useful, advantage * scale, delta)
        weights = tf.stop_gradient(weights)

        tf.debugging.assert_equal(tf.shape(weights), tf.shape(advantage))
        tf.debugging.assert_all_finite(weights, "Non-finite specialist weights.")

        return weights

    @tf.function
    def _train_on(
        self,
        data: Dict[str, tf.Tensor],
        y: Dict[str, tf.Tensor],
        teacher_outputs: List[PredictionOutput],
    ) -> Dict[str, tf.Tensor]:
        """Student training step with teacher guidance.

        Uses ModelWrapper.call() to orchestrate embeddings and gaze prediction.
        Computes losses for final predictions with optional:
        - Knowledge distillation via dual latent feature matching (if teacher exists)

        Args:
            data: Input data dictionary with facial data (embeddings added by wrapper)
            y: Ground truth target dictionary with 'result' key containing targets of shape (B, T, 2)

        Returns:
            Dictionary mapping loss names to scalar tensor values.
            If distillation enabled, includes 'latent_intermediate' and 'latent_final' keys
            for feature matching losses.

        Distillation logic:
            - convert student latents to teacher
            - convert teacher latents to student
            - pull them to match
        """
        # Validate ground truth points are in valid range
        y_validated = validate_data_dict(y)
        student_output = self._model_wrapper.call(inputs=data, training=True)

        losses = calculate_losses(
            {"result": student_output.result},
            y_validated,
            training=True,
        )
        student_loss = losses["result"][..., None]

        k = 1
        teachers_losses = self._calc_teachers_losses(y_validated, teacher_outputs)
        teachers_losses = tf.stop_gradient(
            tf.concat(teachers_losses + [student_loss] * k, axis=-1)
        )
        shp = tf.shape(student_loss)
        B, N = shp[0], shp[1]
        M = len(teacher_outputs)
        tf.debugging.assert_equal(tf.shape(teachers_losses), (B, N, M + k))
        top_k_losses_idx = tf.argsort(teachers_losses, axis=-1, direction="ASCENDING")
        top_k_losses_idx = top_k_losses_idx[..., :k]

        top_mask = M <= top_k_losses_idx
        in_top_k = tf.math.reduce_any(top_mask[..., :1], axis=-1)
        in_top_k = tf.cast(in_top_k, tf.float32)
        losses["student_acc"] = tf.stop_gradient(tf.reduce_mean(in_top_k))
        # calc adapters loss
        for idx, _ in enumerate(teacher_outputs):
            top_mask = idx == top_k_losses_idx

            in_top_k = tf.math.reduce_any(top_mask[..., :1], axis=-1)
            in_top_k = tf.cast(in_top_k, tf.float32)
            losses[f"{idx}_teacher_acc"] = tf.reduce_mean(in_top_k)

        advantage = self._specialist_weights(teachers_losses[..., :-k], student_loss)
        best_raw = tf.argmin(teachers_losses, axis=-1, output_type=tf.int32)
        best_index = tf.where(best_raw >= M, tf.zeros_like(best_raw), best_raw + 1)
        # apply adapters
        teacher_latents = [x.latents for x in teacher_outputs]
        if False:
            intermediate = [x.intermediate_latents for x in teacher_outputs]
            teacher_latents.extend(intermediate)
            advantage = tf.concat([advantage, advantage], axis=-1)

        temperature = 1e-2
        loss_weights = self._normalize_specialist_weights(advantage / temperature)
        if "final" not in self._exclude:
            adapted_loss = self._adapters["final"].calc_loss(
                student_latents=student_output.latents,
                teachers=teacher_latents,
                loss_weights=loss_weights,
                best_index=best_index,
                scale=self._feature_match_loss_weight,
            )
            losses = {**losses, **adapted_loss}

        if "intermediate" not in self._exclude:
            adapted_loss = self._adapters["intermediate"].calc_loss(
                student_latents=student_output.intermediate_latents,
                teachers=teacher_latents,
                loss_weights=loss_weights,
                best_index=best_index,
                scale=self._feature_match_loss_weight * 0.1,
            )
            losses = {**losses, **adapted_loss}

        losses = {k: tf.reduce_mean(v) for k, v in losses.items()}
        # scale distilation gradients
        distil_names = ["distil_final", "distil_intermediate"]
        distil_names = [name for name in distil_names if name in losses]
        if distil_names:
            distil_total = tf.add_n([losses[name] for name in distil_names])

            max_scale = 0.25
            scale = tf.stop_gradient(
                tf.clip_by_value(
                    tf.math.divide_no_nan(max_scale * losses["result"], distil_total),
                    0.0,
                    max_scale,
                )
            )

            for name in distil_names:
                losses[name] *= scale

        return losses

    def _train_on_subset(self, x_augm, y, teacher_outputs):
        batch_size = tf.shape(x_augm["points"])[0]

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
            micro_x = {k: v[start_idx:end_idx] for k, v in x_augm.items()}
            micro_y = {k: v[start_idx:end_idx] for k, v in y.items()}
            micro_teacher_output = [
                teacher_output.slice(start_idx, end_idx)
                for teacher_output in teacher_outputs
            ]

            with tf.GradientTape() as tape:
                batch_losses = self._train_on(micro_x, micro_y, micro_teacher_output)

                # Store losses as 1D tensor for this micro-batch
                loss_values = tf.stack(
                    [batch_losses[k] for k in self._loss_keys], axis=0
                )
                accumulated_losses = accumulated_losses.write(idx, loss_values)
                total_loss = sum(batch_losses.values(), tf.constant(0.0))

            # Get all trainable variables
            trainable_vars = self.trainable_variables
            # Compute and apply gradients
            gradients = tape.gradient(total_loss, trainable_vars)
            self._optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Average losses across accumulated micro-batches
        stacked_losses = accumulated_losses.stack()
        avg_losses = tf.reduce_mean(stacked_losses, axis=0)
        return avg_losses

    @tf.function
    def _train_step(
        self,
        data: Tuple[Dict[str, Dict[str, tf.Tensor]], Dict[str, tf.Tensor]],
        ids: List[str],
    ) -> Dict[str, tf.Tensor]:
        """Single training step with gradient updates and micro-batch support.

        Trains on both clean and augmented data variants with micro-batch processing
        for memory efficiency. Applies gradients to all trainable variables
        via AdamW optimizer.

        Args:
            data: Tuple of (X_dict, (Y_tensor,)) where:
                - X_dict: Dictionary with 'clean' and 'augmented' sub-dicts containing facial data
                - Y_tensor: Ground truth gaze points of shape (B, T, 1, 2)

        Returns:
            Dictionary of scalar loss tensor values with keys:
            - '{loss}': Prediction losses
            - 'total-clean': Sum of all clean data losses
            - 'total-augmented': Sum of all augmented data losses
            - 'loss': Total combined loss (used for gradient computation)
        """
        x, y = data
        y = {**x["clean"], **y}
        x_augm = x["augmented"]
        x_clean = x["clean"]
        # Get teacher outputs in inference mode
        teacher_outputs = self._gather_teacher_outputs(x_clean, ids=ids)
        avg_losses = self._train_on_subset(x_augm, y, teacher_outputs)
        return {k: avg_losses[idx] for idx, k in enumerate(self._loss_keys)}

    def fit(
        self, data: Tuple[Dict[str, Dict[str, tf.Tensor]], Tuple[tf.Tensor]]
    ) -> Dict[str, Any]:
        """Perform one training step.

        Args:
            data: Training data tuple containing input tensors and targets

        Returns:
            Dictionary of loss values as numpy arrays
        """
        ids = [x.cache_id for x in self._teachers_models]
        losses = self._train_step(data, ids=ids)
        return Utils.to_numpy(losses)

    def eval(
        self,
        data: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]],
    ) -> Dict[str, float]:
        """Evaluate model performance on validation data.

        Args:
            data: Evaluation data tuple

        Returns:
            Dictionary of evaluation metrics as floats
        """
        losses = self._model_wrapper.evaluate(data)
        return Utils.to_numpy(losses)

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable variables including adapters.

        Returns combined list of:
        - Student model wrapper variables
        - Adapter final variables
        - Adapter intermediate variables
        """
        trainable_vars = self._model_wrapper.trainable_variables
        for adapter in self._adapters.values():
            trainable_vars.extend(adapter.trainable_variables)
        return trainable_vars

    def save(self, folder: str = "models", postfix: str = "") -> None:
        """Save student model weights and adapters to disk.

        Saves the student model wrapper with all components and distillation adapters:
        - GazePredictionModel
        - PredictorBlock
        - EmbeddingsTable
        - EmbeddingsProcessor
        - Distillation adapters (if created)

        Adapters are saved alongside model wrapper components using consistent folder structure.

        Args:
            folder: Directory to save model files (default: "models")
            postfix: Optional suffix for filenames to distinguish different saves
                    (e.g., "best", "latest") (default: "")

        Example:
            >>> trainer.save("checkpoints", postfix="best")
            # Creates: checkpoints/model_name/best/model.npz, predictor.npz, adapters, etc.
        """
        # Save model wrapper (existing code)
        self._model_wrapper.save(folder, postfix)

        # Get checkpoint path using wrapper's helper method
        model_path = self._model_wrapper.get_checkpoint_path(folder, postfix)

        for adapter in self._adapters.values():
            adapter.save_npz(model_path)

    def load(
        self,
        folder: str,
        postfix: str = "",
        embeddings: bool = True,
        force: bool = False,
    ) -> None:
        """Load student model weights and adapters from disk.

        Loads the student model wrapper with optional embeddings table and
        distillation adapters. Handles missing adapters gracefully for backward
        compatibility with old checkpoints (adapters will be created on first training).

        Args:
            folder: Directory containing saved model files
            postfix: Suffix used when files were saved (default: "")
            embeddings: Whether to load EmbeddingsTable weights (default: True)
            force: If False, raise error on incomplete adapters. If True, skip missing.
                   (default: False)

        Example:
            >>> trainer.load("checkpoints", postfix="best", force=False)
            # Loads from: checkpoints/model_name/best/model.npz, predictor.npz, adapters, etc.
        """
        # Load model wrapper
        self._model_wrapper.load(folder, postfix, embeddings, force)

        # Load adapters
        model_path = self._model_wrapper.get_checkpoint_path(folder, postfix)

        for adapter in self._adapters.values():
            adapter.load_npz(model_path)

        logger.info(f"Loaded adapters from {model_path}")
