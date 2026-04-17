"""Adapter for knowledge distillation between teacher and student models.

Handles the adaptation of student model latents to match teacher model latent
spaces through neural network adapters.
"""

import tensorflow as tf
import NN.Utils as NNU

from Core.logging_config import get_logger
from Core.models.PredictionOutputTypes import PredictionOutput
from NN.models.AdapterMLP import AdapterMLP
from Core.losses import calculate_losses

logger = get_logger(__name__)


class TrainerAdapter:
    """Adapter for knowledge distillation between teacher and student models.

    Manages neural network adapters that transform student model latents to match
    teacher model latent spaces for effective knowledge distillation.
    """

    _intermediate: AdapterMLP = None
    _final: AdapterMLP = None

    def _create_adapters_if_needed(
        self, student_latent_dim, teacher_latent_dim
    ) -> None:
        if self._intermediate:
            return
        self._intermediate = AdapterMLP(
            student_dim=student_latent_dim,
            teacher_dim=teacher_latent_dim,
            name="DistillationAdapterIntermediate",
        )

        self._final = AdapterMLP(
            student_dim=student_latent_dim,
            teacher_dim=teacher_latent_dim,
            name="DistillationAdapterFinal",
        )

        logger.info(
            f"Created adapters: student_dim={student_latent_dim}, "
            f"teacher_dim={teacher_latent_dim}"
        )

    def _preprocess_student_outputs(self, student: PredictionOutput):
        """Preprocess student outputs by projecting to teacher dimension space via adapters."""
        # Project student latents to teacher dimension space via adapters
        adapted_to_teacher_intermediate = self._intermediate(
            NNU.normalize_std(student.intermediate_latents),
            training=True,
        )
        adapted_to_teacher_final = self._final(
            NNU.normalize_std(student.latents),
            training=True,
        )
        return (
            NNU.normalize_std(adapted_to_teacher_intermediate),
            NNU.normalize_std(adapted_to_teacher_final),
        )

    def _preprocess_teacher_outputs(self, teacher: PredictionOutput):
        """Preprocess teacher outputs by normalizing latents."""
        return (
            NNU.normalize_std(teacher.intermediate_latents),
            NNU.normalize_std(teacher.latents),
        )

    def _process_distil(self, student, teacher):
        """Process student and teacher outputs through adapters.

        Args:
            student: Student model prediction output
            teacher: Teacher model prediction output

        Returns:
            Tuple of (processed_student_outputs, processed_teacher_outputs)
        """
        self._create_adapters_if_needed(
            student_latent_dim=student.latents.shape[-1],
            teacher_latent_dim=teacher.latents.shape[-1],
        )
        return (
            self._preprocess_student_outputs(student),
            self._preprocess_teacher_outputs(teacher),
        )

    @property
    def trainable_variables(self):
        """Get all trainable variables from the adapters."""
        res = []
        for x in [
            self._intermediate,
            self._final,
        ]:
            if x:
                res.extend(x.trainable_variables)
        return res

    def _add_distil(
        self,
        student: PredictionOutput,
        teacher: PredictionOutput,
        predictions,
        y,
        target,
    ):
        adapted_l, teacher_l = self._process_distil(student=student, teacher=teacher)
        adapted_to_teacher_intermediate, adapted_to_teacher_final = adapted_l
        teacher_intermediate_latents, teacher_latents = teacher_l

        return (
            {
                **predictions,
                "teacher_result": teacher.result,
                # student->teacher
                "adapted_intermediate": adapted_to_teacher_intermediate,
                "adapted_final": adapted_to_teacher_final,
            },
            {
                **y,
                "teacher_result": target,
                # student->teacher
                "adapted_intermediate": teacher_intermediate_latents,
                "adapted_final": teacher_latents,
            },
        )

    def calc_loss(
        self,
        student: PredictionOutput,
        teacher: PredictionOutput,
        target,
        target_student_loss,
        loss_weight,
    ):
        predictions, y_target = {}, {}
        predictions, y_target = self._add_distil(
            student=student,
            teacher=teacher,
            predictions=predictions,
            y=y_target,
            target=target,
        )

        # ===== Compute Main Losses (Task + Feature Matching) =====
        # calculate_losses computes loss for all keys, including latent keys
        computed_losses = calculate_losses(predictions, y_target, training=True)

        aux_losses = set(computed_losses.keys())
        aux_losses.discard("teacher_result")

        teacher_mask = tf.stop_gradient(
            tf.where(computed_losses["teacher_result"] < target_student_loss, 1.0, 0.0)
        )
        teacher_acc = tf.reduce_mean(teacher_mask)
        computed_losses["teacher_acc"] = teacher_acc
        loss_weight *= teacher_mask + teacher_acc + 1.0

        loss_weight = loss_weight / float(len(aux_losses))
        for name in aux_losses:
            computed_losses[name] *= loss_weight

        return computed_losses

    def save_npz(self, model_path, idx):
        """Save adapters to NPZ format.

        Args:
            model_path: Base path for saving adapters
            idx: Index for this adapter (for multi-teacher setups)
        """

        def save(model, name):
            if model:
                path = f"{model_path}/{name}-{idx}"
                model.save_npz(path)
                logger.info(f"Saved to {path}.npz")

        save(self._intermediate, "adapter_intermediate")
        save(self._final, "adapter_final")

    def load_npz(self, model_path, idx):
        """Load adapters from NPZ format.

        Args:
            model_path: Base path for loading adapters
            idx: Index for this adapter (for multi-teacher setups)
        """

        def load(model, name):
            if model:
                path = f"{model_path}/{name}-{idx}"
                model.load_npz(path, force=True)
                logger.info(f"Saved to {path}.npz")

        load(self._intermediate, "adapter_intermediate")
        load(self._final, "adapter_final")
