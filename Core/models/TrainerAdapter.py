"""Adapter for knowledge distillation between teacher and student models.

Handles the adaptation of student model latents to match teacher model latent
spaces through neural network adapters.
"""

from typing import List
import tensorflow as tf
import NN.Utils as NNU

from Core.logging_config import get_logger
from Core.models.PredictionOutputTypes import PredictionOutput
from NN.models.AdapterMLP import AdapterMLP
from NN.models.ResidualAE import ResidualAE
from Core.losses import calculate_losses

logger = get_logger(__name__)


def partial_gradient(v, weight):
    part = weight * v
    return part + tf.stop_gradient(v - part)


class TrainerAdapter:
    """Adapter for knowledge distillation between teacher and student models.

    Manages neural network adapters that transform student model latents to match
    teacher model latent spaces for effective knowledge distillation.
    """

    _latents: AdapterMLP = None
    _reconstruction_nograd: ResidualAE = None
    _reconstruction: ResidualAE = None

    def __init__(self, name, getter):
        super().__init__()
        self._name = name
        self._getter = getter

    @property
    def trainable_variables(self):
        """Get all trainable variables from the adapters."""
        res = []
        for x in [self._latents, self._reconstruction]:
            if x:
                res.extend(x.trainable_variables)
        return res

    def _update_reconstruction_nograd(self):
        for a, b in zip(
            self._reconstruction_nograd.weights, self._reconstruction.weights
        ):
            a.assign(b)

    def _create_if_needed(self, student_dim, teachers_dim):
        if not self._latents:
            self._latents = AdapterMLP(
                teacher_dim=teachers_dim, student_dim=student_dim
            )

        if not self._reconstruction:
            self._reconstruction = ResidualAE(student_dim)
            self._reconstruction_nograd = ResidualAE(student_dim)

    def _normalize(self, v):
        return NNU.normalize_std(v)

    def _calc_regularization(self, v):
        mean = tf.abs(tf.reduce_mean(v, axis=-1))
        mean = tf.maximum(mean - 1.0, 0.0)  # 0..1 - ok
        std = tf.abs(tf.math.reduce_std(v, axis=-1))
        std = tf.maximum(std - 1.0, 0.0)  # 0..1 - ok
        return mean + std

    def _masked_loss(self, ypred, ytrue, mask):
        mask = tf.cast(mask, ypred.dtype)
        N = tf.reduce_sum(mask, axis=-1)
        diff = ypred - ytrue
        loss = tf.square(diff)
        return tf.reduce_sum(loss * mask, axis=-1) / N

    def _mi_loss(self, student_latents, N=10, keep_rate=0.5):
        """
        Calculate mutual information loss using a two-pass reconstruction approach with repulsion.

        This method implements a self-supervised learning strategy that uses masked reconstruction
        followed by repulsion learning to encourage redundancy and robustness in latent representations.
        The approach uses two parallel reconstruction networks to separate learning objectives.

        Args:
            student_latents: Student model latent representations (batch_size, latent_dim)
            N: Number of negative samples to generate per original sample (default: 10)
            keep_rate: Percentage of elements to keep (not mask) during reconstruction (default: 0.75)

        Returns:
            Dictionary containing two loss components:
            - mi_reconstruction: Loss from first pass learning to reconstruct masked features
            - mi_repulsion: Repulsion loss from second pass that removes redundancy

            The combined loss encourages the model to learn robust representations that can
            reconstruct from incomplete information while avoiding over-reliance on specific features.

        Detailed two-pass process:
        1) Initialize reconstruction_nograd as a copy of reconstruction weights
        2) Create N copies of student latents
        3) Create binary mask with keep_rate probability for each element
        4) First pass - reconstruction learning:
           - Use stop_gradient on masked latents to prevent gradient flow
           - Train reconstruction network to recover full latents from masked version
           - Calculate reconstruction loss between predicted and target latents
        5) Update reconstruction_nograd with current reconstruction weights
        6) Second pass - repulsion learning:
           - Use reconstruction_nograd (no gradient updates)
           - Train on masked latents to recover unmasked portions
           - Apply negative loss on unmasked portions to encourage redundancy removal
           - Calculate repulsion loss with reversed sign for gradient maximization

        Mathematical rationale:
        The first pass teaches the model to reconstruct from incomplete information,
        while the second pass actively discourages the model from using predictable
        patterns in unmasked portions. This creates a balance between reconstructive
        capability and redundancy elimination, leading to more robust and generalizable
        latent representations that are less sensitive to individual features.
        """
        student_latents = tf.repeat(student_latents, N, axis=0)
        student_shp = tf.shape(student_latents)

        mask = tf.random.uniform(student_shp) < keep_rate
        masked = tf.where(mask, student_latents, 0.0)
        # first pass to learn masked->full
        reconstructed = self._reconstruction(
            {
                "features": tf.stop_gradient(masked),
                "mask": tf.cast(mask, tf.float32),
            },
            training=True,
        )
        reconstructed = self._normalize(reconstructed)
        loss_reconstruction = self._masked_loss(
            reconstructed, tf.stop_gradient(student_latents), mask=tf.logical_not(mask)
        )
        loss_reconstruction += self._masked_loss(
            reconstructed, tf.stop_gradient(student_latents), mask=mask
        )
        # second pass to remove redundency
        self._update_reconstruction_nograd()
        mask = tf.random.uniform(student_shp) < keep_rate
        weights = tf.linspace(0.0, 10.0, student_shp[-1])[None, None]
        masked = tf.where(mask, partial_gradient(student_latents, weights), 0.0)
        reconstructed_nograd = self._reconstruction_nograd(
            {
                "features": masked,
                "mask": tf.cast(mask, tf.float32),
            },
            training=False,
        )
        reconstructed_nograd = self._normalize(reconstructed_nograd)
        loss_repulsion = self._masked_loss(
            reconstructed_nograd,
            tf.stop_gradient(student_latents),
            mask=tf.logical_not(mask),
        )
        # high reconstruction loss => low repulsion
        # low reconstruction loss => high repulsion
        loss_repulsion = tf.math.log(loss_repulsion + 1e-6)

        return {
            "mi_reconstruction": loss_reconstruction,
            "mi": -1e-2 * loss_repulsion,
        }

    def _distill_loss(self, student_latents, teachers, loss_weights):
        predictions, y_target = {}, {}
        reconstructed_teacher = self._latents(student_latents, training=True)
        start_idx = 0
        for idx, teacher_latent in enumerate(teachers):
            sz = teacher_latent.shape[-1]
            reconstructed = reconstructed_teacher[..., start_idx : start_idx + sz]
            predictions[f"{idx}_teacher"] = self._normalize(reconstructed)
            y_target[f"{idx}_teacher"] = teacher_latent
            start_idx += sz

        # calculate_losses computes loss for all latent keys
        computed_losses = calculate_losses(predictions, y_target, training=True)
        for idx, weight in enumerate(loss_weights):
            computed_losses[f"{idx}_teacher"] *= weight
        return computed_losses

    def calc_loss(
        self,
        student: PredictionOutput,
        teachers: List[PredictionOutput],
        loss_weights: List[tf.Tensor],
    ):
        student_latents = self._normalize(self._getter(student))
        teachers = [self._normalize(self._getter(v)) for v in teachers]
        self._create_if_needed(
            student_dim=student_latents.shape[-1],
            teachers_dim=sum([v.shape[-1] for v in teachers], 0),
        )

        computed_losses = {
            **self._distill_loss(student_latents, teachers, loss_weights),
            **self._mi_loss(student_latents),
            "reg": self._calc_regularization(student_latents),
        }
        return {f"{nm}_{self._name}": v for nm, v in computed_losses.items()}

    def _apply_io(self, f):
        f(self._latents, f"adapter_{self._name}")
        f(self._reconstruction, f"reconstruction_{self._name}")

    def save_npz(self, model_path):
        """Save adapters to NPZ format.

        Args:
            model_path: Base path for saving adapters
        """

        def f(model, name):
            path = f"{model_path}/{name}"
            model.save_npz(path)
            logger.info(f"Saved to {path}.npz")

        self._apply_io(f)

    def load_npz(self, model_path):
        """Load adapters from NPZ format.

        Args:
            model_path: Base path for loading adapters
        """

        def f(model, name):
            path = f"{model_path}/{name}"
            model.load_npz(path, force=True)
            logger.info(f"Loaded from {path}.npz")

        self._apply_io(f)
