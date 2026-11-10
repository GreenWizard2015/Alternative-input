"""Adapter for knowledge distillation between teacher and student models.

Handles the adaptation of student model latents to match teacher model latent
spaces through neural network adapters.
"""

from typing import List
import tensorflow as tf
import NN.Utils as NNU
from NN.Utils import flatten
import Core.losses as L

from Core.logging_config import get_logger
from NN.models.ContrastiveEBM import ContrastiveEBM


logger = get_logger(__name__)


class TrainerAdapter:
    """Adapter for knowledge distillation between teacher and student models.

    Manages neural network adapters that transform student model latents to match
    teacher model latent spaces for effective knowledge distillation.
    """

    _latents: ContrastiveEBM = None
    _latents_nograd: ContrastiveEBM = None

    def __init__(self, name):
        super().__init__()
        self.name = name

    @property
    def trainable_variables(self):
        """Get all trainable variables from the adapters."""
        res = []
        parts = [self._latents]
        for x in parts:
            res.extend(x.trainable_variables)
        return res

    def _assign_weights(self, A, B):
        for a, b in zip(A.weights, B.weights):
            a.assign(b)

    def _update_latents_nograd(self):
        self._assign_weights(self._latents_nograd, self._latents)
        pass

    def _create_if_needed(self, student_dim):
        if not self._latents:
            self._latents = ContrastiveEBM(name="latents")
            self._latents_nograd = ContrastiveEBM(name="latents_nograd")
            self._latents_nograd.trainable = False

    def _calc_regularization(self, v, scale):
        mean = tf.abs(tf.reduce_mean(v, axis=-1))
        mean = tf.maximum(mean - 1.0, 0.0)  # 0..1 - ok
        std = tf.abs(tf.math.reduce_std(v, axis=-1))
        std = tf.maximum(std - 1.0, 0.0)  # 0..1 - ok
        loss = tf.reduce_mean(mean + std)
        # stabilize training
        loss /= tf.stop_gradient(tf.where(1.0 < loss, loss, 1.0))
        return {"reg": loss * scale}

    def _latents_sgld(self, teachers, loss_weights):
        def ebm(z, mask):
            flat_mask = tf.reshape(mask, (-1,))
            shp = tf.shape(loss_weights)
            N = shp[-1]
            null_energy = tf.fill(shp, float("inf"))
            null_energy = tf.reshape(null_energy, (-1, N))
            w = tf.reshape(tf.boolean_mask(loss_weights, mask), (-1, N))

            inputs = [z] + teachers
            energy = self._latents_nograd(
                flatten([tf.boolean_mask(x, mask) for x in inputs]),
                training=False,
            )

            energy = tf.reshape(energy, tf.shape(w))
            idx = tf.where(flat_mask)
            energy = tf.tensor_scatter_nd_update(null_energy, idx, energy * w)
            return tf.reshape(energy, shp)

        def F(z):
            return NNU.find_optimal_latents_sgld(
                energy_model=ebm, z=z, max_steps=10, delta=1e-5
            )

        return F

    def _latents_adam(self, teachers, loss_weights):
        def ebm(z, mask):
            tf.debugging.assert_rank_at_least(z, 2)
            tf.debugging.assert_type(mask, tf.bool)
            tf.debugging.assert_equal(tf.shape(z)[:-1], tf.shape(mask))

            extra_rank = tf.rank(z) - tf.rank(loss_weights)
            tf.debugging.assert_greater_equal(extra_rank, 0)
            extra_shape = tf.shape(z)[:extra_rank]

            weights, *expanded_teachers = [
                tf.broadcast_to(x, tf.concat([extra_shape, tf.shape(x)], axis=0))
                for x in [loss_weights] + teachers
            ]

            tf.debugging.assert_equal(tf.shape(weights)[:-1], tf.shape(mask))
            for teacher in expanded_teachers:
                tf.debugging.assert_equal(tf.shape(teacher)[:-1], tf.shape(mask))

            active_z = tf.boolean_mask(z, mask)
            active_weights = tf.boolean_mask(weights, mask)
            active_teachers = [tf.boolean_mask(x, mask) for x in expanded_teachers]

            tf.debugging.assert_rank(active_z, 2)
            tf.debugging.assert_rank(active_weights, 2)
            tf.debugging.assert_equal(
                tf.shape(active_z)[0], tf.shape(active_weights)[0]
            )
            tf.debugging.assert_greater(tf.shape(active_z)[0], 0)

            energy = self._latents_nograd(
                flatten([active_z] + active_teachers),
                training=False,
            )

            tf.debugging.assert_equal(tf.shape(energy), tf.shape(active_weights))
            tf.debugging.assert_all_finite(energy, "Non-finite EBM energy.")

            energy *= tf.cast(active_weights, energy.dtype)

            full_energy = tf.fill(tf.shape(weights), tf.cast(1e9, energy.dtype))
            full_energy = tf.tensor_scatter_nd_update(
                full_energy, tf.where(mask), energy
            )

            tf.debugging.assert_equal(tf.shape(full_energy), tf.shape(weights))
            tf.debugging.assert_all_finite(full_energy, "Non-finite masked EBM energy.")
            return full_energy

        def F(z):
            tf.debugging.assert_rank_at_least(z, 2)
            result = NNU.find_optimal_latents_adam(
                energy_model=ebm,
                z=z,
                max_steps=8,
                num_candidates=8,
                step_size=0.003,
                step_decay=0.97,
                max_distance=20.0,
                initial_noise=0.005,
                delta=0e-5,
            )
            tf.debugging.assert_equal(tf.shape(result["z"]), tf.shape(z))
            tf.debugging.assert_equal(tf.shape(result["energy"]), tf.shape(z)[:-1])
            tf.debugging.assert_equal(tf.shape(result["improvement"]), tf.shape(z)[:-1])
            tf.debugging.assert_equal(tf.shape(result["distance"]), tf.shape(z)[:-1])
            tf.debugging.assert_all_finite(result["z"], "Non-finite optimized latents.")
            tf.debugging.assert_all_finite(
                result["energy"], "Non-finite optimized energy."
            )
            tf.debugging.assert_all_finite(
                result["improvement"], "Non-finite latent improvement."
            )
            tf.debugging.assert_all_finite(
                result["distance"], "Non-finite latent distance."
            )
            return result

        return F

    def _distill_loss(
        self,
        student_latents,
        teachers,
        loss_weights,
        best_index,
        scale,
    ):
        self._update_latents_nograd()

        w = flatten([loss_weights])[0]
        best_index = tf.reshape(best_index, (-1,))
        losses = self._latents.learn(
            flatten([tf.stop_gradient(student_latents)] + teachers),
            best_index=best_index,
        )
        use_energy_distill = True
        if use_energy_distill:
            energy = self._latents_nograd(
                flatten([student_latents] + teachers),
                training=False,
                best_index=best_index,
            )
            tf.debugging.assert_equal(tf.shape(energy), tf.shape(w))
            losses["distil"] = tf.reduce_sum(energy * w, axis=-1) * scale
        else:
            target_latents = self._latents_adam(teachers, loss_weights)(
                tf.stop_gradient(student_latents)
            )

            losses["search_improvement"] = target_latents["improvement"]
            losses["search_distance"] = target_latents["distance"]

            fraction = 1.0
            target = tf.stop_gradient(
                student_latents + fraction * (target_latents["z"] - student_latents)
            )
            latents_loss = L.huber_loss(target, student_latents)
            losses["distil"] = tf.reduce_mean(latents_loss) * scale * 0.01

        return losses

    def calc_loss(
        self,
        student_latents: tf.Tensor,
        teachers: List[tf.Tensor],
        loss_weights: tf.Tensor,
        best_index,
        scale: float,
    ):
        self._create_if_needed(student_latents.shape[-1])

        computed_losses = {
            **self._distill_loss(
                student_latents,
                teachers,
                loss_weights,
                best_index,
                scale,
            ),
            # **self._calc_regularization(student_latents, scale=scale * 1e-4),
        }
        return {f"{nm}_{self.name}": v for nm, v in computed_losses.items()}

    def _apply_io(self, f):
        f(self._latents, f"adapter_{self.name}")
        # f(self._pos, f"adapter_pos_{self.name}")
        pass

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
