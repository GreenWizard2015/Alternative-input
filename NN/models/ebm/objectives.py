"""Common contrastive objectives and fraction/path metrics."""

import tensorflow as tf

from NN.models.EBMMetric import create_metric, get_metric_defaults
from NN.models.ebm.config import EBMConfig
from NN.models.ebm.energy import EBMEnergy
from NN.models.ebm.routing import EBMRouting
from NN.models.ebm.types import (
    CalibrationLoss,
    LevelLosses,
    PathMetrics,
    SystemLevels,
    TrainingLevel,
)


class EBMObjectives:
    """Compute all losses with the same pipeline for Direct, Reference and STE."""

    def __init__(
        self, config: EBMConfig, energy: EBMEnergy, routing: EBMRouting
    ) -> None:
        self._config = config
        self._energy = energy
        self._routing = routing
        self._metric = create_metric(config.metric)
        defaults = get_metric_defaults(config.metric)
        self._positive_target = float(defaults.positive_target)
        self._negative_margin = float(defaults.negative_margin)

    def _cross_system_loss(self, students):
        """Contrast each sample's distinct DIRECT system heads, not an STE anchor."""
        if self._config.num_systems <= 1:
            return tf.zeros([], dtype=students.dtype)
        cross = self._energy.cross_system_energy(students)
        mask = tf.logical_not(tf.eye(self._config.num_systems, dtype=tf.bool))[
            None, :, :
        ]
        violation = tf.square(
            tf.nn.relu(tf.cast(self._negative_margin, cross.dtype) - cross)
        )
        count = tf.cast(
            tf.shape(cross)[0]
            * self._config.num_systems
            * (self._config.num_systems - 1),
            cross.dtype,
        )
        return tf.math.divide_no_nan(
            tf.reduce_sum(tf.where(mask, violation, tf.zeros_like(violation))), count
        )

    def _alignment_loss(self, student, reference, best_index):
        # Same systems and Matryoshka prefixes for student and selected reference.
        # Stop gradients into the reference only for this alignment objective.
        # Self-selected anchors are excluded (student == reference).
        per_fraction = []
        for dim in self._config.matryoshka_dims:
            aligned = self._metric.alignment(
                student[..., :dim], tf.stop_gradient(reference[..., :dim])
            )
            per_fraction.append(tf.reduce_mean(aligned, axis=1))  # [B]
        per_sample = sum(per_fraction, 0.0) / float(len(self._config.matryoshka_dims))
        teacher_selected = tf.cast(best_index > 0, per_sample.dtype)
        return tf.math.divide_no_nan(
            tf.reduce_sum(per_sample * teacher_selected),
            tf.reduce_sum(teacher_selected),
        )

    def _calibration_loss(self, selected_energy, valid_teacher=None):
        """Same calibration for Direct and Reference at ALL fractions.

        Exclude tautological Reference==Teacher positives, plus negatives for
        the corresponding invalid reference/teacher anchor pairs.
        """
        tf.debugging.assert_rank(selected_energy, 3)
        batch_size = tf.shape(selected_energy)[0]
        positive = tf.einsum("ii...->i...", selected_energy)
        if valid_teacher is None:
            valid_teacher = tf.ones(tf.shape(positive), dtype=tf.bool)
        valid = tf.cast(valid_teacher, selected_energy.dtype)
        positive_penalty = tf.square(
            tf.nn.relu(positive - tf.cast(self._positive_target, selected_energy.dtype))
        )
        negative_penalty = tf.square(
            tf.nn.relu(
                tf.cast(self._negative_margin, selected_energy.dtype) - selected_energy
            )
        )
        positive_loss = tf.math.divide_no_nan(
            tf.reduce_sum(positive_penalty * valid), tf.reduce_sum(valid)
        )
        off_diagonal = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))[:, :, None]
        negative_mask = tf.logical_and(off_diagonal, valid_teacher[None, :, :])
        negative_loss = tf.math.divide_no_nan(
            tf.reduce_sum(
                tf.where(
                    negative_mask, negative_penalty, tf.zeros_like(negative_penalty)
                )
            ),
            tf.cast(batch_size - 1, selected_energy.dtype) * tf.reduce_sum(valid),
        )
        return CalibrationLoss(
            total=positive_loss + negative_loss,
            positive=positive_loss,
            negative=negative_loss,
        )

    def _repulsion_loss(self, teachers, students):
        # [N,B,D] teacher groups and [S,B,D] student groups.
        teacher_groups = tf.transpose(teachers, [1, 0, 2])
        student_groups = tf.transpose(students, [1, 0, 2])
        return 0.5 * (
            self._metric.repulsion(teacher_groups)
            + self._metric.repulsion(student_groups)
        )

    def _contrastive(self, energy, temperature, valid_teacher=None):
        """Shared bidirectional contrastive loss, with optional self-pair mask.

        E[i,j,n] is teacher i against anchor j. Only positive anchor-teacher
        pairs marked valid can participate in the corresponding row/column
        softplus comparisons. Direct uses all pairs; Reference masks exact
        self-comparisons when it selects the same teacher.
        """
        batch_size = tf.shape(energy)[0]
        dtype = energy.dtype
        temperature = tf.cast(temperature, dtype)
        positive = tf.einsum("ii...->i...", energy)
        row_gap = (energy - positive[:, None, :]) / temperature
        col_gap = (energy - positive[None, :, :]) / temperature
        off_diagonal = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))[:, :, None]
        if valid_teacher is None:
            valid_teacher = tf.ones(tf.shape(positive), dtype=tf.bool)
        row_mask = tf.logical_and(off_diagonal, valid_teacher[:, None, :])
        col_mask = tf.logical_and(off_diagonal, valid_teacher[None, :, :])
        row_loss = tf.where(row_mask, tf.nn.softplus(-row_gap), tf.zeros_like(row_gap))
        col_loss = tf.where(col_mask, tf.nn.softplus(-col_gap), tf.zeros_like(col_gap))
        count = (
            2.0
            * tf.cast(batch_size - 1, dtype)
            * tf.reduce_sum(tf.cast(valid_teacher, dtype))
        )
        return tf.math.divide_no_nan(tf.reduce_sum(row_loss + col_loss), count)

    def _valid_pair_order(self, energy, valid_teacher=None):
        """Pair order for Direct or masked Reference/STE, macro across teachers."""
        batch = tf.shape(energy)[0]
        positives = tf.einsum("ii...->i...", energy)
        off = tf.logical_not(tf.eye(batch, dtype=tf.bool))[:, :, None]
        if valid_teacher is None:
            valid_teacher = tf.ones(tf.shape(positives), dtype=tf.bool)
        row = tf.logical_and(off, valid_teacher[:, None, :])
        col = tf.logical_and(off, valid_teacher[None, :, :])
        hits = tf.reduce_sum(
            tf.cast(tf.logical_and(row, energy > positives[:, None, :]), tf.float32)
        )
        hits += tf.reduce_sum(
            tf.cast(tf.logical_and(col, energy > positives[None, :, :]), tf.float32)
        )
        count = (
            2.0
            * tf.cast(batch - 1, tf.float32)
            * tf.reduce_sum(tf.cast(valid_teacher, tf.float32))
        )
        return tf.stop_gradient(tf.math.divide_no_nan(hits, count))

    def _training_level(self, per_system, temperature, valid_teacher=None):
        """One objective pipeline shared by Direct, Reference and STE.

        All paths use identical system aggregation, contrastive, routing,
        and calibration. Reference and STE mask exact teacher self-pairs.
        """
        energy = self._energy.combine_systems(per_system)
        scores = self._routing.scores(per_system, temperature)
        routing = self._routing.statistics(scores, valid_teacher)
        calibration = self._calibration_loss(energy, valid_teacher)
        return TrainingLevel(
            contrastive=self._contrastive(energy, temperature, valid_teacher),
            balance=routing.balance,
            entropy=routing.entropy,
            calibration=calibration.total,
            energy=energy,
            usage_sum=routing.usage_sum,
            usage_count=routing.usage_count,
        )

    @staticmethod
    def _mean_fraction_metrics(values):
        """Arithmetic mean across Matryoshka fractions (no tf.add_n)."""
        return (
            sum(values, 0.0) / float(len(values))
            if values
            else tf.zeros([], tf.float32)
        )

    def _average_path_metrics(self, orders, usage_sum, usage_count, enabled=True):
        """Mean fractions within a path; omit disabled/empty paths in cross-path means."""
        return PathMetrics(
            pair_order=self._mean_fraction_metrics(orders),
            system_usage=tf.math.divide_no_nan(usage_sum, usage_count),
            active=tf.cast(usage_count > 0, tf.float32) * float(enabled),
        )

    def compute(
        self,
        systems: SystemLevels,
        teacher_base: tf.Tensor,
        best_index: tf.Tensor,
        direct_per_fraction: tf.Tensor,
        reference_per_fraction: tf.Tensor,
        ste_per_fraction: tf.Tensor | None,
        reference_valid: tf.Tensor,
        temperature: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Average losses and diagnostics over prefixes and valid training paths."""
        students_direct = systems.student
        align = self._alignment_loss(students_direct, systems.reference, best_index)
        repulsion = self._repulsion_loss(teacher_base, students_direct)
        cross_system = self._cross_system_loss(students_direct)
        ste_valid = reference_valid
        direct_losses = LevelLosses.zeros()
        reference_losses = LevelLosses.zeros()
        ste_losses = LevelLosses.zeros()
        direct_orders, reference_orders, ste_orders = [], [], []
        direct_usage_sum = tf.zeros([self._config.num_systems], tf.float32)
        reference_usage_sum = tf.zeros_like(direct_usage_sum)
        ste_usage_sum = tf.zeros_like(direct_usage_sum)
        direct_usage_count = tf.zeros([], tf.float32)
        reference_usage_count = tf.zeros([], tf.float32)
        ste_usage_count = tf.zeros([], tf.float32)
        for index in range(len(self._config.matryoshka_dims)):
            direct_level = self._training_level(direct_per_fraction[index], temperature)
            reference_level = self._training_level(
                reference_per_fraction[index], temperature, reference_valid
            )
            direct_losses = direct_losses.add_level(direct_level)
            reference_losses = reference_losses.add_level(reference_level)
            direct_orders.append(self._valid_pair_order(direct_level.energy))
            reference_orders.append(
                self._valid_pair_order(reference_level.energy, reference_valid)
            )
            direct_usage_sum += direct_level.usage_sum
            direct_usage_count += direct_level.usage_count
            reference_usage_sum += reference_level.usage_sum
            reference_usage_count += reference_level.usage_count
            if self._config.ste_training_weight > 0.0:
                assert (
                    ste_per_fraction is not None
                ), "Enabled STE requires its energies."
                ste_level = self._training_level(
                    ste_per_fraction[index], temperature, ste_valid
                )
                ste_losses = ste_losses.add_level(ste_level)
                ste_orders.append(self._valid_pair_order(ste_level.energy, ste_valid))
                ste_usage_sum += ste_level.usage_sum
                ste_usage_count += ste_level.usage_count
        scale = 1.0 / float(len(self._config.matryoshka_dims))
        direct_losses = direct_losses.scaled(scale)
        reference_losses = reference_losses.scaled(scale)
        ste_losses = ste_losses.scaled(scale)
        contrastive = (
            direct_losses.contrastive
            + self._config.reference_teacher_weight * reference_losses.contrastive
            + self._config.ste_training_weight * ste_losses.contrastive
            + self._config.cross_system_weight * cross_system
            + self._config.alignment_weight * align
        )
        balance = (
            direct_losses.balance
            + self._config.reference_teacher_weight * reference_losses.balance
            + self._config.ste_training_weight * ste_losses.balance
        )
        entropy = (
            direct_losses.entropy
            + self._config.reference_teacher_weight * reference_losses.entropy
            + self._config.ste_training_weight * ste_losses.entropy
        )
        calibration = (
            direct_losses.calibration
            + self._config.reference_teacher_weight * reference_losses.calibration
            + self._config.ste_training_weight * ste_losses.calibration
        )
        # Each path is averaged across ALL Matryoshka fractions.
        # Cross-path statistics are equal-weight means of Direct/Reference/STE,
        # excluding paths disabled by settings or empty after self-pair masks.
        metrics_direct = self._average_path_metrics(
            direct_orders, direct_usage_sum, direct_usage_count
        )
        metrics_reference = self._average_path_metrics(
            reference_orders,
            reference_usage_sum,
            reference_usage_count,
            enabled=self._config.reference_teacher_weight > 0.0,
        )
        metrics_ste = self._average_path_metrics(
            ste_orders,
            ste_usage_sum,
            ste_usage_count,
            enabled=self._config.ste_training_weight > 0.0,
        )
        active_paths = sum(
            (
                metrics_direct.active,
                metrics_reference.active,
                metrics_ste.active,
            ),
            0.0,
        )
        mean_pair_order = tf.math.divide_no_nan(
            sum(
                (
                    metrics_direct.pair_order * metrics_direct.active,
                    metrics_reference.pair_order * metrics_reference.active,
                    metrics_ste.pair_order * metrics_ste.active,
                ),
                0.0,
            ),
            active_paths,
        )
        usage = tf.math.divide_no_nan(
            sum(
                (
                    metrics_direct.system_usage * metrics_direct.active,
                    metrics_reference.system_usage * metrics_reference.active,
                    metrics_ste.system_usage * metrics_ste.active,
                ),
                0.0,
            ),
            active_paths,
        )
        routing = (
            self._config.balance_weight * balance
            + self._config.routing_entropy_weight * entropy
        )
        for value, message in (
            (contrastive, "contrastive"),
            (repulsion, "repulsion"),
            (routing, "routing"),
            (calibration, "calibration"),
        ):
            tf.debugging.assert_all_finite(value, message)
        output = {
            "ebm_loss_contrastive": contrastive,
            "ebm_loss_repulsion": repulsion,
            "ebm_loss_balance": routing,
            "ebm_loss_calibration": self._config.calibration_weight * calibration,
            "ebm_metrics_pair_order": tf.stop_gradient(mean_pair_order),
        }
        output.update(
            {
                f"ebm_metrics_system_usage_{i}": tf.stop_gradient(usage[i])
                for i in range(self._config.num_systems)
            }
        )
        return output
