"""Contrastive EBM coordinator with shared Direct, Reference and STE pipelines.

Cosine/RMSE energy is in [0,2], additive cosine+RMSE in [0,4].
Inference sums Direct + valid STE without averaging the two paths.
All weights are built before evaluation. NPZ uses Keras variable paths;
component nesting may change paths relative to the former flat model.
"""

from collections.abc import Mapping

import tensorflow as tf

from NN.models.NpzModelMixin import NpzModelMixin
from NN.models.ebm.config import EBMConfig
from NN.models.ebm.encoder import EBMEncoder
from NN.models.ebm.energy import EBMEnergy
from NN.models.ebm.objectives import EBMObjectives
from NN.models.ebm.routing import EBMRouting
from NN.models.ebm.types import ValidatedFeatures


class ContrastiveEBM(NpzModelMixin, tf.keras.Model):
    """Coordinate encoding, energy, routing and training objectives."""

    def __init__(
        self,
        num_systems: int = 4,
        matryoshka_n: int | None = None,
        energy_dim: int | None = None,
        use_pair_mlp: bool = False,
        metric: str = "cosine+rmse",
        settings: Mapping | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._config = EBMConfig.create(
            num_systems, matryoshka_n, energy_dim, use_pair_mlp, metric, settings
        )

    def build(self, input_shapes: list[tf.TensorShape]) -> None:
        """Validate dimensions and build components with one resolved configuration."""
        if len(input_shapes) < 2:
            raise ValueError("Expected [student, teacher_0, teacher_1, ...].")
        input_shapes = [tf.TensorShape(shape) for shape in input_shapes]
        student_shape = input_shapes[0]
        if (
            student_shape.rank != 2
            or student_shape[-1] is None
            or int(student_shape[-1]) < 1
        ):
            raise ValueError(
                "Student input must have shape [batch, known_positive_dim]."
            )
        teacher_dims = []
        for i, shape in enumerate(input_shapes[1:]):
            if shape.rank != 2 or shape[-1] is None or int(shape[-1]) < 1:
                raise ValueError(
                    f"Teacher {i} input must have shape [batch, known_positive_dim]."
                )
            teacher_dims.append(int(shape[-1]))
        self._config = self._config.with_dimensions(
            int(student_shape[-1]), tuple(teacher_dims)
        )
        self._routing = EBMRouting(self._config)
        self._encoder = EBMEncoder(self._config, name="ebm_encoder")
        self._energy = EBMEnergy(self._config, self._routing, name="ebm_energy")
        self._encoder.build(input_shapes)
        self._energy.build(input_shapes)
        self._objectives = EBMObjectives(self._config, self._energy, self._routing)
        super().build(input_shapes)

    def _ensure_built(self, features):
        if not self.built:
            self.build([feature.shape for feature in features])

    def _validate_features(self, features):
        if len(features) != self._config.num_teachers + 1:
            raise ValueError("Unexpected number of teachers.")

        student = features[0]
        batch_size = tf.shape(student)[0]
        tf.debugging.assert_equal(tf.shape(student)[-1], self._config.student_dim)
        teachers = []

        for i, teacher in enumerate(features[1:]):
            tf.debugging.assert_equal(tf.shape(teacher)[0], batch_size)
            tf.debugging.assert_equal(
                tf.shape(teacher)[-1], self._config.teacher_dims[i]
            )
            teachers.append(teacher)

        return ValidatedFeatures(
            student=student, teachers=teachers, batch_size=batch_size
        )

    def _reference_teacher_mask(self, best_index):
        # [B, N]: only omit the tautological comparison of sample b with
        # teacher n when the reference for that SAME sample is teacher n.
        return tf.not_equal(
            best_index[:, None],
            tf.range(self._config.num_teachers, dtype=tf.int32)[None, :] + 1,
        )

    def _reference_training_mask(self, best_index):
        """Valid Reference/STE objectives: teacher reference, excluding its own pair.

        [B,N] mask; when best_index[b] == 0 the entire Reference loss for
        sample b is disabled to avoid duplicating the Direct objective.
        The same mask also prevents STE from duplicating Direct for best_index=0.
        """
        return tf.logical_and(
            self._reference_teacher_mask(best_index), best_index[:, None] > 0
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def learn(
        self,
        features: list[tf.Tensor],
        temperature: float = 0.1,
        *,
        best_index: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Return four losses, mean pair order and mean usage for each system."""
        self._ensure_built(features)
        validated = self._validate_features(features)
        raw_student = validated.student
        raw_teachers = validated.teachers
        batch = validated.batch_size
        tf.debugging.assert_greater(batch, 1)
        tf.debugging.assert_positive(temperature)
        teacher_base = self._encoder.teacher_forward(raw_teachers, training=True)
        student_base = self._encoder.student_forward(raw_student, training=True)
        selection = self._encoder.best_reference(student_base, teacher_base, best_index)
        best_index = selection.best_index
        systems = self._encoder.system_levels(
            student_base, teacher_base, best_index, training=True
        )
        students_direct = systems.student
        references = systems.reference
        teachers = systems.teachers

        # All three paths share the SAME independent system heads, Pair MLP,
        # Matryoshka fractions and Contrastive + Routing + Calibration pipeline.
        # STE is constructed AFTER the system heads: forward=Reference,
        # gradient wrt Student=identity. Reference losses are excluded when
        # best_index == 0 (identical to Direct); STE is also masked there,
        # excluding the selected teacher and disabling STE if Reference is student.
        students_ste = self._encoder.ste_student(students_direct, references)
        direct_per_fraction = self._energy.levels(
            teachers, students_direct, training=True, all_pairs=True
        )
        reference_per_fraction = self._energy.levels(
            teachers, references, training=True, all_pairs=True
        )
        reference_valid = self._reference_training_mask(best_index)
        ste_per_fraction = None
        if self._config.ste_training_weight > 0.0:
            # Freeze ONLY the STE teacher-side tensor. STE still updates the
            # student encoder, shared system heads and Pair MLP via student input.
            # Direct and Reference retain their original teacher gradients.
            ste_per_fraction = self._energy.levels(
                tf.stop_gradient(teachers), students_ste, training=True, all_pairs=True
            )
        return self._objectives.compute(
            systems,
            teacher_base,
            best_index,
            direct_per_fraction,
            reference_per_fraction,
            ste_per_fraction,
            reference_valid,
            temperature,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def call(
        self,
        features: list[tf.Tensor],
        training: bool = False,
        *,
        best_index: tf.Tensor,
    ) -> tf.Tensor:
        """Return mean-prefix Direct + valid STE energy [B,N]."""
        self._ensure_built(features)
        validated = self._validate_features(features)
        raw_student = validated.student
        raw_teachers = validated.teachers
        batch = validated.batch_size
        teacher_base = self._encoder.teacher_forward(raw_teachers, training=training)
        student_base = self._encoder.student_forward(raw_student, training=training)
        selection = self._encoder.best_reference(student_base, teacher_base, best_index)
        best_index = selection.best_index
        systems = self._encoder.system_levels(
            student_base, teacher_base, best_index, training=training
        )
        student_direct = systems.student
        reference = systems.reference
        teachers = systems.teachers
        # Same STE as in learn(); this preserves gradient through
        # the real student while the forward value equals the selected reference.
        student_ste = self._encoder.ste_student(student_direct, reference)
        # Evaluate STE and Direct together for all Matryoshka prefixes.
        # Pack paths along batch: [B,2,S,D] -> [2B,S,D]; one system-head
        # result and one batched Pair-MLP call for each independent system.
        student_paths = tf.reshape(
            tf.stack([student_ste, student_direct], axis=1),
            [2 * batch, self._config.num_systems, self._config.energy_dim],
        )
        teacher_paths = tf.reshape(
            tf.broadcast_to(
                teachers[:, None, ...],
                [
                    batch,
                    2,
                    self._config.num_systems,
                    self._config.num_teachers,
                    self._config.energy_dim,
                ],
            ),
            [
                2 * batch,
                self._config.num_systems,
                self._config.num_teachers,
                self._config.energy_dim,
            ],
        )
        levels = self._energy.levels(
            teacher_paths, student_paths, training=training, all_pairs=False
        )
        combined = self._energy.combine_systems(levels)  # [F, 2B, N]
        combined = tf.reshape(
            combined,
            [len(self._config.matryoshka_dims), batch, 2, self._config.num_teachers],
        )
        # First average over Matryoshka fractions, then separate STE / Direct.
        ste_energy, direct_energy = tf.unstack(
            tf.reduce_mean(combined, axis=0), axis=1
        )  # each [B,N]
        # NEVER divide the total Energy: an average artificially halves the
        # useful gradient. Exclude STE entirely when Reference is the student
        # (duplicate Direct) or the compared teacher (trivial self-match).
        valid_ste = self._reference_training_mask(best_index)  # [B,N], graph/XLA-safe
        ste_energy = tf.where(valid_ste, ste_energy, tf.zeros_like(ste_energy))
        result = (
            direct_energy + ste_energy
        )  # [B,N]; 1 or 2 nontrivial Energy contributions
        tf.debugging.assert_equal(tf.shape(result), [batch, self._config.num_teachers])
        return result
