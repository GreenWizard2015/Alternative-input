"""Shared student, teacher and system projections."""

import tensorflow as tf

from NN.layers.BranchedMLP import BranchedMLP
from NN.models.EBMMetric import create_metric
from NN.models.ebm.config import EBMConfig
from NN.models.ebm.types import ReferenceSelection, SystemLevels


class EBMEncoder(tf.keras.layers.Layer):
    """Own all projections and preserve reference/STE gradient boundaries."""

    def __init__(self, config: EBMConfig, **kwargs) -> None:
        kwargs["name"] = kwargs.get("name") or "ebm_encoder"
        super().__init__(**kwargs)
        self._config = config
        self._metric = create_metric(config.metric)

    def build(self, input_shapes: list[tf.TensorShape]) -> None:
        """Create and build every projection before encoding."""
        self._teacher_projectors = []

        for i, shape in enumerate(input_shapes[1:]):
            projector = BranchedMLP(
                sizes=[self._config.energy_dim],
                branches=1,
                activation="linear",
                last_activation="linear",
                name=f"teacher_projector_{i}",
            )
            projector.build(shape)
            self._teacher_projectors.append(projector)
            setattr(self, f"_teacher_projector_{i}", projector)

        hidden_dim = min(
            512, max(self._config.student_dim, self._config.energy_dim * 2)
        )
        self._student_encoder = BranchedMLP(
            sizes=[hidden_dim, hidden_dim, self._config.energy_dim],
            branches=1,
            activation="relu",
            last_activation="linear",
            name="student_encoder",
        )
        # S independent heads; the same heads process student and all teachers.
        self._system_heads = BranchedMLP(
            sizes=[self._config.energy_dim * 2, self._config.energy_dim],
            branches=self._config.num_systems,
            activation="relu",
            last_activation="linear",
            dropout=0.0,
            name="system_heads",
        )
        self._student_encoder.build(input_shapes[0])
        self._system_heads.build(
            tf.TensorShape(
                [None, self._config.num_teachers + 1, self._config.energy_dim]
            )
        )

        super().build(input_shapes)

    def teacher_forward(self, teachers: list[tf.Tensor], training: bool) -> tf.Tensor:
        """Project teacher inputs into prepared base embeddings [B,N,D]."""
        return tf.stack(
            [
                self._metric.prepare(projector(teacher, training=training))
                for projector, teacher in zip(self._teacher_projectors, teachers)
            ],
            axis=1,
        )

    def student_forward(self, student: tf.Tensor, training: bool) -> tf.Tensor:
        """Encode the student input into a prepared base embedding [B,D]."""
        return self._metric.prepare(self._student_encoder(student, training=training))

    def ste_student(self, student: tf.Tensor, reference: tf.Tensor) -> tf.Tensor:
        """Use reference values with an identity gradient only to the student."""
        # Forward value == reference, but d(output)/d(student) == identity.
        # The STE itself contributes ZERO gradient to reference; reference-based
        # contrastive and reference-teacher objectives still differentiate reference.
        return student + tf.stop_gradient(reference - student)

    def system_levels(
        self,
        student_base: tf.Tensor,
        teacher_base: tf.Tensor,
        best_index: tf.Tensor,
        training: bool,
    ) -> SystemLevels:
        """One shared system-head call; STE is created afterwards by learn()/call().

        Returns real student [B,S,D], selected reference [B,S,D]
        and teachers [B,S,N,D]. Both learn() and call() construct STE afterwards.
        """
        entities = tf.concat([student_base[:, None, :], teacher_base], axis=1)
        systems = self._metric.prepare(self._system_heads(entities, training=training))
        if self._config.num_systems == 1:
            systems = systems[:, :, None, :]
        student = systems[:, 0, :, :]
        reference = tf.gather(systems, best_index, axis=1, batch_dims=1)
        teachers = tf.transpose(systems[:, 1:, :, :], [0, 2, 1, 3])
        return SystemLevels(student=student, reference=reference, teachers=teachers)

    def best_reference(
        self,
        student: tf.Tensor,
        teachers: tf.Tensor,
        best_index: tf.Tensor,
    ) -> ReferenceSelection:
        """Validate per-sample indices and gather differentiable references."""
        # best_index[b]: 0=student[b], 1=teacher_0[b], ..., N=teacher_(N-1)[b].
        # No stop_gradient: gradients flow through the selected student/teacher representation.
        # Strict per-sample selection: best_index must be a vector [B], never a scalar.
        best_index = tf.convert_to_tensor(best_index, dtype=tf.int32)
        batch_size = tf.shape(student)[0]
        tf.debugging.assert_rank(
            best_index, 1, message="best_index must be a rank-1 tensor [B]."
        )
        tf.debugging.assert_equal(
            tf.shape(best_index)[0],
            batch_size,
            message="best_index length must match batch size.",
        )
        tf.debugging.assert_greater_equal(best_index, 0)
        tf.debugging.assert_less(best_index, self._config.num_teachers + 1)
        candidates = tf.concat([student[:, None, :], teachers], axis=1)
        reference = tf.gather(candidates, best_index, axis=1, batch_dims=1)
        tf.debugging.assert_equal(tf.shape(reference), tf.shape(student))
        return ReferenceSelection(reference=reference, best_index=best_index)
