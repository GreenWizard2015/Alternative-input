"""Ultra-thin adapter MLP for knowledge distillation feature matching.

Provides lightweight neural network adapter for projecting student latent features
to teacher latent feature space during knowledge distillation training.
"""

import tensorflow as tf
from NN.layers.sMLP import sMLP

from NN.models.NpzModelMixin import NpzModelMixin


class AdapterMLP(NpzModelMixin, tf.keras.Model):
    """Ultra-thin adapter MLP: student_dim → intermediate_dim → teacher_dim.

    Adapter handles projection from student latent dimension space to teacher
    latent dimension space using a bottleneck architecture to minimize parameters
    while enabling feature alignment.

    Separate instances created for intermediate and final latents to allow
    independent learning of feature projections for each layer.

    Attributes:
        _student_dim: Input dimension (student latent size)
        _teacher_dim: Output dimension (teacher latent size)
        _intermediate_dim: Bottleneck dimension for parameter efficiency
        _dense1: First dense layer (student_dim → intermediate_dim)
        _dense2: Second dense layer (intermediate_dim → teacher_dim)
    """

    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        name: str = "AdapterMLP",
    ) -> None:
        """Initialize ultra-thin adapter MLP.

        Args:
            student_dim: Dimension of input student features (e.g., 256).
            teacher_dim: Dimension of target teacher features (e.g., 512).
            intermediate_dim: Bottleneck dimension for compression.
                If None, uses geometric mean: sqrt(student_dim * teacher_dim).
                Must be >= 1 and <= max(student_dim, teacher_dim).
            name: Name of the adapter layer.

        Raises:
            ValueError: If dimensions are invalid or inconsistent.
        """
        super().__init__(name=name)

        # Input validation
        if teacher_dim <= 0:
            raise ValueError(f"teacher_dim must be positive, got {teacher_dim}")
        if student_dim <= 0:
            raise ValueError(f"student_dim must be positive, got {student_dim}")

        N = 2
        sizes = ([student_dim] * N) + ([teacher_dim] * N)
        self._to_teacher = sMLP(sizes=sizes, name="to_teacher")
        self._to_teacher.build((None, None, student_dim))

    def call(self, features: tf.Tensor) -> tf.Tensor:
        """Project student features to teacher dimension space."""
        return self._to_teacher(features, training=True)
