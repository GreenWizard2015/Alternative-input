"""Ultra-thin adapter MLP for knowledge distillation feature matching.

Provides lightweight neural network adapter for projecting student latent features
to teacher latent feature space during knowledge distillation training.
"""

import tensorflow as tf
from NN.layers.sMLP import sMLP

from NN.models.NpzModelMixin import NpzModelMixin


class ResidualAE(NpzModelMixin, tf.keras.Model):
    def __init__(
        self,
        dim: int,
        name: str = "ResidualAE",
    ) -> None:
        super().__init__(name=name)

        # Input validation
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        N = 3
        sizes = ([int(dim * 2)] * N) + [dim]
        self._ae = sMLP(sizes=sizes, name="to_teacher")
        self._ae.build((None, None, dim))

    def call(self, features: tf.Tensor) -> tf.Tensor:
        """Project student features to teacher dimension space."""
        return self._ae(features, training=True) + features
