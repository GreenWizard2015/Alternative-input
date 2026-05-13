"""Ultra-thin adapter MLP for knowledge distillation feature matching.

Provides lightweight neural network adapter for projecting student latent features
to teacher latent feature space during knowledge distillation training.
"""

from typing import Dict

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
        sizes = [int(dim * 2)] * N
        self._ae = sMLP(sizes=sizes, activation="relu", name="to_teacher")
        self._ae.build((None, None, 2 * dim))
        self._linear = tf.keras.layers.Dense(dim, activation="linear")
        self._linear.build((None, None, 2 * dim))

    def call(self, data: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Project student features to teacher dimension space."""
        features = data["features"]
        res = self._ae(
            tf.concat([features, data["mask"]], axis=-1),
            training=True,
        )
        return self._linear(res, training=True) + features
