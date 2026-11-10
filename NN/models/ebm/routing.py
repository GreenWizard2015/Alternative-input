"""Stable softmax, soft-min and system routing statistics."""

import tensorflow as tf

from NN.models.ebm.config import EBMConfig
from NN.models.ebm.types import RoutingStatistics, SoftmaxResult


class EBMRouting:
    """Route using shared temperatures and system count, without weights."""

    def __init__(self, config: EBMConfig) -> None:
        self._config = config

    def log_softmax(self, logits: tf.Tensor, axis: int) -> tf.Tensor:
        """Compute log probabilities with the winner excluded from logsumexp."""
        # Barron et al. (2026), Eq. 5: exclude the winner from logsumexp and
        # combine the remaining exponentials via softplus. The winner's
        # shifted logit is an exact constant zero, avoiding gradient collapse.
        logits = tf.cast(logits, tf.float32)
        axis = axis if axis >= 0 else logits.shape.rank + axis
        if logits.shape[axis] == 1:
            return tf.zeros_like(logits)
        winners = tf.argmax(logits, axis=axis, output_type=tf.int32)
        mask = tf.one_hot(
            winners,
            depth=tf.shape(logits)[axis],
            axis=axis,
            on_value=True,
            off_value=False,
            dtype=tf.bool,
        )
        shifted = tf.where(
            mask,
            tf.zeros_like(logits),
            logits - tf.reduce_max(logits, axis=axis, keepdims=True),
        )
        others = tf.where(
            mask,
            tf.fill(tf.shape(logits), tf.constant(-float("inf"), logits.dtype)),
            shifted,
        )
        correction = tf.nn.softplus(
            tf.reduce_logsumexp(others, axis=axis, keepdims=True)
        )
        return shifted - correction

    def softmax(self, logits: tf.Tensor, axis: int) -> SoftmaxResult:
        """Return stable probabilities and their logarithms."""
        log_p = self.log_softmax(logits, axis)
        return SoftmaxResult(probabilities=tf.exp(log_p), log_probabilities=log_p)

    def logsumexp(self, logits: tf.Tensor, axis: int) -> tf.Tensor:
        """Reduce logits using the same winner-exclusion correction."""
        # Stable logsumexp with the argmax removed from the exponential sum.
        axis = axis if axis >= 0 else logits.shape.rank + axis
        if logits.shape[axis] == 1:
            return tf.squeeze(logits, axis=axis)
        winners = tf.argmax(logits, axis=axis, output_type=tf.int32)
        mask = tf.one_hot(
            winners,
            depth=tf.shape(logits)[axis],
            axis=axis,
            on_value=True,
            off_value=False,
            dtype=tf.bool,
        )
        maximum = tf.reduce_max(logits, axis=axis, keepdims=True)
        shifted = tf.where(mask, tf.zeros_like(logits), logits - maximum)
        others = tf.where(
            mask,
            tf.fill(tf.shape(logits), tf.cast(-float("inf"), logits.dtype)),
            shifted,
        )
        return tf.squeeze(
            maximum
            + tf.nn.softplus(tf.reduce_logsumexp(others, axis=axis, keepdims=True)),
            axis=axis,
        )

    def scores(self, energy: tf.Tensor, temperature: tf.Tensor) -> tf.Tensor:
        """Compute per-system InfoNCE routing scores [B,S,N]."""
        # energy[i,j,s,n]: anchor/teacher i vs student j, system s, n=1 or N.
        # Hard-target CE: -T * softplus(logsumexp_{i!=j}(logit_i-logit_j)).
        # Excluding the positive logit prevents the Softmax Collapse bug;
        # subtraction of two nearly equal logsumexp values is also avoided.
        energy = tf.cast(energy, tf.float32)
        temperature = tf.cast(temperature, tf.float32)
        batch = tf.shape(energy)[0]
        logits = -energy / temperature
        positive = tf.einsum("ii...->i...", logits)
        negative_mask = tf.eye(batch, dtype=tf.bool)[:, :, None, None]
        differences = tf.where(
            negative_mask,
            tf.fill(tf.shape(logits), tf.constant(-float("inf"), tf.float32)),
            logits - positive[None, :, :, :],
        )
        return -temperature * tf.nn.softplus(tf.reduce_logsumexp(differences, axis=0))

    def statistics(
        self,
        scores: tf.Tensor,
        valid_teacher: tf.Tensor | None = None,
    ) -> RoutingStatistics:
        """Identical routing statistics for Direct and Reference, with optional
        exclusion of trivial reference==teacher anchor-teacher pairs.

        scores: [B,S,N]; valid_teacher: [B,N]. Balancing is measured per
        teacher, so one frequently selected teacher cannot hide collapse in
        another teacher's routing.
        """
        logits = tf.cast(scores, tf.float32) / self._config.system_temperature
        softmax = self.softmax(logits, axis=1)
        probabilities = softmax.probabilities
        log_probabilities = softmax.log_probabilities
        if valid_teacher is None:
            expected_usage = tf.reduce_mean(probabilities, axis=0)
            balance = tf.reduce_mean(
                tf.square(expected_usage - 1.0 / self._config.num_systems)
            )
            entropy = -tf.reduce_mean(
                tf.reduce_sum(probabilities * log_probabilities, axis=1)
            )
        else:
            valid = tf.cast(valid_teacher, probabilities.dtype)
            counts = tf.reduce_sum(valid, axis=0)  # [N]
            expected_usage = tf.math.divide_no_nan(
                tf.reduce_sum(probabilities * valid[:, None, :], axis=0),
                counts[None, :],
            )  # [S,N]
            active_teacher = tf.cast(counts > 0, probabilities.dtype)
            balance = tf.math.divide_no_nan(
                tf.reduce_sum(
                    tf.square(expected_usage - 1.0 / self._config.num_systems)
                    * active_teacher[None, :]
                ),
                tf.cast(self._config.num_systems, probabilities.dtype)
                * tf.reduce_sum(active_teacher),
            )
            entropy = -tf.math.divide_no_nan(
                tf.reduce_sum(
                    tf.reduce_sum(probabilities * log_probabilities, axis=1) * valid
                ),
                tf.reduce_sum(valid),
            )
        # Expected routing usage for every system, averaged over actual
        # valid anchors/teachers. No Gumbel sampling noise in diagnostics.
        if valid_teacher is None:
            usage_sum = tf.reduce_sum(probabilities, axis=[0, 2])
            usage_count = tf.cast(
                tf.shape(scores)[0] * tf.shape(scores)[2], probabilities.dtype
            )
        else:
            usage_sum = tf.reduce_sum(probabilities * valid[:, None, :], axis=[0, 2])
            usage_count = tf.reduce_sum(valid)
        return RoutingStatistics(
            balance=balance,
            entropy=entropy,
            usage_sum=usage_sum,
            usage_count=usage_count,
        )
