"""Named tensor results shared by EBM components."""

from typing import NamedTuple

import tensorflow as tf


class ValidatedFeatures(NamedTuple):
    student: tf.Tensor
    teachers: list[tf.Tensor]
    batch_size: tf.Tensor


class SystemLevels(NamedTuple):
    student: tf.Tensor
    reference: tf.Tensor
    teachers: tf.Tensor


class SoftmaxResult(NamedTuple):
    probabilities: tf.Tensor
    log_probabilities: tf.Tensor


class RoutingStatistics(NamedTuple):
    balance: tf.Tensor
    entropy: tf.Tensor
    usage_sum: tf.Tensor
    usage_count: tf.Tensor


class ReferenceSelection(NamedTuple):
    reference: tf.Tensor
    best_index: tf.Tensor


class CalibrationLoss(NamedTuple):
    total: tf.Tensor
    positive: tf.Tensor
    negative: tf.Tensor


class LevelLosses(NamedTuple):
    contrastive: tf.Tensor
    balance: tf.Tensor
    entropy: tf.Tensor
    calibration: tf.Tensor

    @classmethod
    def zeros(cls):
        zero = tf.zeros([], dtype=tf.float32)
        return cls(
            contrastive=zero,
            balance=zero,
            entropy=zero,
            calibration=zero,
        )

    def add_level(self, level):
        return LevelLosses(
            contrastive=self.contrastive + level.contrastive,
            balance=self.balance + level.balance,
            entropy=self.entropy + level.entropy,
            calibration=self.calibration + level.calibration,
        )

    def scaled(self, factor):
        return LevelLosses(
            contrastive=self.contrastive * factor,
            balance=self.balance * factor,
            entropy=self.entropy * factor,
            calibration=self.calibration * factor,
        )


class PathMetrics(NamedTuple):
    """Fraction-averaged metrics for one Direct, Reference or STE path."""

    pair_order: tf.Tensor
    system_usage: tf.Tensor
    active: tf.Tensor


class TrainingLevel(NamedTuple):
    contrastive: tf.Tensor
    balance: tf.Tensor
    entropy: tf.Tensor
    calibration: tf.Tensor
    energy: tf.Tensor
    usage_sum: tf.Tensor
    usage_count: tf.Tensor
