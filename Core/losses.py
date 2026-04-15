"""Loss functions for model training.

Provides multiple loss functions for 2D point prediction tasks (gaze, facial landmarks).
Adapted from reference implementation to handle multi-dimensional point outputs.
"""

from typing import Dict
import tensorflow as tf
from Core.Constants import PSEUDO_HUBER_LOSS_DELTA


def huber_loss(ytrue: tf.Tensor, ypred: tf.Tensor) -> tf.Tensor:
    """Calculate pseudo-Huber loss.

    Uses smooth approximation of absolute loss that is less sensitive
    to outliers than mean squared error.

    Args:
        ytrue: Ground truth points of shape (B, T, ?) or (B, ?)
        ypred: Predicted points of shape (B, T, ?) or (B, ?)

    Returns:
        Loss tensor of shape (B, T) or (B,)
    """
    diff = tf.square(ytrue - ypred)
    loss = tf.sqrt(diff + PSEUDO_HUBER_LOSS_DELTA**2) - PSEUDO_HUBER_LOSS_DELTA
    return tf.reduce_mean(loss, axis=-1)


def calculate_losses(
    predictions: Dict[str, tf.Tensor], y: Dict[str, tf.Tensor], training: bool = False
) -> Dict[str, tf.Tensor]:
    """Calculate combined losses using multiple metrics.

    Args:
        predictions: Dictionary mapping loss keys to predicted tensors of shape (B, T, ?) or (B, ?)
        y: Dictionary mapping loss keys to ground truth tensors of shape (B, T, ?) or (B, ?)
        training: Whether in training mode

    Returns:
        Dictionary mapping same keys to combined loss tensors
    """
    losses = {}
    for key, pred in predictions.items():
        y_true = y[key]
        tf.debugging.assert_equal(tf.shape(pred), tf.shape(y_true))

        huber = huber_loss(ytrue=y_true, ypred=pred)
        denom = tf.stop_gradient(tf.math.reduce_std(huber)) + 1e-9
        normalized = (huber - tf.stop_gradient(tf.reduce_mean(huber))) / denom
        losses[key] = normalized + tf.stop_gradient(huber - normalized)

    return losses
