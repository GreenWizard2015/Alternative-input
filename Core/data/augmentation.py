"""Image and point augmentation utilities.

Provides functions for applying augmentations like brightness adjustments,
noise addition, and dropout to eye images and face mesh points.
"""

from typing import Tuple
import tensorflow as tf
from Core.landmarks import FACE_MESH_INVALID_VALUE

# Constants for augmentation
BRIGHTNESS_TRUNCATED_NORMAL_STDDEV = 0.5


def apply_brightness_augmentation(
    imgA: tf.Tensor,
    imgB: tf.Tensor,
    brightnessFactor: tf.Tensor,
    N: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply random brightness augmentation to images.

    Args:
        imgA: First image batch of shape (B, H, W)
        imgB: Second image batch of shape (B, H, W)
        brightnessFactor: Maximum brightness factor
        N: Batch size

    Returns:
        Tuple of augmented (imgA, imgB) tensors
    """

    def clip(x: tf.Tensor) -> tf.Tensor:
        """Clip values to [0.0, 1.0] range.

        Args:
            x: Input tensor.

        Returns:
            Clipped tensor.
        """
        return tf.clip_by_value(x, 0.0, 1.0)

    def sampleBrightness(a: tf.Tensor, b: tf.Tensor, mid: float = 1.0) -> tf.Tensor:
        """Sample brightness factor from truncated normal distribution.

        Args:
            a: Lower bound.
            b: Upper bound.
            mid: Midpoint value (default: 1.0).

        Returns:
            Sampled brightness factors of shape (N,).
        """
        # first sample from truncated normal
        # Note: N is a scalar tensor; use tf.stack to create proper shape [N]
        TN = tf.random.truncated_normal(
            tf.stack([N]), mean=0.0, stddev=BRIGHTNESS_TRUNCATED_NORMAL_STDDEV
        )
        # then transform values [-1, 0] to [a, mid] and [0, 1] to [mid, b]
        return tf.where(TN < 0.0, a + (mid - a) * (TN + 1.0), mid + (b - mid) * TN)

    def apply_brightness() -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply brightness augmentation to images.

        Returns:
            Tuple of brightness-augmented (imgA, imgB) tensors.
        """
        brightness = sampleBrightness(1.0 / brightnessFactor, brightnessFactor)[
            :, None, None
        ]
        return clip(imgA * brightness), clip(imgB * brightness)

    return tf.cond(0.0 < brightnessFactor, apply_brightness, lambda: (imgA, imgB))


def apply_additive_noise(
    imgA: tf.Tensor,
    imgB: tf.Tensor,
    eyesAdditiveNoise: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply random additive noise to images.

    Args:
        imgA: First image batch
        imgB: Second image batch
        eyesAdditiveNoise: Noise standard deviation

    Returns:
        Tuple of augmented (imgA, imgB) tensors
    """

    def clip(x: tf.Tensor) -> tf.Tensor:
        """Clip values to [0.0, 1.0] range.

        Args:
            x: Input tensor.

        Returns:
            Clipped tensor.
        """
        return tf.clip_by_value(x, 0.0, 1.0)

    def apply_noise() -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply additive Gaussian noise to images.

        Returns:
            Tuple of noise-augmented (imgA, imgB) tensors.
        """
        noiseA = tf.random.normal(tf.shape(imgA), stddev=eyesAdditiveNoise)
        noiseB = tf.random.normal(tf.shape(imgB), stddev=eyesAdditiveNoise)
        return clip(imgA + noiseA), clip(imgB + noiseB)

    return tf.cond(0.0 < eyesAdditiveNoise, apply_noise, lambda: (imgA, imgB))


def apply_dropout(
    imgA: tf.Tensor,
    imgB: tf.Tensor,
    eyesDropout: tf.Tensor,
    N: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply random dropout to images.

    Args:
        imgA: First image batch
        imgB: Second image batch
        eyesDropout: Dropout probability
        N: Batch size

    Returns:
        Tuple of augmented (imgA, imgB) tensors with dropout applied
    """

    def apply_dropout_inner() -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply dropout to images by zeroing random samples.

        Returns:
            Tuple of dropout-augmented (imgA_out, imgB_out) tensors.
        """
        mask = tf.random.uniform(tf.stack([N])) < eyesDropout
        maskA = 0.5 < tf.random.uniform(tf.stack([N]))
        maskB = tf.logical_not(maskA)
        imgA_out = tf.where(tf.logical_and(mask, maskA)[:, None, None], 0.0, imgA)
        imgB_out = tf.where(tf.logical_and(mask, maskB)[:, None, None], 0.0, imgB)
        return imgA_out, imgB_out

    return tf.cond(0.0 < eyesDropout, apply_dropout_inner, lambda: (imgA, imgB))


def apply_points_noise(points: tf.Tensor, pointsNoise: tf.Tensor) -> tf.Tensor:
    """Apply random noise to face mesh points.

    Args:
        points: Face mesh points
        pointsNoise: Noise standard deviation

    Returns:
        Augmented points tensor
    """

    def apply_noise() -> tf.Tensor:
        """Apply Gaussian noise to face mesh points.

        Returns:
            Augmented points tensor.
        """
        return points + tf.random.normal(tf.shape(points), stddev=pointsNoise)

    return tf.cond(0.0 < pointsNoise, apply_noise, lambda: points)


def apply_points_dropout(
    points: tf.Tensor,
    pointsDropout: tf.Tensor,
    N: tf.Tensor,
) -> tf.Tensor:
    """Apply random dropout to face mesh points.

    Args:
        points: Face mesh points
        pointsDropout: Dropout probability
        N: Batch size

    Returns:
        Augmented points tensor with dropout applied
    """

    def apply_dropout() -> tf.Tensor:
        """Apply dropout to face mesh points by masking with invalid value.

        Returns:
            Augmented points tensor with dropout applied.
        """
        mask = tf.random.uniform(tf.stack([N, tf.shape(points)[1]])) < pointsDropout
        return tf.where(mask[:, :, None], FACE_MESH_INVALID_VALUE, points)

    return tf.cond(0.0 < pointsDropout, apply_dropout, lambda: points)
