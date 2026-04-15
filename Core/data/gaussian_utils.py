"""Gaussian distribution utilities for spatial processing.

Provides functions for computing Gaussian distributions and heatmaps
used in data augmentation and spatial modeling.
"""

from typing import Tuple, Union
import numpy as np
import tensorflow as tf


def gaussian(values: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Compute Gaussian distribution values.

    Args:
        values: Input values
        mu: Mean of the Gaussian
        sig: Standard deviation

    Returns:
        Gaussian probability values
    """
    return np.exp(-np.power(values - mu, 2.0) / (2 * np.power(sig, 2.0)))


def get_gaussian(
    mu: tf.Tensor, tril: tf.Tensor, HW: Union[int, tf.Tensor]
) -> tf.Tensor:
    """Generate Gaussian heatmaps for spatial locations.

    Creates 2D Gaussian distributions normalized by peak probability.

    Args:
        mu: Mean locations of shape (B, 2)
        tril: Lower triangular covariance matrices of shape (B, 2, 2)
        HW: Height/width of output heatmaps

    Returns:
        Normalized Gaussian heatmaps of shape (B, HW, HW)
    """
    B = tf.shape(mu)[0]
    xy = tf.linspace(0.0, 1.0, HW)
    # meshgrid as a list of [x,y] coordinates
    coords = tf.reshape(tf.stack(tf.meshgrid(xy, xy), axis=-1), (-1, 2))
    N = HW * HW  # Total number of coordinate points

    # Compute multivariate normal probability using Cholesky decomposition
    # p(x) = exp(-0.5 * (x-mu)^T * Sigma^{-1} * (x-mu)) / (det(Sigma)^0.5 * (2*pi)^{d/2})
    # where Sigma = tril @ tril^T

    # For each batch, compute the log determinant of Sigma
    # det(Sigma) = det(tril @ tril^T) = det(tril)^2
    log_det_tril = 2.0 * tf.reduce_sum(
        tf.math.log(tf.abs(tf.linalg.diag_part(tril))), axis=-1
    )

    # Solve (tril @ tril^T)^{-1} @ (x - mu) using Cholesky solve
    # First: solve tril @ z = (x - mu)
    # Then: solve tril^T @ w = z to get (tril @ tril^T)^{-1} @ (x - mu)

    # Expand mu and coords for batch operations: (B, 1, 2) and (N, 1, 2)
    mu_expanded = mu[:, None, :]  # (B, 1, 2)
    coords_expanded = coords[None, :, :]  # (1, N, 2)

    # Compute differences (x - mu) for all combinations
    diff = coords_expanded - mu_expanded  # (B, N, 2)

    # Compute (x - mu)^T @ Sigma^{-1} @ (x - mu) using Cholesky decomposition
    # Solve tril @ z = diff^T in two steps:
    # First: solve tril @ w = diff^T for w
    # Then: solve tril^T @ z = w for z to get Sigma^{-1} @ diff
    # Reshape diff for batch triangular solve: (B*N, 2, 1)
    B_times_N = B * N
    diff_reshaped = tf.reshape(diff, (B_times_N, 2, 1))

    # Solve tril @ w = diff using batched triangular solve
    # First solve: tril @ w = diff
    w = tf.linalg.triangular_solve(
        matrix=tf.repeat(tril, N, axis=0),  # Expand tril to (B*N, 2, 2)
        rhs=diff_reshaped,
        lower=True,
    )  # (B*N, 2, 1)

    # Second solve: tril^T @ z = w
    z = tf.linalg.triangular_solve(
        matrix=tf.repeat(tril, N, axis=0),  # Same tril matrix
        rhs=w,
        lower=True,
        adjoint=True,
    )  # (B*N, 2, 1)

    # Reshape back to (B, N, 2, 1)
    z = tf.reshape(z, shape=(B, N, 2, 1))

    # Compute (x - mu)^T @ z
    mahal = tf.reduce_sum(diff[:, :, :, None] * z, axis=2)  # (B, N, 1)
    mahal = tf.squeeze(mahal, axis=-1)  # (B, N)

    # Compute probability: exp(-0.5 * mahal) / normalizer
    # normalizer = det(Sigma)^0.5 * (2*pi)^1 = exp(0.5 * log_det_Sigma) * 2*pi
    log_prob = -0.5 * mahal - 0.5 * log_det_tril[:, None] - tf.math.log(2.0 * np.pi)
    gauss = tf.exp(log_prob)  # (B, N)

    # Compute normalization at mu
    mu_log_prob = -0.5 * log_det_tril - tf.math.log(2.0 * np.pi)
    mu_prob = tf.exp(mu_log_prob)  # (B,)

    # Reshape and normalize
    gauss = tf.reshape(gauss, shape=(B, HW, HW))
    return tf.math.divide_no_nan(tf.maximum(gauss, 0.0), mu_prob[:, None, None])


def addLightBlob(
    imgA: tf.Tensor,
    imgB: tf.Tensor,
    brightness: tf.Tensor,
    shared: bool,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Add random light blob augmentation to images.

    Generates Gaussian light blobs with random positions and scales,
    multiplies with images to create realistic lighting variations.

    Args:
        imgA: First image batch of shape (B, H, W)
        imgB: Second image batch of shape (B, H, W)
        brightness: Brightness factors of shape (B,)
        shared: If True, use same blobs for both images

    Returns:
        Tuple of augmented (imgA, imgB) tensors
    """
    N = tf.shape(imgA)[0]
    HW = tf.shape(imgA)[1]

    def makeBlobs() -> tf.Tensor:
        """Generate random gaussian light blobs for image augmentation.

        Samples random gaussian distributions and converts them to light patterns
        applied to images for illumination augmentation.

        Returns:
            Light pattern tensor with shape matching input images.
        """
        # randomly sample gaussian mu and scale
        lightMu = tf.random.uniform(tf.stack([N, 2]), minval=-0.1, maxval=1.1)
        lightScale = tf.random.uniform(tf.stack([N, 2, 2]), minval=0.01, maxval=0.5)
        light = get_gaussian(lightMu, lightScale, HW)
        tf.debugging.assert_equal(tf.shape(light), tf.shape(imgA))
        tf.debugging.assert_equal(tf.shape(light), tf.shape(imgB))
        lightB = tf.reshape(brightness, (N, 1, 1))
        light = 1.0 + light * lightB
        return light

    lightA = makeBlobs()
    lightB = lightA if shared else makeBlobs()
    return (
        tf.clip_by_value(imgA * lightA, 0.0, 1.0),
        tf.clip_by_value(imgB * lightB, 0.0, 1.0),
    )
