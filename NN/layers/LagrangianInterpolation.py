"""Lagrange polynomial interpolation with batch and multi-dimensional support.

Implements Lagrange polynomial interpolation for smooth temporal or spatial
interpolation of multi-dimensional values at arbitrary query points.
"""

from typing import Tuple
import tensorflow as tf


def get_x_range(x_values: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Get min/max x values from input data.

    Computes minimum and maximum x values for each batch.

    Args:
        x_values: Input x-values tensor of shape (batch_size, n).

    Returns:
        Tuple of:
            - min_x_per_batch: Minimum x value for each batch, shape (batch_size, 1)
            - max_x_per_batch: Maximum x value for each batch, shape (batch_size, 1)
    """
    min_x_per_batch = tf.reduce_min(x_values, axis=1, keepdims=True)
    max_x_per_batch = tf.reduce_max(x_values, axis=1, keepdims=True)

    return min_x_per_batch, max_x_per_batch


def normalize_and_check_targets(
    x_values: tf.Tensor, x_targets: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Normalize x values and check if targets are inside values range.

    Normalizes both x_values and x_targets to [0, 1] range, then asserts that
    all targets fall within the corresponding values range for each batch.

    Args:
        x_values: Original x-values tensor of shape (batch_size, n).
            Must be sorted in ascending order per batch.
        x_targets: Target x-values tensor of shape (batch_size, m).
            Must be within [x_values_min, x_values_max] for each batch.

    Returns:
        Tuple of:
            - x_values_normalized: Normalized x-values to [0, 1], shape (batch_size, n)
            - x_targets_normalized: Normalized x-targets to [0, 1], shape (batch_size, m)

    Raises:
        tf.errors.InvalidArgumentError: If any target is outside the corresponding
            values range for its batch.
    """
    # Find min/max x values from original data
    x_values_min, x_values_max = get_x_range(x_values)
    x_targets_min, x_targets_max = get_x_range(x_targets)

    # Compute overall min/max for normalization
    x_min = tf.minimum(x_values_min, x_targets_min)
    x_max = tf.maximum(x_values_max, x_targets_max)

    # Normalize to [0, 1] range
    x_duration = (x_max - x_min) + 1e-8
    x_values_normalized = (x_values - x_min) / x_duration
    x_targets_normalized = (x_targets - x_min) / x_duration

    # Check if targets are inside values range for each batch
    targets_inside = tf.logical_and(
        tf.greater_equal(x_targets_min, x_values_min),
        tf.less_equal(x_targets_max, x_values_max),
    )  # Shape: (batch_size, 1)

    # Create detailed error message with actual values
    def create_error_message():
        failed_batches = tf.where(~tf.squeeze(targets_inside, axis=1))
        msg = (
            "Not all targets are inside values range. "
            + "Failed batches (0-indexed): "
            + tf.as_string(failed_batches)
            + ". "
            + "x_values_min: "
            + tf.as_string(tf.reshape(x_values_min, [-1]))
            + ", "
            + "x_values_max: "
            + tf.as_string(tf.reshape(x_values_max, [-1]))
            + ", "
            + "x_targets_min: "
            + tf.as_string(tf.reshape(x_targets_min, [-1]))
            + ", "
            + "x_targets_max: "
            + tf.as_string(tf.reshape(x_targets_max, [-1]))
        )
        return msg

    tf.debugging.assert_equal(
        tf.reduce_all(targets_inside),
        True,
        message=create_error_message(),
    )

    return x_values_normalized, x_targets_normalized


def apply_boundary_points_conditional(
    x_values: tf.Tensor, y_values: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply boundary points to smooth interpolation at edges.

    Adds boundary points at -0.01 and 1.01 to extend the interpolation domain.
    Since x_values are normalized to [0, 1], boundary points are always -0.01 and 1.01.

    Args:
        x_values: x-values tensor of shape (batch_size, n).
            Must be normalized to [0, 1] range.
        y_values: y-values tensor of shape (batch_size, n, d).

    Returns:
        Tuple of:
            - x_values_processed: x-values with boundary points, shape (batch_size, n+2)
            - y_values_processed: y-values with replicated endpoints, shape (batch_size, n+2, d)
    """
    # Since x_values are normalized to [0, 1], boundary points are simply:
    # boundary_left = -0.01 (0 - 0.01 * 1)
    # boundary_right = 1.01 (1 + 0.01 * 1)
    batch_size = tf.shape(x_values)[0]
    boundary_left = tf.fill((batch_size, 1), -0.01)  # Shape: (batch_size, 1)
    boundary_right = tf.fill((batch_size, 1), 1.01)  # Shape: (batch_size, 1)

    # Add boundary points: concat along axis 1 (sequence dimension)
    x_values_processed = tf.concat(
        [boundary_left, x_values, boundary_right], axis=1
    )  # Shape: (batch_size, n+2)

    # For y_values, replicate first and last points
    y_first = y_values[:, :1, :]  # Shape: (batch_size, 1, d)
    y_last = y_values[:, -1:, :]  # Shape: (batch_size, 1, d)
    y_values_processed = tf.concat(
        [y_first, y_values, y_last], axis=1
    )  # Shape: (batch_size, n+2, d)

    return x_values_processed, y_values_processed


def lagrange_interpolation(
    x_values: tf.Tensor, y_values: tf.Tensor, x_targets: tf.Tensor
) -> tf.Tensor:
    """Perform Lagrange polynomial interpolation with batch and multidimensional support.

    Interpolates multidimensional y-values at target x-points using Lagrange basis
    polynomials. Normalizes inputs to [0, 1] and adds boundary points for smooth
    interpolation at domain edges.

    Args:
        x_values: Original x-values tensor of shape (batch_size, n).
            Must be sorted in ascending order for each batch.
        y_values: Original y-values tensor of shape (batch_size, n, d).
            Contains d-dimensional values at each x-value.
        x_targets: Query x-values tensor of shape (batch_size, m).
            All targets must be within [x_values_min, x_values_max] per batch.

    Returns:
        Interpolated y-values tensor of shape (batch_size, m, d).
            Smooth d-dimensional values at target x-points.

    Raises:
        tf.errors.InvalidArgumentError: If any target is outside the corresponding
            values range for its batch.

    Example:
        >>> x_vals = tf.constant([[0.0, 1.0, 2.0]], dtype=tf.float32)
        >>> y_vals = tf.constant([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]], dtype=tf.float32)
        >>> x_targets = tf.constant([[0.5, 1.5]], dtype=tf.float32)
        >>> result = lagrange_interpolation(x_vals, y_vals, x_targets)
        >>> assert result.shape == (1, 2, 2)
    """
    # Normalize x values and check if targets are inside values range
    x_values, x_targets = normalize_and_check_targets(x_values, x_targets)

    # Apply boundary points conditionally per batch
    x_values, y_values = apply_boundary_points_conditional(x_values, y_values)

    batch_size = tf.shape(x_values)[0]
    num_original_points = tf.shape(x_values)[1]
    num_target_points = tf.shape(x_targets)[-1]
    value_dimension = tf.shape(y_values)[2]

    tf.debugging.assert_equal(tf.shape(x_values), (batch_size, num_original_points))
    tf.debugging.assert_equal(
        tf.shape(y_values), (batch_size, num_original_points, value_dimension)
    )
    tf.debugging.assert_equal(tf.shape(x_targets), (batch_size, num_target_points))
    # Reshape tensors for broadcasting
    x_values_i = tf.reshape(
        x_values, (batch_size, num_original_points, 1, 1)
    )  # Shape: (batch_size, num_original_points, 1, 1)
    x_values_j = tf.reshape(
        x_values, (batch_size, 1, num_original_points, 1)
    )  # Shape: (batch_size, 1, num_original_points, 1)

    x_targets_k = tf.reshape(
        x_targets, (batch_size, 1, 1, num_target_points)
    )  # Shape: (batch_size, 1, 1, num_target_points)

    # Compute the denominators (x_i - x_j)
    denominators = (
        x_values_i - x_values_j
    )  # Shape: (batch_size, num_original_points, num_original_points, 1)
    # Replace zeros on the diagonal with ones to avoid division by zero
    denominators = tf.where(
        tf.equal(denominators, 0.0), tf.ones_like(denominators), denominators
    )

    # Compute the numerators (x_k - x_j)
    numerators = (
        x_targets_k - x_values_j
    )  # Shape: (batch_size, 1, num_original_points, num_target_points)

    # Compute the terms (x_k - x_j) / (x_i - x_j)
    terms = (
        numerators / denominators
    )  # Shape: (batch_size, num_original_points, num_original_points, num_target_points)

    # Exclude the terms where i == j by setting them to 1
    identity_matrix = tf.eye(
        num_original_points, dtype=terms.dtype
    )  # Shape: (num_original_points, num_original_points)
    identity_matrix = tf.expand_dims(
        identity_matrix, 0
    )  # Shape: (1, num_original_points, num_original_points)
    identity_matrix = tf.repeat(
        identity_matrix, batch_size, axis=0
    )  # Shape: (batch_size, num_original_points, num_original_points)
    identity_matrix = tf.reshape(
        identity_matrix, (batch_size, num_original_points, num_original_points, 1)
    )  # Shape: (batch_size, num_original_points, num_original_points, 1)
    terms = tf.where(tf.equal(identity_matrix, 1.0), tf.ones_like(terms), terms)

    # Compute the product over j for each i and x_k
    basis_polynomials = tf.reduce_prod(
        terms, axis=2
    )  # Shape: (batch_size, num_original_points, num_target_points)

    # Multiply each basis polynomial by the corresponding y_i
    # Adjust shapes for broadcasting
    basis_polynomials_expanded = tf.expand_dims(
        basis_polynomials, axis=-1
    )  # Shape: (batch_size, n, m, 1)
    y_values_expanded = tf.expand_dims(y_values, axis=2)  # Shape: (batch_size, n, 1, d)
    products = (
        basis_polynomials_expanded * y_values_expanded
    )  # Shape: (batch_size, n, m, d)

    # Sum over i to get the interpolated values
    interpolated_values = tf.reduce_sum(products, axis=1)  # Shape: (batch_size, m, d)

    return interpolated_values
