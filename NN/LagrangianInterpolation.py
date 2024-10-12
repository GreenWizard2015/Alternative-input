import tensorflow as tf

def lagrange_interpolation(x_values, y_values, x_targets):
    """
    Perform Lagrange Polynomial Interpolation using TensorFlow with batch support
    and multidimensional y_values.

    Parameters:
    - x_values: Tensor of shape (batch_size, n), original x-values for each batch.
    - y_values: Tensor of shape (batch_size, n, d), original y-values for each batch.
    - x_targets: Tensor of shape (batch_size, m), x-values where interpolation is computed.

    Returns:
    - interpolated_values: Tensor of shape (batch_size, m, d), interpolated y-values for each batch.
    """
    minX = tf.reduce_min(x_values, axis=1)
    maxX = tf.reduce_max(x_values, axis=1)
    # Check if x_targets in the range of x_values
    tf.debugging.assert_greater_equal(x_targets, minX, message="x_targets out of range")
    tf.debugging.assert_less_equal(x_targets, maxX, message="x_targets out of range")

    batch_size = tf.shape(x_values)[0]
    n = tf.shape(x_values)[1]
    m = tf.shape(x_targets)[-1]
    d = tf.shape(y_values)[2]

    tf.assert_equal(tf.shape(x_values), (batch_size, n))
    tf.assert_equal(tf.shape(y_values), (batch_size, n, d))
    tf.assert_equal(tf.shape(x_targets), (batch_size, m))
    # Reshape tensors for broadcasting
    x_values_i = tf.reshape(x_values, (batch_size, n, 1, 1))        # Shape: (batch_size, n, 1, 1)
    x_values_j = tf.reshape(x_values, (batch_size, 1, n, 1))        # Shape: (batch_size, 1, n, 1)

    x_targets_k = tf.reshape(x_targets, (batch_size, 1, 1, m))  # Shape: (batch_size, 1, 1, m)

    # Compute the denominators (x_i - x_j)
    denominators = x_values_i - x_values_j                          # Shape: (batch_size, n, n, 1)
    # Replace zeros on the diagonal with ones to avoid division by zero
    denominators = tf.where(tf.equal(denominators, 0.0), tf.ones_like(denominators), denominators)

    # Compute the numerators (x_k - x_j)
    numerators = x_targets_k - x_values_j                           # Shape: (batch_size, 1, n, m)

    # Compute the terms (x_k - x_j) / (x_i - x_j)
    terms = numerators / denominators                               # Shape: (batch_size, n, n, m)

    # Exclude the terms where i == j by setting them to 1
    identity_matrix = tf.eye(n, batch_shape=[batch_size], dtype=tf.float64)  # Shape: (batch_size, n, n)
    identity_matrix = tf.reshape(identity_matrix, (batch_size, n, n, 1))     # Shape: (batch_size, n, n, 1)
    terms = tf.where(tf.equal(identity_matrix, 1.0), tf.ones_like(terms), terms)

    # Compute the product over j for each i and x_k
    basis_polynomials = tf.reduce_prod(terms, axis=2)               # Shape: (batch_size, n, m)

    # Multiply each basis polynomial by the corresponding y_i
    # Adjust shapes for broadcasting
    basis_polynomials_expanded = tf.expand_dims(basis_polynomials, axis=-1)  # Shape: (batch_size, n, m, 1)
    y_values_expanded = tf.expand_dims(y_values, axis=2)                     # Shape: (batch_size, n, 1, d)
    products = basis_polynomials_expanded * y_values_expanded                # Shape: (batch_size, n, m, d)

    # Sum over i to get the interpolated values
    interpolated_values = tf.reduce_sum(products, axis=1)                    # Shape: (batch_size, m, d)

    return interpolated_values