"""Utility functions and classes for neural network models.

Provides helper functions for vector normalization, fusion blocks for multi-stream
processing, and optimizer configuration.
"""

from typing import Optional, Dict, Any, NamedTuple
import tensorflow as tf


class NormVecResult(NamedTuple):
    """Result of vector normalization.

    Attributes:
        normalized: Unit vectors (L2 normalized) with same shape as input.
            NaN values from zero-length vectors replaced with 0.
        length: Magnitudes (L2 norms) of input vectors with shape (...,).
    """

    normalized: tf.Tensor
    length: tf.Tensor


# Constants
SMLP_GLOBAL_DROPOUT = 0.01

# Optimizer configuration defaults
DEFAULT_OPTIMIZER_NAME = "AdamW"
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
OPTIMIZER_ADAMW = "AdamW"
OPTIMIZER_ADAM = "Adam"
DEFAULT_EXCLUDE_FROM_WEIGHT_DECAY = [
    "batch_normalization",
    "bias",
    "gamma",
    "beta",
    # Exclude CoordsEncodingLayer from weight decay
    "CEL_frequency",
    "CEL_gates",
    "CEL_shifts",
    "CEL_fusion_w",
    "CEL_freq_deltas",
    "CEL_fusion_b",
    # Exclude some custom layers variables
    "_gate",
    "_PE",
    "_scale",
]


def norm_vec(x: tf.Tensor) -> NormVecResult:
    """Normalize vectors and compute magnitudes.

    Normalizes vectors along the last axis to unit length and computes their
    magnitudes (L2 norms). Handles edge cases by replacing NaN values with zeros.

    Args:
        x: Input tensor of any shape (..., N) where vectors are along last axis.

    Returns:
        NormVecResult with fields:
            - normalized: Unit vectors (L2 normalized) with same shape as input.
              NaN values (from zero-length vectors) replaced with 0.
            - length: Magnitudes (L2 norms) of input vectors with shape (...,).

    Raises:
        ValueError: If input tensor rank is less than 1.

    Example:
        >>> vectors = tf.constant([[3.0, 4.0], [0.0, 0.0]])  # 2 vectors
        >>> result = norm_vec(vectors)
        >>> assert result.normalized[0].numpy() == [0.6, 0.8]  # Normalized
        >>> assert result.length[0].numpy() == 5.0  # Magnitude
    """
    if len(x.shape) < 1:
        raise ValueError(f"Input tensor must have rank >= 1, got shape {x.shape}")
    V, L_norm = tf.linalg.normalize(x, axis=-1)
    V = tf.where(tf.math.is_nan(V), 0.0, V)
    return NormVecResult(normalized=V, length=L_norm)


def create_optimizer(
    config: Optional[Dict[str, Any]] = None,
) -> tf.optimizers.Optimizer:
    """Create optimizer (AdamW or Adam) with optional weight decay exclusions.

    Creates an optimizer configured for training neural networks with
    support for excluding specific variables from weight decay (AdamW only).

    Args:
        config: Optional configuration dictionary with keys:
            - 'name': Optimizer name - "AdamW" or "Adam" (default: "AdamW")
            - 'learning_rate': Learning rate (default: 1e-4)
            - 'weight_decay': Weight decay coefficient (default: 1e-4, AdamW only)
            - 'exclude_from_weight_decay': List of variable name patterns to exclude (AdamW only)

    Returns:
        Configured optimizer instance (AdamW or Adam)

    Example:
        >>> optimizer = create_optimizer({'name': 'AdamW', 'learning_rate': 0.001})
        >>> optimizer = create_optimizer({'name': 'Adam'})
        >>> optimizer = create_optimizer()  # Uses defaults (AdamW)
    """
    if config is None:
        config = {
            "name": DEFAULT_OPTIMIZER_NAME,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "exclude_from_weight_decay": DEFAULT_EXCLUDE_FROM_WEIGHT_DECAY,
        }

    optimizer_name = config.get("name", DEFAULT_OPTIMIZER_NAME)
    learning_rate = config.get("learning_rate", DEFAULT_LEARNING_RATE)

    if optimizer_name == OPTIMIZER_ADAMW:
        weight_decay = config.get("weight_decay", DEFAULT_WEIGHT_DECAY)
        optimizer = tf.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        var_names = config.get("exclude_from_weight_decay")
        if var_names is not None:
            _ = optimizer.exclude_from_weight_decay(var_names=var_names)
        return optimizer

    if optimizer_name == OPTIMIZER_ADAM:
        return tf.optimizers.Adam(learning_rate=learning_rate)

    raise ValueError(
        f"Unknown optimizer: {optimizer_name}. Use '{OPTIMIZER_ADAMW}' or '{OPTIMIZER_ADAM}'"
    )


def normalize_std(x):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    std = tf.math.reduce_std(x, axis=-1, keepdims=True) + 1e-8
    return (x - mean) / std


def partial_gradient(v, weight):
    part = weight * v
    return part + tf.stop_gradient(v - part)


def JensenShannonDivergence(p, q):
    m = (p + q) / 2.0
    return (tf.losses.kld(p, m) + tf.losses.kld(q, m)) / 2.0


def flatten(inputs):
    return [tf.reshape(x, (-1, x.shape[-1])) for x in inputs]


def find_optimal_latents_sgld(
    energy_model,
    z,
    max_steps,
    step_size=0.03,
    step_decay=0.9,
    delta=0e-2,
):
    """
    Find low-energy latents with short-run Langevin / SGLD.
    """

    def cast_value(value):
        return tf.cast(value, z.dtype)

    def schedule(base, decay, itr):
        return cast_value(base) * tf.pow(cast_value(decay), cast_value(itr))

    def calc_energy(z, mask):
        energy = energy_model(z, mask=mask)
        return tf.reduce_mean(energy, axis=-1, keepdims=True)

    def add_noise(z, lr):
        noise_std = tf.sqrt(cast_value(2.0) * lr)
        return tf.random.normal(tf.shape(z), mean=z, stddev=noise_std, dtype=z.dtype)

    def update_best(z, energy, best_z, best_energy):
        is_better = energy < best_energy

        best_z = tf.where(is_better, z, best_z)
        best_energy = tf.where(is_better, energy, best_energy)

        return best_z, best_energy

    def calc_grads(z, mask):
        z = tf.stop_gradient(z)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(z)
            energy = calc_energy(z, mask=mask)

        grad = tape.gradient(energy, z)
        grad = tf.where(mask[..., None], grad, 0.0)
        return grad, energy

    mask = tf.fill(tf.shape(z)[:-1], True)
    best_z = start_z = z
    best_energy = initial_energy = calc_energy(z, mask=mask)
    step = 0
    while tf.logical_and(step < max_steps, tf.math.reduce_any(mask)):
        lr = schedule(step_size, step_decay, step)
        z = add_noise(z, lr)

        grad, energy = calc_grads(z, mask=mask)
        best_z, best_energy = update_best(z, energy, best_z, best_energy)
        grad *= lr
        z -= grad
        step += 1
        mask = delta < tf.reduce_max(tf.abs(grad), axis=-1)

    res = {
        "z": best_z,
        "energy": best_energy,
        "initial_energy": initial_energy,
        "improvement": tf.squeeze(initial_energy - best_energy, axis=-1),
        "distance": tf.norm(start_z - best_z, axis=-1),
    }
    return {k: tf.stop_gradient(v) for k, v in res.items()}


def find_optimal_latents_adam(
    energy_model,
    z,
    max_steps,
    num_candidates=8,
    step_size=0.03,
    step_decay=0.97,
    max_distance=0.5,
    initial_noise=0.05,
    beta1=0.9,
    beta2=0.999,
    delta=0.0,
):
    def cast(value):
        return tf.cast(value, z.dtype)

    def calc_energy(value, mask):
        tf.debugging.assert_equal(tf.shape(value)[:-1], tf.shape(mask))
        energy = energy_model(value, mask=mask)
        tf.debugging.assert_equal(tf.shape(energy)[:-1], tf.shape(mask))
        tf.debugging.assert_all_finite(energy, "Non-finite latent energy.")
        energy = tf.reduce_sum(energy, axis=-1, keepdims=True)
        tf.debugging.assert_equal(
            tf.shape(energy), tf.concat([tf.shape(mask), [1]], axis=0)
        )
        return energy

    def calc_grads(value, mask):
        value = tf.stop_gradient(value)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(value)
            energy = calc_energy(value, mask)
        grad = tape.gradient(energy, value)
        grad = tf.where(mask[..., None], grad, tf.zeros_like(grad))
        tf.debugging.assert_equal(tf.shape(grad), tf.shape(value))
        tf.debugging.assert_all_finite(grad, "Non-finite latent gradient.")
        return grad, energy

    def project(value, origin):
        offset = value - origin
        distance = tf.norm(offset, axis=-1, keepdims=True)
        scale = tf.minimum(
            cast(1.0), cast(max_distance) / tf.maximum(distance, cast(1e-8))
        )
        value = origin + offset * scale
        tf.debugging.assert_less_equal(
            tf.norm(value - origin, axis=-1), cast(max_distance) + cast(1e-5)
        )
        return value

    def update_best(value, energy, best_value, best_energy):
        better = energy < best_energy
        return tf.where(better, value, best_value), tf.where(
            better, energy, best_energy
        )

    tf.debugging.assert_rank_at_least(z, 2)
    tf.debugging.assert_greater_equal(max_steps, 1)
    tf.debugging.assert_greater_equal(num_candidates, 1)
    tf.debugging.assert_positive(step_size)
    tf.debugging.assert_positive(step_decay)
    tf.debugging.assert_less_equal(step_decay, 1.0)
    tf.debugging.assert_positive(max_distance)
    tf.debugging.assert_non_negative(initial_noise)
    tf.debugging.assert_greater_equal(beta1, 0.0)
    tf.debugging.assert_less(beta1, 1.0)
    tf.debugging.assert_greater_equal(beta2, 0.0)
    tf.debugging.assert_less(beta2, 1.0)
    tf.debugging.assert_non_negative(delta)
    tf.debugging.assert_all_finite(z, "Non-finite initial latent.")

    start_z = tf.stop_gradient(z)
    initial_mask = tf.fill(tf.shape(start_z)[:-1], True)
    initial_energy = calc_energy(start_z, initial_mask)

    candidate_z = tf.repeat(start_z[None], num_candidates, axis=0)
    candidate_start = tf.repeat(start_z[None], num_candidates, axis=0)
    noise = tf.random.normal(
        tf.shape(candidate_z), stddev=cast(initial_noise), dtype=z.dtype
    )
    noise = tf.concat([tf.zeros_like(noise[:1]), noise[1:]], axis=0)
    candidate_z = project(candidate_z + noise, candidate_start)

    best_z = candidate_start
    best_energy = tf.repeat(initial_energy[None], num_candidates, axis=0)
    first_moment = tf.zeros_like(candidate_z)
    second_moment = tf.zeros_like(candidate_z)
    mask = tf.fill(tf.shape(candidate_z)[:-1], True)
    step = tf.constant(0)

    tf.debugging.assert_equal(tf.shape(candidate_z)[1:], tf.shape(start_z))
    tf.debugging.assert_equal(tf.shape(best_energy)[1:-1], tf.shape(start_z)[:-1])
    tf.debugging.assert_equal(tf.shape(mask), tf.shape(candidate_z)[:-1])

    while tf.logical_and(step < max_steps, tf.reduce_any(mask)):
        grad, energy = calc_grads(candidate_z, mask)
        best_z, best_energy = update_best(candidate_z, energy, best_z, best_energy)

        first_moment = cast(beta1) * first_moment + (cast(1.0) - cast(beta1)) * grad
        second_moment = cast(beta2) * second_moment + (
            cast(1.0) - cast(beta2)
        ) * tf.square(grad)

        step_value = cast(step + 1)
        first_hat = first_moment / (cast(1.0) - tf.pow(cast(beta1), step_value))
        second_hat = second_moment / (cast(1.0) - tf.pow(cast(beta2), step_value))
        lr = cast(step_size) * tf.pow(cast(step_decay), cast(step))
        update = lr * first_hat / (tf.sqrt(second_hat) + cast(1e-8))
        update = tf.where(mask[..., None], update, tf.zeros_like(update))

        previous_z = candidate_z
        candidate_z = project(candidate_z - update, candidate_start)
        movement = tf.reduce_max(tf.abs(candidate_z - previous_z), axis=-1)
        mask = movement > cast(delta)
        step += 1

        tf.debugging.assert_equal(tf.shape(update), tf.shape(candidate_z))
        tf.debugging.assert_all_finite(update, "Non-finite Adam update.")
        tf.debugging.assert_all_finite(candidate_z, "Non-finite optimized latent.")

    final_mask = tf.fill(tf.shape(candidate_z)[:-1], True)
    final_energy = calc_energy(candidate_z, final_mask)
    best_z, best_energy = update_best(candidate_z, final_energy, best_z, best_energy)

    best_index = tf.argmin(best_energy[..., 0], axis=0, output_type=tf.int32)
    selector = tf.one_hot(best_index, num_candidates, dtype=z.dtype)
    selector = tf.transpose(
        selector,
        tf.concat([[tf.rank(selector) - 1], tf.range(tf.rank(selector) - 1)], axis=0),
    )
    best_z = tf.reduce_sum(best_z * selector[..., None], axis=0)
    best_energy = tf.reduce_sum(best_energy * selector[..., None], axis=0)

    improvement = initial_energy - best_energy
    distance = tf.norm(start_z - best_z, axis=-1, keepdims=True)

    tf.debugging.assert_equal(tf.shape(best_z), tf.shape(start_z))
    tf.debugging.assert_equal(tf.shape(best_energy), tf.shape(initial_energy))
    tf.debugging.assert_equal(tf.shape(improvement), tf.shape(initial_energy))
    tf.debugging.assert_equal(tf.shape(distance), tf.shape(initial_energy))
    tf.debugging.assert_greater_equal(improvement, cast(-1e-6))
    tf.debugging.assert_less_equal(distance, cast(max_distance) + cast(1e-5))
    tf.debugging.assert_all_finite(best_z, "Non-finite best latent.")
    tf.debugging.assert_all_finite(best_energy, "Non-finite best energy.")
    tf.debugging.assert_all_finite(improvement, "Non-finite latent improvement.")
    tf.debugging.assert_all_finite(distance, "Non-finite latent distance.")

    result = {
        "z": best_z,
        "energy": tf.squeeze(best_energy, axis=-1),
        "initial_energy": tf.squeeze(initial_energy, axis=-1),
        "improvement": tf.squeeze(improvement, axis=-1),
        "distance": tf.squeeze(distance, axis=-1),
    }
    return {name: tf.stop_gradient(value) for name, value in result.items()}
