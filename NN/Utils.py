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
    "CEL_",  # Exclude CoordsEncodingLayer from weight decay
    "_gate",
    "_PE",
    "_scale",  # Exclude some custom layers variables
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


def structured_latent_dropout(latents, training, min_rate=0.1, max_rate=0.9):
    if not training:
        return latents

    shp = tf.shape(latents)
    keep_rates = tf.linspace(max_rate, min_rate, shp[-1] - 1)
    keep_rates = tf.concat([tf.constant([1.0]), keep_rates], axis=-1)
    keep_rates = tf.broadcast_to(keep_rates, shp)

    shared_seed = tf.random.uniform(tf.shape(keep_rates[..., :1]))
    mask = tf.cast(shared_seed < keep_rates, latents.dtype)
    tf.debugging.assert_equal(tf.shape(mask), tf.shape(latents))
    # inverted dropout scaling
    return latents * mask / keep_rates
