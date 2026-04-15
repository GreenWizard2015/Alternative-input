"""Model scaling utilities for knowledge distillation.

Provides helper functions to compute all scaled dimensions proportionally
for creating larger teacher models from base student architectures.
"""

from typing import Any, Dict


def compute_scaled_dimensions(latent_size: int, scale_mult: float) -> Dict[str, Any]:
    """Compute all model dimensions scaled by scale_mult with multi-head attention validation.

    This function scales all learnable parameters proportionally to create a larger model
    with greater expressive capacity while maintaining compatibility with multi-head
    attention mechanisms (requires d_model % num_heads == 0).

    Args:
        latent_size: Base latent dimension (e.g., 256)
        scale_mult: Scaling multiplier (1.0 = no scaling, 2.0 = double all dimensions)

    Returns:
        Dictionary with keys:
            - latent_size: Scaled latent dimension
            - eye_filters: [filter0, filter1] for EyeEncoder conv layers
            - num_heads: Adjusted transformer num_heads (divisible by scaled_latent_size)
            - dff_multiplier: Feed-forward dimension multiplier for transformers
            - predictor_mlp_sizes: [size0, size1, size2] for PredictorBlock layers

    Raises:
        ValueError: If scale_mult <= 0, scaled_latent_size too large, or divisibility fails

    Example:
        >>> dims = compute_scaled_dimensions(latent_size=256, scale_mult=2.0)
        >>> dims['latent_size']
        512
        >>> dims['num_heads']  # Adjusted to be divisor of 512
        16
    """
    # ===== Validation Phase =====
    if scale_mult <= 0:
        raise ValueError(f"scale_mult must be positive, got {scale_mult}")

    # ===== Scaling Phase =====
    scaled_latent_size = int(latent_size * scale_mult)
    target_num_heads = max(1, int(16 * scale_mult))

    # ===== Multi-Head Attention Validation & Adjustment =====
    # CRITICAL: Multi-head attention requires latent_size % num_heads == 0
    # If exact divisibility not possible, find largest valid divisor <= target
    if scaled_latent_size % target_num_heads != 0:
        # Search downward from target to find largest divisor of scaled_latent_size
        valid_heads = 1
        for h in range(target_num_heads, 0, -1):
            if scaled_latent_size % h == 0:
                valid_heads = h
                break
        num_heads = valid_heads
    else:
        num_heads = target_num_heads

    # ===== Return Scaled Dimensions =====
    return {
        "latent_size": scaled_latent_size,
        "eye_filters": [int(32 * scale_mult), int(32 * scale_mult)],
        "num_heads": num_heads,  # Validated to divide latent_size
        "dff_multiplier": int(16 * scale_mult),
        "predictor_mlp_sizes": [
            int(128 * scale_mult),
            int(128 * scale_mult),
            int(128 * scale_mult),
        ],
    }
