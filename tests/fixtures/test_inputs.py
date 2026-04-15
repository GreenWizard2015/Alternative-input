"""Factory functions for creating test inputs with flexible configuration."""

from typing import Dict, Any, Optional
import tensorflow as tf
from Core.Utils import FACE_MESH_POINTS


def create_test_inputs(
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = 1,
    timesteps: int = 2,
    input_type: str = "ones",
    include_embeddings: bool = False,
    include_ids: bool = True,
    latent_size: Optional[int] = None,
) -> Dict[str, tf.Tensor]:
    """Create test inputs with flexible configuration.

    Args:
        config: Configuration dict for legacy API (optional)
        batch_size: Batch size for inputs (default: 1)
        timesteps: Number of timesteps (default: 2)
        input_type: Type of input values - "ones", "random_normal", "random_uniform" (default: "ones")
        include_embeddings: Whether to include embeddings field (default: False)
        include_ids: Whether to include ID fields (userId, placeId, etc.) (default: True)
        latent_size: If provided, includes latent field for Step2Latent models

    Returns:
        Dictionary of test input tensors

    Examples:
        # Basic inputs with ones
        inputs = create_test_inputs(batch_size=2, timesteps=5)

        # Random normal inputs with embeddings
        inputs = create_test_inputs(
            batch_size=2, timesteps=5, input_type="random_normal", include_embeddings=True
        )

        # For Step2Latent models
        inputs = create_test_inputs(
            batch_size=2, timesteps=5, latent_size=256, input_type="random_normal"
        )

        # Legacy API (config dict as first positional argument)
        inputs = create_test_inputs({"timesteps": 5}, batch_size=2)
    """
    # Support legacy config parameter
    if config is not None:
        timesteps = config.get("timesteps", timesteps)

    # Helper function to create tensor based on input_type
    def create_tensor(shape, dtype=tf.float32):
        """Create tensor of specified shape with configured method."""
        if input_type == "random_normal":
            return tf.random.normal(shape, dtype=dtype)
        elif input_type == "random_uniform":
            return tf.random.uniform(shape, dtype=dtype)
        else:  # "ones" or default
            return tf.ones(shape, dtype=dtype)

    inputs = {}

    # Facial and eye inputs
    inputs["points"] = create_tensor((batch_size, timesteps, FACE_MESH_POINTS, 2))
    inputs["left eye"] = create_tensor((batch_size, timesteps, 32, 32, 1))
    inputs["right eye"] = create_tensor((batch_size, timesteps, 32, 32, 1))

    # Time input
    if input_type == "random_uniform":
        inputs["time"] = tf.random.uniform(
            (batch_size, timesteps, 1), 0, 1, dtype=tf.float32
        )
    else:
        inputs["time"] = create_tensor((batch_size, timesteps, 1))

    # Optional: embeddings (used by some models)
    if include_embeddings:
        inputs["embeddings"] = create_tensor((batch_size, timesteps, 160))

    # Optional: latent field for Step2Latent models
    if latent_size is not None:
        inputs["latent"] = create_tensor((batch_size, timesteps, latent_size))

    # Optional: ID fields
    if include_ids:
        # Create ID tensors with proper shape - one ID per sample
        inputs["userId"] = tf.constant([[0]] * batch_size, dtype=tf.int32)
        inputs["placeId"] = tf.constant([[0]] * batch_size, dtype=tf.int32)
        inputs["screenId"] = tf.constant([[0]] * batch_size, dtype=tf.int32)
        inputs["cameraId"] = tf.constant([[0]] * batch_size, dtype=tf.int32)
        inputs["monitorId"] = tf.constant([[0]] * batch_size, dtype=tf.int32)

    return inputs
