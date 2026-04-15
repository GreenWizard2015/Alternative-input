"""Shared fixtures for embedding tests."""

import pytest
import tensorflow as tf

# ============================================================================
# CONSOLIDATED CONFIG DICTS (SINGLE SOURCE OF TRUTH)
# ============================================================================

MODEL_CONFIGS = {
    # All ModelWrapper configurations for testing (single source of truth)
    "default": {"timesteps": 2, "embeddingSize": 8, "latent_size": 64},
    "minimal": {"timesteps": 1, "embeddingSize": 4, "latent_size": 32},
    "timesteps_1": {"timesteps": 1, "embeddingSize": 4, "latent_size": 32},
    "timesteps_2": {"timesteps": 2, "embeddingSize": 8, "latent_size": 64},
    "timesteps_3": {"timesteps": 3, "embeddingSize": 16, "latent_size": 128},
    "fast": {"timesteps": 2, "embeddingSize": 8, "latent_size": 64},
    "balanced": {"timesteps": 3, "embeddingSize": 16, "latent_size": 128},
}


@pytest.fixture
def stats_data():
    """Create sample stats data."""
    return {
        "userId": [f"user_{i}" for i in range(10)],
        "placeId": [f"place_{i}" for i in range(5)],
        "screenId": [f"screen_{i}" for i in range(8)],
        "cameraId": [f"camera_{i}" for i in range(3)],
        "monitorId": [f"monitor_{i}" for i in range(4)],
    }


@pytest.fixture(scope="session")
def cached_model_wrappers():
    """Pre-built ModelWrapper instances from MODEL_CONFIGS (single source of truth).

    Creates one model instance per configuration and reuses across all tests.
    All configs are defined in MODEL_CONFIGS at the top of this file.
    """
    from Core.models.ModelWrapper import ModelWrapper

    # Create stats data locally to avoid scope mismatch with function-scoped fixture
    stats_data = {
        "userId": [f"user_{i}" for i in range(10)],
        "placeId": [f"place_{i}" for i in range(5)],
        "screenId": [f"screen_{i}" for i in range(8)],
        "cameraId": [f"camera_{i}" for i in range(3)],
        "monitorId": [f"monitor_{i}" for i in range(4)],
    }

    wrappers = {}
    for name, config in MODEL_CONFIGS.items():
        wrapper = ModelWrapper(stats=stats_data, **config)

        # Build dummy inputs dynamically based on timesteps config
        dummy_inputs = {
            "points": tf.ones((1, config["timesteps"], 478, 2), dtype=tf.float32),
            "left eye": tf.ones((1, config["timesteps"], 32, 32, 1), dtype=tf.float32),
            "right eye": tf.ones((1, config["timesteps"], 32, 32, 1), dtype=tf.float32),
            "time": tf.ones((1, config["timesteps"], 1), dtype=tf.float32),
            "userId": tf.constant([[0] * config["timesteps"]], dtype=tf.int32),
            "placeId": tf.constant([[0] * config["timesteps"]], dtype=tf.int32),
            "screenId": tf.constant([[0] * config["timesteps"]], dtype=tf.int32),
            "cameraId": tf.constant([[0] * config["timesteps"]], dtype=tf.int32),
            "monitorId": tf.constant([[0] * config["timesteps"]], dtype=tf.int32),
        }
        wrapper.call(dummy_inputs, training=False)
        wrappers[name] = wrapper

    return wrappers


@pytest.fixture(scope="session")
def shared_test_inputs():
    """Create reusable test inputs for consistent testing.

    Returns a dictionary of input tensors that can be reused across
    multiple tests, ensuring consistency and reducing setup time.
    """
    import numpy as np

    # Fixed seed for reproducible test results
    np.random.seed(42)

    return {
        "small_batch": {
            "points": tf.constant(
                np.random.normal(size=(1, 2, 478, 2)).astype(np.float32)
            ),
            "left eye": tf.constant(
                np.random.normal(size=(1, 2, 32, 32, 1)).astype(np.float32)
            ),
            "right eye": tf.constant(
                np.random.normal(size=(1, 2, 32, 32, 1)).astype(np.float32)
            ),
            "time": tf.ones((1, 2, 1), dtype=tf.float32),
            "userId": tf.constant([[0, 0]], dtype=tf.int32),
            "placeId": tf.constant([[0, 0]], dtype=tf.int32),
            "screenId": tf.constant([[0, 0]], dtype=tf.int32),
            "cameraId": tf.constant([[0, 0]], dtype=tf.int32),
            "monitorId": tf.constant([[0, 0]], dtype=tf.int32),
        },
        "medium_batch": {
            "points": tf.constant(
                np.random.normal(size=(2, 2, 478, 2)).astype(np.float32)
            ),
            "left eye": tf.constant(
                np.random.normal(size=(2, 2, 32, 32, 1)).astype(np.float32)
            ),
            "right eye": tf.constant(
                np.random.normal(size=(2, 2, 32, 32, 1)).astype(np.float32)
            ),
            "time": tf.ones((2, 2, 1), dtype=tf.float32),
            "userId": tf.constant([[0, 0], [1, 1]], dtype=tf.int32),
            "placeId": tf.constant([[0, 0], [1, 1]], dtype=tf.int32),
            "screenId": tf.constant([[0, 0], [1, 1]], dtype=tf.int32),
            "cameraId": tf.constant([[0, 0], [0, 0]], dtype=tf.int32),
            "monitorId": tf.constant([[0, 0], [0, 0]], dtype=tf.int32),
        },
        "small_batch_timesteps_1": {
            "points": tf.ones((1, 1, 478, 2), dtype=tf.float32),
            "left eye": tf.ones((1, 1, 32, 32, 1), dtype=tf.float32),
            "right eye": tf.ones((1, 1, 32, 32, 1), dtype=tf.float32),
            "time": tf.ones((1, 1, 1), dtype=tf.float32),
            "userId": tf.constant([[0]], dtype=tf.int32),
            "placeId": tf.constant([[0]], dtype=tf.int32),
            "screenId": tf.constant([[0]], dtype=tf.int32),
            "cameraId": tf.constant([[0]], dtype=tf.int32),
            "monitorId": tf.constant([[0]], dtype=tf.int32),
        },
        "small_batch_timesteps_3": {
            "points": tf.ones((1, 3, 478, 2), dtype=tf.float32),
            "left eye": tf.ones((1, 3, 32, 32, 1), dtype=tf.float32),
            "right eye": tf.ones((1, 3, 32, 32, 1), dtype=tf.float32),
            "time": tf.ones((1, 3, 1), dtype=tf.float32),
            "userId": tf.constant([[0, 0, 0]], dtype=tf.int32),
            "placeId": tf.constant([[0, 0, 0]], dtype=tf.int32),
            "screenId": tf.constant([[0, 0, 0]], dtype=tf.int32),
            "cameraId": tf.constant([[0, 0, 0]], dtype=tf.int32),
            "monitorId": tf.constant([[0, 0, 0]], dtype=tf.int32),
        },
    }
