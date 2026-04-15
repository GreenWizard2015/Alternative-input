"""Testing script for neural network models with dummy data.

This module provides integration tests for all main neural network model classes
used for eye-tracking and gaze prediction. It tests models from the NN.models
subpackage to verify they work correctly with dummy data.

Tested Models:
    - GazePredictionModel: Complete end-to-end model (Face2Step + Step2Latent)

Run with: python NN/networks.py
"""

import tensorflow as tf
from NN.models.GazePredictionModel import GazePredictionModel
from Core.Utils import FACE_MESH_POINTS
from Core.logging_config import get_logger

# Test configuration constants
TEST_BATCH_SIZE = 2
TEST_STEPS = 5
TEST_EYE_SIZE = 32
TEST_LATENT_SIZE = 256
TEST_EMB_SIZE = 64
TEST_KP = 5
SEPARATOR_WIDTH = 80


def _run_tests() -> None:
    """Run all neural network model tests with dummy data."""
    logger = get_logger(__name__)
    separator = "=" * SEPARATOR_WIDTH
    logger.info(separator)
    logger.info("Testing All Main/Top-Level Models with Dummy Data")
    logger.info(separator)

    # Test 1: GazePredictionModel
    separator_short = "-" * SEPARATOR_WIDTH
    logger.info("\n[1] GazePredictionModel")
    logger.info(separator_short)
    gaze_model = GazePredictionModel(
        pointsN=FACE_MESH_POINTS,
        eyeSize=TEST_EYE_SIZE,
        steps=TEST_STEPS,
        latent_size=TEST_LATENT_SIZE,
        embeddings=dict(size=TEST_EMB_SIZE),
    )
    logger.info("GazePredictionModel instantiated successfully")
    logger.info("  Model type: %s", type(gaze_model).__name__)
    logger.info("  Is tf.keras.Model: %s", isinstance(gaze_model, tf.keras.Model))

    # Create dummy inputs - GazePredictionModel expects 5 embeddings (userId, screenId, cameraId, monitorId, placeId)
    dummy_inputs = {
        "points": tf.random.normal((TEST_BATCH_SIZE, TEST_STEPS, FACE_MESH_POINTS, 2)),
        "left eye": tf.random.normal(
            (TEST_BATCH_SIZE, TEST_STEPS, TEST_EYE_SIZE, TEST_EYE_SIZE, 1)
        ),
        "right eye": tf.random.normal(
            (TEST_BATCH_SIZE, TEST_STEPS, TEST_EYE_SIZE, TEST_EYE_SIZE, 1)
        ),
        "time": tf.random.uniform((TEST_BATCH_SIZE, TEST_STEPS, 1), 0, 1),
        "userId": tf.random.normal((TEST_BATCH_SIZE, TEST_STEPS, TEST_EMB_SIZE)),
        "screenId": tf.random.normal((TEST_BATCH_SIZE, TEST_STEPS, TEST_EMB_SIZE)),
        "cameraId": tf.random.normal((TEST_BATCH_SIZE, TEST_STEPS, TEST_EMB_SIZE)),
        "monitorId": tf.random.normal((TEST_BATCH_SIZE, TEST_STEPS, TEST_EMB_SIZE)),
        "placeId": tf.random.normal((TEST_BATCH_SIZE, TEST_STEPS, TEST_EMB_SIZE)),
    }

    # Forward pass
    output = gaze_model(inputs=dummy_inputs, training=False)
    logger.info("Forward pass successful")
    logger.info("  Output keys: %s", list(output.keys()))
    if "result" in output:
        logger.info("  Output shape: %s", output["result"].shape)

    logger.info("\n" + separator)
    logger.info("All tests completed!")
    logger.info(separator)


if __name__ == "__main__":
    _run_tests()
