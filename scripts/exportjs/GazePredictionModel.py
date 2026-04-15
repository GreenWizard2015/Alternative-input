"""Export GazePredictionModel model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_gazepredictionmodel(args) -> Dict[str, Any]:
    """Export GazePredictionModel model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.GazePredictionModel import GazePredictionModel
    from NN.models.npz_utils import model_to_dict

    model = GazePredictionModel()

    # Create test inputs to build the model and create weights
    from Core.Constants import FACEMESH_LANDMARK_COUNT

    rng = np.random.RandomState(42)
    batch_size = 2
    seq_len = 1
    emb_size = 5  # Common embedding size

    # Points: (batch, seq_len, FACEMESH_LANDMARK_COUNT, 2)
    points = stabilize_input(
        rng.randn(batch_size, seq_len, FACEMESH_LANDMARK_COUNT, 2).astype(np.float32)
    )

    # Left eye: (batch, seq_len, 32, 32, 1)
    left_eye = stabilize_input(
        rng.randn(batch_size, seq_len, 32, 32, 1).astype(np.float32)
    )

    # Right eye: (batch, seq_len, 32, 32, 1)
    right_eye = stabilize_input(
        rng.randn(batch_size, seq_len, 32, 32, 1).astype(np.float32)
    )

    # Time: (batch, seq_len, 1)
    time = stabilize_input(rng.randn(batch_size, seq_len, 1).astype(np.float32))

    # Embeddings: (batch, seq_len, 5*emb_size)
    embeddings = stabilize_input(
        rng.randn(batch_size, seq_len, 5 * emb_size).astype(np.float32)
    )

    test_inputs = {
        "points": points,
        "left eye": left_eye,
        "right eye": right_eye,
        "time": time,
        "embeddings": embeddings,
    }

    # Run model inference first to build the model and create weights
    model(test_inputs, training=False)
    stabilize(model)
    outputs = model(test_inputs, training=False)

    # Now export weights after model has been built
    model_weights = model_to_dict(model)

    result = {"weights": model_weights}

    if args.test:
        result["inputs"] = test_inputs
        result["outputs"] = {
            "intermediate_latent": outputs["intermediate_latent"].numpy(),
            "final_latent": outputs["final_latent"].numpy(),
        }

    return result
