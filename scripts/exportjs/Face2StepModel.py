"""Export Face2StepModel model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_face2stepmodel(args) -> Dict[str, Any]:
    """Export Face2StepModel model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs
        latent_size: Base dimension of latent feature representations
        scale_mult: Scaling multiplier for filter dimensions

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.Face2StepModel import Face2StepModel
    from NN.models.npz_utils import model_to_dict

    model = Face2StepModel(latent_size=32, scale_mult=1.0)

    # Create test inputs to build the model and create weights
    from Core.Constants import FACEMESH_LANDMARK_COUNT

    rng = np.random.RandomState(42)
    batch_size = 2
    seq_len = 1

    # Face mesh points: (batch, seq_len, FACEMESH_LANDMARK_COUNT, 2)
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

    # Embeddings: (batch, seq_len, 32)
    embeddings = stabilize_input(rng.randn(batch_size, seq_len, 32).astype(np.float32))

    test_inputs = {
        "points": points,
        "left eye": left_eye,
        "right eye": right_eye,
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
        result["outputs"] = {"predictions": outputs.numpy()}

    return result
