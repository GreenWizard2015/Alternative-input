"""Export EyeEncoder model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_eyeencoder(args) -> Dict[str, Any]:
    """Export EyeEncoder model weights and test data.

    Args:
        test: If True, also package test results and test_inputs
        latent_size: Base dimension of latent feature representations
        scale_mult: Scaling multiplier for filter dimensions

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.EyeEncoder import EyeEncoder
    from NN.models.npz_utils import model_to_dict

    model = EyeEncoder(latent_size=32, scale_mult=1.0)

    # Create test inputs to build the model and create weights
    # EyeEncoder expects list of [left_eye, right_eye] with shape (batch, height, width, channels)
    rng = np.random.RandomState(42)
    left_eye = stabilize_input(rng.randn(1, 32, 32, 1).astype(np.float32))
    right_eye = stabilize_input(rng.randn(1, 32, 32, 1).astype(np.float32))

    # Run model inference first to build the model and create weights
    model([left_eye, right_eye], training=False)
    stabilize(model)
    outputs = model([left_eye, right_eye], training=False)

    # Now export weights after model has been built
    model_weights = model_to_dict(model)

    result = {"weights": model_weights}

    if args.test:
        result["inputs"] = {
            "left_eye": left_eye,
            "right_eye": right_eye,
        }
        result["outputs"] = {
            "predictions": outputs.numpy(),
        }

    return result
