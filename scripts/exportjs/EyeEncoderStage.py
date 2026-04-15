"""Export EyeEncoderStage model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_eyeencoderstage(args) -> Dict[str, Any]:
    """Export EyeEncoderStage model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.EyeEncoderStage import EyeEncoderStage
    from NN.models.npz_utils import model_to_dict

    model = EyeEncoderStage(latent_size=32, num_filters=32)

    # Create test inputs to build the model and create weights
    # EyeEncoderStage expects input with shape (batch, height, width, channels)
    rng = np.random.RandomState(42)
    inputs = stabilize_input(
        rng.randn(2, 32, 32, 2).astype(np.float32)
    )  # batch=2, height=32, width=32, channels=2

    # Run model inference first to build the model and create weights
    model(inputs, training=False)
    stabilize(model)
    outputs = model(inputs, training=False)

    # Now export weights after model has been built
    model_weights = model_to_dict(model)

    result = {"weights": model_weights}

    if args.test:
        result["inputs"] = {
            "inputs": inputs,
        }
        result["outputs"] = {
            "feature_map": outputs[0].numpy(),  # First element is feature map
            "latent": outputs[1].numpy(),  # Second element is latent vector
        }

    return result
