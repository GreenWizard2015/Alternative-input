"""Export EyeEncoderConv model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_eyeencoderconv(args) -> Dict[str, Any]:
    """Export EyeEncoderConv model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.EyeEncoderConv import EyeEncoderConv
    from NN.models.npz_utils import model_to_dict

    model = EyeEncoderConv(latent_size=32, scale_mult=1.0)

    # Create test inputs to build the model and create weights
    # EyeEncoderConv expects stereo eye images with shape (batch, height, width, channels)
    rng = np.random.RandomState(42)
    eyes_input = stabilize_input(
        rng.randn(2, 32, 32, 2).astype(np.float32)
    )  # batch=2, height=32, width=32, channels=2

    # Run model inference first to build the model and create weights
    model(eyes_input, training=False)
    stabilize(model)
    outputs = model(eyes_input, training=False)

    # Now export weights after model has been built
    model_weights = model_to_dict(model)

    result = {"weights": model_weights}

    if args.test:
        result["inputs"] = {
            "eyes_input": eyes_input,
        }
        result["outputs"] = {}

        # EyeEncoderConv's model returns outputs where each element is [conv_output, latent_output]
        for i, stage_outputs in enumerate(outputs):
            result["outputs"][f"scale_{i}_latent"] = stage_outputs.numpy()

    return result
