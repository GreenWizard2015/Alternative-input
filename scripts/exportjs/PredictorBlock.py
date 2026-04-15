"""Export PredictorBlock model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_predictorblock(args) -> Dict[str, Any]:
    """Export PredictorBlock model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.PredictorBlock import PredictorBlock
    from NN.models.npz_utils import model_to_dict

    model = PredictorBlock(mode="result")

    # Create test inputs to build the model and create weights
    # PredictorBlock expects input tensor of shape (batch, seq_len, latent_dim)
    rng = np.random.RandomState(42)
    input_tensor = stabilize_input(
        rng.randn(2, 10, 32).astype(np.float32)
    )  # batch=2, seq_len=10, latent_dim=32

    # Run model inference first to build the model and create weights
    model(input_tensor, training=False)
    stabilize(model)
    outputs = model(input_tensor, training=False)

    # Now export weights after model has been built
    model_weights = model_to_dict(model)

    result = {"weights": model_weights}

    if args.test:
        result["inputs"] = {
            "input": input_tensor,
        }
        result["outputs"] = {key: value.numpy() for key, value in outputs.items()}

    return result
