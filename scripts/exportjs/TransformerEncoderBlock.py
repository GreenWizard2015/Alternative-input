"""Export TransformerEncoderBlock weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_transformerencoderblock(args) -> Dict[str, Any]:
    """Export TransformerEncoderBlock weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.layers.TransformerEncoderBlock import TransformerEncoderBlock
    from NN.models.npz_utils import model_to_dict

    layer = TransformerEncoderBlock(d_model=64)

    # Create test inputs to build the layer and create weights
    rng = np.random.RandomState(42)
    test_inputs = {
        "input": stabilize_input(rng.randn(2, 10, 64).astype(np.float32)),
    }

    # Run layer inference first to build the layer and create weights
    layer(test_inputs["input"], training=False)
    stabilize(layer)
    outputs = layer(test_inputs["input"], training=False)

    # Now export weights after layer has been built
    layer_weights = model_to_dict(layer)

    result = {"weights": layer_weights}

    if args.test:
        result["inputs"] = test_inputs
        result["outputs"] = {"predictions": outputs.numpy()}

    return result
