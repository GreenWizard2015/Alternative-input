"""Export CoordsEncodingLayer weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_coordsencodinglayer(args) -> Dict[str, Any]:
    """Export CoordsEncodingLayer weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.layers.CoordsEncodingLayer import CoordsEncodingLayer
    from NN.models.npz_utils import model_to_dict

    layer = CoordsEncodingLayer(N=32)
    # Create simple test inputs for CoordsEncodingLayer
    rng = np.random.RandomState(42)
    test_inputs = {
        "input": stabilize_input(rng.randn(2, 3, 2).astype(np.float32)),
    }

    # Run layer inference
    layer(
        test_inputs["input"], training=False
    )  # This builds the layer and creates weights
    stabilize(layer)
    outputs = layer(test_inputs["input"], training=False)

    result = {"weights": model_to_dict(layer)}
    if args.test:
        result["inputs"] = test_inputs
        result["outputs"] = {"predictions": outputs.numpy()}

    return result
