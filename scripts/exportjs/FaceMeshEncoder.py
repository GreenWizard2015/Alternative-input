"""Export FaceMeshEncoder model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_facemeshencoder(args) -> Dict[str, Any]:
    """Export FaceMeshEncoder model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.FaceMeshEncoder import FaceMeshEncoder
    from NN.models.npz_utils import model_to_dict

    model = FaceMeshEncoder(latent_size=32)

    # Create test inputs to build the model and create weights
    # FaceMeshEncoder expects facial landmarks with shape (batch, num_points, coord_dim)
    rng = np.random.RandomState(42)
    # 478 facial landmarks with 2 coordinates each (x, y)
    landmarks_input = stabilize_input(
        rng.randn(2, 478, 2).astype(np.float32)
    )  # batch=2, 478 points, 2 coordinates

    # Run model inference first to build the model and create weights
    model(landmarks_input, training=False)
    stabilize(model)
    outputs = model(landmarks_input, training=False)

    # Now export weights after model has been built
    model_weights = model_to_dict(model)

    result = {"weights": model_weights}

    if args.test:
        result["inputs"] = {
            "input": landmarks_input,
        }
        result["outputs"] = {
            "predictions": outputs.numpy(),
        }

    return result
