"""Export EmbeddingsTable model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_embeddingstable(args) -> Dict[str, Any]:
    """Export EmbeddingsTable model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.EmbeddingsTable import EmbeddingsTable
    from NN.models.npz_utils import model_to_dict
    from Core.Constants import HIERARCHY_LEVELS

    # Create test vocabulary
    vocab = {
        "userId": 2,
        "screenId": 2,
        "cameraId": 2,
        "monitorId": 2,
        "placeId": 2,
    }

    model = EmbeddingsTable(vocab=vocab, embedding_size=32)

    # Create test inputs
    batch_size, timesteps = 2, 3

    # Create ID tensors for each hierarchy level
    test_ids = {}
    for level in HIERARCHY_LEVELS:
        test_ids[level] = stabilize_input(np.array([[0], [1]], np.int32))
    shape = stabilize_input(np.array([batch_size, timesteps], dtype=np.int32))

    test_inputs = {
        "shape": shape,
    }

    # Add actual ID tensors
    for level in HIERARCHY_LEVELS:
        test_inputs[level] = test_ids[level]

    # Run model inference
    model(**test_inputs, training=False)
    stabilize(model)
    outputs = model(**test_inputs, training=False)

    # Export weights after model has been built
    model_weights = model_to_dict(model)

    result = {"weights": model_weights}

    if args.test:
        result["inputs"] = test_inputs
        result["outputs"] = {"predictions": outputs.numpy()}

    return result
