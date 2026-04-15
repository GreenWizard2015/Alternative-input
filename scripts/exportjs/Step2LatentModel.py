"""Export Step2LatentModel model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_step2latentmodel(args) -> Dict[str, Any]:
    """Export Step2LatentModel model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.Step2LatentModel import Step2LatentModel
    from NN.models.npz_utils import model_to_dict

    model = Step2LatentModel(latent_size=32, scale_mult=1.0)

    # Create test inputs to build the model and create weights
    # Step2LatentModel expects dictionary with keys: latent, time, embeddings
    rng = np.random.RandomState(42)
    batch_size = 2
    seq_len = 5

    # latent: (batch, seq_len, latent_size)
    latent = stabilize_input(rng.randn(batch_size, seq_len, 32).astype(np.float32))

    # time: (batch, seq_len, 1)
    time = stabilize_input(rng.randn(batch_size, seq_len, 1).astype(np.float32))

    # embeddings: (batch, seq_len, embeddings_size) - let's use 64 for embeddings_size
    embeddings = stabilize_input(rng.randn(batch_size, seq_len, 64).astype(np.float32))

    inputs = {
        "latent": latent,
        "time": time,
        "embeddings": embeddings,
    }

    # Run model inference first to build the model and create weights
    model(inputs, training=False)
    stabilize(model)
    outputs = model(inputs, training=False)

    # Now export weights after model has been built
    model_weights = model_to_dict(model)

    result = {"weights": model_weights}

    if args.test:
        result["inputs"] = inputs
        result["outputs"] = {
            "predictions": outputs.numpy(),
        }

    return result
