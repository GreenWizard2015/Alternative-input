"""Export EmbeddingsProcessor model weights and test data for JavaScript."""

import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize, stabilize_input


def export_embeddingsprocessor(args) -> Dict[str, Any]:
    """Export EmbeddingsProcessor model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from NN.models.EmbeddingsProcessor import EmbeddingsProcessor
    from NN.models.npz_utils import model_to_dict

    # Test with both mixing methods
    model_attention = EmbeddingsProcessor(embedding_size=32, mixing_method="attention")

    # Create test inputs
    rng = np.random.RandomState(42)

    # Test data: concatenated embeddings with shape [batch, 1, 5*embedding_size]
    batch_size, timesteps = 2, 3
    concatenated_embeddings = stabilize_input(
        rng.randn(batch_size, 1, 5 * 32).astype(np.float32)
    )  # 5 embeddings * 32 embedding_size, scaled by 1e5

    shape = np.array([batch_size, timesteps], dtype=np.int32)

    test_inputs = {
        "concatenated_embeddings": concatenated_embeddings,
        "shape": shape,
    }

    # Run model inference for both mixing methods
    model_attention(**test_inputs, training=False)
    stabilize(model_attention)
    output_attention = model_attention(**test_inputs, training=False)

    # Export weights after models have been built
    result = {"weights": model_to_dict(model_attention)}

    if args.test:
        result["inputs"] = test_inputs
        result["outputs"] = {"predictions": output_attention.numpy()}

    return result
