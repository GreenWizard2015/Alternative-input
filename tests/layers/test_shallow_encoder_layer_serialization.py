"""Tests for ShallowEncoderLayer serialization."""

import numpy as np
from NN.layers.ShallowEncoderLayer import ShallowEncoderLayer
from NN.models.npz_utils import model_to_dict


def test_shallow_encoder_layer_serialization():
    """Test model_to_dict with real ShallowEncoderLayer using normal construction."""
    # Try to create the layer with normal construction
    layer = ShallowEncoderLayer()

    # Try to build
    layer.build((1, 5, 1))

    # Test model_to_dict
    result = model_to_dict(layer)

    # Verify results
    assert all(isinstance(v, np.ndarray) for v in result.values())
    actual = set(result.keys())
    expected = set(
        [
            "ShallowEncoderLayer_layer/Encoder/CoordsEncoding/CEL_shifts",
            "ShallowEncoderLayer_layer/Encoder/CoordsEncoding/CEL_fusion_w",
            "ShallowEncoderLayer_layer/Encoder/CoordsEncoding/CEL_frequency",
            "ShallowEncoderLayer_layer/Encoder/CoordsEncoding/CEL_freq_deltas",
            "ShallowEncoderLayer_layer/Encoder/CoordsEncoding/CEL_fusion_b",
            "ShallowEncoderLayer_layer/Encoder/CoordsEncoding/CEL_gates",
            "ShallowEncoderLayer_layer/Encoder/CoordsEncoding/_bottleneck/kernel",
        ]
    )
    diff = expected.symmetric_difference(actual)
    assert not diff, f"Diff {diff}"
