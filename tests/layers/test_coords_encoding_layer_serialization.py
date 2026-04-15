"""Tests for CoordsEncodingLayer serialization."""

import numpy as np
from NN.layers.CoordsEncodingLayer import CoordsEncodingLayer
from NN.models.npz_utils import model_to_dict


def test_coords_encoding_layer_serialization():
    """Test model_to_dict with real CoordsEncodingLayer using normal construction."""
    # Try to create the layer with normal construction
    layer = CoordsEncodingLayer(
        N=4,
        max_frequency=5.0,
        use_shifts=True,
        use_low_bands=True,
        use_high_bands=True,
        final_dropout=0.4,
        bands_dropout=True,
        shared_transformation=False,
    )

    # Try to build
    layer.build((1, 5, 4))

    # Test model_to_dict
    result = model_to_dict(layer)

    # Verify results
    assert all(isinstance(v, np.ndarray) for v in result.values())
    actual = set(result.keys())
    expected = set(
        [
            "CoordsEncodingLayer_layer/CEL_frequency",
            "CoordsEncodingLayer_layer/CEL_gates",
            "CoordsEncodingLayer_layer/CEL_shifts",
            "CoordsEncodingLayer_layer/CEL_fusion_w",
            "CoordsEncodingLayer_layer/CEL_freq_deltas",
            "CoordsEncodingLayer_layer/CEL_fusion_b",
            "CoordsEncodingLayer_layer/_bottleneck/kernel",
        ]
    )
    diff = expected.symmetric_difference(actual)
    assert not diff, f"Diff {diff}"
