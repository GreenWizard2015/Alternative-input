"""Export ModelWrapper model weights and test data for JavaScript."""

import tensorflow as tf
import numpy as np
from typing import Dict, Any
from scripts.exportjs.utils import stabilize_input


def stabilize(model):
    from NN.models.npz_utils import model_to_raw_dict

    weights = model_to_raw_dict(model)
    for w in weights.values():
        w.assign(tf.where(0.0 < w, 0.01, -0.01))


def export_modelwrapper(args) -> Dict[str, Any]:
    """Export ModelWrapper model weights and test data.

    Args:
        args: Object with test attribute. If args.test is True, also package test results and test_inputs

    Returns:
        Dictionary with weights, test_inputs, test_outputs, and metadata
    """
    from Core.models.ModelWrapper import ModelWrapper
    from NN.models.npz_utils import model_to_dict

    seq_len = 5
    model = ModelWrapper(
        timesteps=seq_len,
        stats={
            "userId": ["user1", "user2", "user3"],
            "screenId": ["screen1", "screen2"],
            "cameraId": ["camera1", "camera2"],
            "monitorId": ["monitor1", "monitor2"],
            "placeId": ["place1", "place2"],
        },
    )

    # Create test inputs to build the model and create weights
    from Core.Constants import FACEMESH_LANDMARK_COUNT

    rng = np.random.RandomState(42)
    batch_size = 2

    # Points: (batch, seq_len, FACEMESH_LANDMARK_COUNT, 2)
    points = stabilize_input(
        rng.randn(batch_size, seq_len, FACEMESH_LANDMARK_COUNT, 2).astype(np.float32)
    )

    # Left eye: (batch, seq_len, 32, 32, 1)
    left_eye = stabilize_input(
        rng.randn(batch_size, seq_len, 32, 32, 1).astype(np.float32)
    )

    # Right eye: (batch, seq_len, 32, 32, 1)
    right_eye = stabilize_input(
        rng.randn(batch_size, seq_len, 32, 32, 1).astype(np.float32)
    )

    # Time: (batch, seq_len, 1)
    time = stabilize_input(rng.randn(batch_size, seq_len, 1).astype(np.float32))

    # Required IDs for ModelWrapper
    user_id = np.full(
        (batch_size, seq_len, 1), 0, dtype=np.int32
    )  # 0 = first user in stats
    screen_id = np.full(
        (batch_size, seq_len, 1), 0, dtype=np.int32
    )  # 0 = first screen in stats
    camera_id = np.full(
        (batch_size, seq_len, 1), 0, dtype=np.int32
    )  # 0 = first camera in stats
    monitor_id = np.full(
        (batch_size, seq_len, 1), 0, dtype=np.int32
    )  # 0 = first monitor in stats
    place_id = np.full(
        (batch_size, seq_len, 1), 0, dtype=np.int32
    )  # 0 = first place in stats

    test_inputs = {
        "points": points,
        "left eye": left_eye,
        "right eye": right_eye,
        "time": time,
        "userId": user_id,
        "screenId": screen_id,
        "cameraId": camera_id,
        "monitorId": monitor_id,
        "placeId": place_id,
    }
    # Run model inference first to build the model and create weights
    _ = model(test_inputs, training=False)

    # stabilize weights after building the model
    for name in ["_model", "_predictor", "_table", "_processor"]:
        if hasattr(model, name):
            field = getattr(model, name)
            stabilize(field)

    outputs = model(test_inputs, training=False)
    # Now export weights after model has been stabilized
    model_weights = {}
    # Export key submodel fields
    for name in ["_model", "_predictor", "_table", "_processor"]:
        if hasattr(model, name):
            field = getattr(model, name)
            weights = model_to_dict(field)
            weights = {f"{name}/{k}": v for k, v in weights.items()}
            model_weights.update(weights)

    result = {"weights": model_weights}

    if args.test:
        result["inputs"] = test_inputs
        result["outputs"] = {"predictions": outputs.result}

    return result
