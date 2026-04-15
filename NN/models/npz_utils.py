"""NPZ serialization utilities for Keras models."""

import numpy as np
from typing import Any
import os


def model_to_raw_dict(model: Any) -> None:
    result = {weight.path: weight for weight in model.weights}
    return result


def model_to_dict(model: Any) -> None:
    """Convert model weights to a dictionary with numpy arrays.

    Args:
        model: Keras model or layer with weights

    Returns:
        Dictionary mapping weight paths to numpy arrays
    """
    result = {name: weight.numpy() for name, weight in model_to_raw_dict(model).items()}
    return result


def save_model_to_npz(model: Any, file_path: str) -> None:
    """Save model weights to NPZ file."""
    npz_path = f"{file_path}.npz"
    weights_dict = model_to_dict(model)
    np.savez_compressed(npz_path, **weights_dict)


def load_model_from_npz(model: Any, file_path: str, force: bool = False) -> None:
    """Load model weights from NPZ file with force support.

    Args:
        model: Keras model to load weights into
        file_path: Path to NPZ file (without .npz extension)
        force: Whether to continue with random initialization for missing weights

    Raises:
        FileNotFoundError: If NPZ file doesn't exist
        ValueError: If weights are missing and force=False, or if loaded weight shape mismatches
    """
    npz_path = f"{file_path}.npz"
    if not os.path.exists(npz_path) and force:
        return
    data = np.load(npz_path, allow_pickle=True)

    missing_layers = []

    for weight in model.weights:
        if weight.path in data.files:
            try:
                loaded_weight = data[weight.path]
                # Assert shape matches before assignment
                if loaded_weight.shape != weight.shape:
                    raise ValueError(
                        f"Shape mismatch for weight '{weight.path}': "
                        f"expected {weight.shape}, got {loaded_weight.shape}"
                    )
                weight.assign(loaded_weight)
            except Exception as e:
                if not force:
                    raise e
        else:
            missing_layers.append(weight.path)

    if missing_layers and not force:
        raise ValueError(f"Missing {missing_layers} layers in '{file_path}' file")
