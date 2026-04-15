"""Test data loader for Keras model evaluation.

Loads pre-packaged test batches from npz files and provides Keras Sequence interface
for evaluation during training.
"""

from typing import Dict, List, Tuple, Protocol, runtime_checkable, Generator
import tensorflow as tf
import numpy as np
import os
import glob
from Core.landmarks import FACE_MESH_INVALID_VALUE


@runtime_checkable
class ModelTargets(Protocol):
    """Protocol for model target outputs.

    Targets can be either:
    - Dict[str, np.ndarray]: Multiple named outputs (multi-task models)
    - np.ndarray: Single output array (single-task models)

    This protocol serves as documentation for the union type Union[Dict[str, np.ndarray], np.ndarray]
    that is used as the return type for model predictions. Single-task models return a numpy array,
    while multi-task models return a dictionary mapping output names to arrays.
    """

    pass


class TestLoader(tf.keras.utils.Sequence):
    """Sequence loader for test batches from npz files.

    Loads test data from npz (numpy compressed) files where each file contains
    a batch of test samples. Implements Keras Sequence interface for use with
    model.evaluate() and model.predict().

    Attributes:
        _batchesNpz: List of paths to test-*.npz files
    """

    def __init__(self, testFolder: str, batch_size: int = -1) -> None:
        """Initialize test loader with folder containing npz files.

        Args:
            testFolder: Path to folder containing test-*.npz files.
        """
        self._batchesNpz: List[str] = [
            f for f in glob.glob(os.path.join(testFolder, "test-*.npz"))
        ]
        self._batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """Signal end of epoch (currently no-op for test loader)."""
        return

    def __len__(self) -> int:
        """Get number of test batches.

        Returns:
            Number of test-*.npz files in the test folder.
        """
        return len(self._batchesNpz)

    def sample(
        self, idx: int, no_face: bool = False, no_eyes: bool = False
    ) -> Generator[Tuple[Dict[str, np.ndarray], ModelTargets], None, None]:
        """Load a single test batch with modality filtering for ablation studies.

        Args:
            idx: Index of batch to load.
            no_face: If True, remove face mesh points from input
            no_eyes: If True, remove eye images from input

        Returns:
            Tuple of (filtered_input_dict, targets) where:
            - input_dict: Contains filtered input features based on modality settings
            - targets: Either Dict[str, np.ndarray] for multi-output models
                      or np.ndarray for single-output models
        """
        with np.load(self._batchesNpz[idx]) as res:
            res = {k: v for k, v in res.items()}

        X: Dict[str, np.ndarray] = {
            k.replace("X_", ""): v for k, v in res.items() if "X_" in k
        }
        if no_face:
            X["points"] = X["points"] * 0.0 + FACE_MESH_INVALID_VALUE

        if no_eyes:
            X["left eye"] *= 0.0
            X["right eye"] *= 0.0

        Y: Dict[str, np.ndarray] = {
            k.replace("Y_", ""): v for k, v in res.items() if "Y_" in k
        }
        if self._batch_size <= 0:
            yield (X, Y)
            return

        N = len(X["points"])
        for idx in range(0, N, self._batch_size):
            max_idx = min(N, idx + self._batch_size)
            resX = {k: v[idx:max_idx] for k, v in X.items()}

            resY = {k: v[idx:max_idx] for k, v in Y.items()}
            yield (resX, resY)

    def reset(self):
        pass
