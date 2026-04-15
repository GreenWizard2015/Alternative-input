"""Context manager for collecting and saving training samples to disk.

Collects samples in memory and periodically flushes them to disk as
npz (numpy compressed) files for efficient storage and batch loading.
"""

from typing import Any, Dict, List, Callable
import Core.Utils as Utils
import numpy as np
from collections import defaultdict
import os
import time
from Core.logging_config import get_logger

logger = get_logger(__name__)


class Dataset:
    """Context manager for dataset sample collection and storage.

    Accumulates training samples in memory and periodically saves them to disk
    as npz files. Useful for collecting data from streaming sources during
    training loop.

    Attributes:
        _timesteps: Number of timesteps per sample
        _totalSamples: Total count of samples processed
        _storedSamples: List of samples waiting to be saved
        _samplesPerChunk: Batch size for disk writing (default: 1000)
        _storeTo: Lambda function to generate output file paths
    """

    def __init__(self, folder: str, timesteps: int) -> None:
        """Initialize dataset with folder for sample storage.

        Args:
            folder: Directory path where samples will be saved.
            timesteps: Number of timesteps per sample.
        """
        self._timesteps: int = timesteps

        self._totalSamples: int = Utils.count_samples_in(folder)
        self._storedSamples: List[Dict[str, Any]] = []
        self._samplesPerChunk: int = 1000
        os.makedirs(folder, exist_ok=True)
        self._storeTo: Callable[[], str] = lambda: os.path.join(
            folder, f"{int(time.time() * 1000)}.npz"
        )

    def __enter__(self) -> "Dataset":
        """Context manager entry.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        """Context manager exit - flushes remaining samples to disk.

        Args:
            type: Exception type if exception occurred.
            value: Exception value if exception occurred.
            traceback: Exception traceback if exception occurred.
        """
        if 0 < len(self._storedSamples):
            self._saveSamples()

    def store(self, data: Dict[str, Any], goal: Any) -> None:
        """Store a sample and flush to disk if threshold is reached.

        Args:
            data: Raw tracked data from tracking system.
            goal: Target/goal for this sample (e.g., gaze coordinates).
        """
        data = Utils.tracked2sample(data)
        sample = {**data, "goal": goal}
        self._storedSamples.append(sample)
        self._totalSamples += 1

        if self._samplesPerChunk <= len(self._storedSamples):
            self._saveSamples()

    def _saveSamples(self) -> None:
        """Flush accumulated samples to disk as npz file.

        Converts list of samples to arrays and saves using numpy.savez.
        Clears the in-memory buffer after successful save.
        """
        data: Dict[str, List[Any]] = defaultdict(list)
        for sample in self._storedSamples:
            for k, v in sample.items():
                data[k].append(v)
        data_arrays: Dict[str, np.ndarray] = {k: np.array(v) for k, v in data.items()}

        np.savez(self._storeTo(), **data_arrays)
        self._storedSamples.clear()

    @property
    def totalSamples(self) -> int:
        """Get total number of samples processed.

        Returns:
            Total sample count including saved and in-memory samples.
        """
        return self._totalSamples
