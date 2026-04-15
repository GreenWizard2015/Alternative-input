"""Fixtures for preprocessing script tests.

Provides mock storage, datasets, and utilities for testing preprocessing
functions in isolation from actual data loading/saving.
"""

from typing import Dict, List, Any, Optional
import pytest
import numpy as np


class _MockStorage:
    """Minimal mock storage implementing SamplesStorage interface.

    Stores samples as list of dicts. Provides dict-like access via __getitem__
    and __len__. Matches SamplesStorage API for testing purposes.
    """

    def __init__(self, data: Optional[Dict[str, np.ndarray]] = None):
        """Initialize with optional data.

        Args:
            data: Dict with keys like 'time', 'data', etc.
                 Each value is a numpy array where index 0 corresponds to sample 0.
        """
        if data is None:
            data = {}
        self.data: Dict[str, Any] = data
        self._sample_count = len(next(iter(data.values()))) if data else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample at index as dict.

        Args:
            idx: Sample index (0-based)

        Returns:
            Dict with keys from data, values are single elements at idx
        """
        return {k: v[idx] for k, v in self.data.items()}

    def __len__(self) -> int:
        """Total number of samples."""
        return self._sample_count

    def add(self, sample: Dict[str, Any]) -> int:
        """Add single sample.

        Args:
            sample: Dict with keys matching data dict

        Returns:
            Index of added sample
        """
        idx = self._sample_count
        for key, value in sample.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        self._sample_count += 1
        return idx

    def addBlock(self, samples: List[Dict[str, Any]]) -> List[int]:
        """Add multiple samples.

        Args:
            samples: List of sample dicts

        Returns:
            List of added sample indices
        """
        indices = []
        for sample in samples:
            indices.append(self.add(sample))
        return indices


# ============================================================================
# STANDARD TEST DATASETS
# ============================================================================


@pytest.fixture
def storage_10_frames():
    """10 frames at 0.1s intervals (1 second total).

    Represents minimal valid dataset for testing.
    """
    times = [i * 0.1 for i in range(10)]
    return _MockStorage({"time": np.array(times)})


@pytest.fixture
def storage_sparse_frames():
    """7 frames with larger gaps (for sparsity testing).

    Times: [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 5.0]
    Min delta: 0.5s
    """
    times = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 5.0]
    return _MockStorage({"time": np.array(times)})


# ============================================================================
# DATASET FIXTURES
# ============================================================================


@pytest.fixture
def dataset_sparse():
    """Sparse dataset for sparsity validation tests."""
    times = np.array([0.0, 1.0, 2.0, 3.0])  # 1.0s gaps
    data = np.arange(len(times))
    return {"time": times, "data": data}
