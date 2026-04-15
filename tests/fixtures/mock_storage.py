"""Shared mock storage for testing fixtures.

Provides a unified _MockStorage implementation used across test fixtures
to avoid duplication and maintain consistency.
"""

from typing import Dict, List, Any, Optional
import numpy as np


class _MockStorage:
    """Mock implementation of SamplesStorage for testing.

    Stores samples with support for both:
    1. Dict-based storage (numpy arrays as values) - for preprocessing tests
    2. List-based storage (simple times list) - for data sampler tests

    Implements minimal SamplesStorage-like interface with __getitem__, __len__,
    add(), and addBlock() methods.
    """

    def __init__(
        self,
        data: Optional[Any] = None,
        times: Optional[List[float]] = None,
        size: int = 10,
    ):
        """Initialize mock storage.

        Supports three initialization patterns:
        1. _MockStorage(times_list) - list directly as first arg
        2. _MockStorage(data_dict) - dict with numpy arrays as first arg
        3. _MockStorage(times=times_list) - explicit keyword arg
        4. _MockStorage(data=data_dict) - explicit keyword arg

        Args:
            data: Dict with keys like 'time', 'data', etc. (preprocessing style),
                  OR a list of times (backward compatibility).
                  Each value in dict is a numpy array where index 0 = sample 0.
            times: Simple list of times. Used for data sampler tests.
                   If provided, creates data dict with just 'time' key.
            size: Default size if neither data nor times provided.
        """
        # Handle case where a list is passed as first positional arg (for backward compat)
        if isinstance(data, list) and times is None:
            # User called _MockStorage([times_list])
            times = data
            data = None

        if data is not None and isinstance(data, dict):
            # Dict-based storage (preprocessing style)
            self.data: Dict[str, Any] = data
            self._sample_count = len(next(iter(data.values()))) if data else 0
            self.times = None
        elif times is not None:
            # List-based storage (sampler style)
            self.times = times
            self.data = {"time": np.array(times)}
            self._sample_count = len(times)
        else:
            # Default empty storage
            self.times = None
            self.data = {}
            self._sample_count = 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample at index as dict.

        Args:
            idx: Sample index (0-based)

        Returns:
            Dict with keys from data, values are single elements at idx
        """
        if self.times is not None:
            # List-based storage
            return {"time": self.times[idx]}
        else:
            # Dict-based storage
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

        if self.times is not None:
            # List-based storage
            time = sample.get("time", idx * 0.1)
            self.times.append(time)
            self.data["time"] = np.array(self.times)
        else:
            # Dict-based storage
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
