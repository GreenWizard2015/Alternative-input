"""Hierarchical storage for samples with chunked memory management.

Implements a linked-list based storage structure for efficiently storing
and retrieving samples while managing memory through chunking.
"""

from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np

from Core.data.SamplesStorageChunk import SamplesStorageChunk
import Core.Constants as Constants


class SamplesStorage:
    """Hierarchical storage for user/screen/camera/monitor/place-specific samples.

    Stores samples associated with a specific user, screen, camera, monitor, and place ID.
    Uses chunked linked-list structure for efficient memory management.
    Caches sample retrieval for fast access.

    Attributes:
        _head: First chunk in the linked list
        _latestT: Timestamp of most recently added sample
        _ids: Dictionary mapping ID key names to their index values (stored once to reduce memory)
    """

    def __init__(self, **ids: int) -> None:
        """Initialize samples storage with hierarchy ID context.

        Args:
            **ids: Keyword arguments mapping ID keys to their index values.
                   Required keys: userId, screenId, cameraId, monitorId, placeId

        Raises:
            AssertionError: If any required ID key is missing.

        Example:
            storage = SamplesStorage(
                userId=0, screenId=1, cameraId=2, monitorId=3, placeId=4
            )
        """
        # Validate all required hierarchy level IDs are provided
        for id_key in Constants.HIERARCHY_LEVELS:
            assert id_key in ids, (
                f"Missing required ID '{id_key}'. "
                f"Required keys: {Constants.HIERARCHY_LEVELS}"
            )

        self._head: Optional[SamplesStorageChunk] = None
        self._latestT: float = -np.inf
        # Store IDs in a dict - they are the same for all samples, so we store once to reduce memory
        # Dictionary only contains the 5 required hierarchy level IDs
        self._ids: Dict[str, int] = {
            key: ids[key] for key in Constants.HIERARCHY_LEVELS
        }

    def __len__(self) -> int:
        """Get total number of samples stored.

        Returns:
            Number of samples in storage.
        """
        if self._head:
            return self._head.count()
        return 0

    @lru_cache(10000)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve a sample by index with caching.

        Args:
            idx: Index of sample to retrieve.

        Returns:
            Dictionary of sample data with userId, screenId, cameraId, monitorId, placeId appended.
        """
        if self._head is None:
            raise ValueError("Cannot retrieve from empty storage")
        res = self._head.get(idx)
        # Append all hierarchy level IDs
        for id_key in Constants.HIERARCHY_LEVELS:
            res[id_key] = self._ids[id_key]
        return res

    def add(self, sample: Dict[str, Any]) -> int:
        """Add a single sample to storage.

        Args:
            sample: Dictionary of field names to values.

        Returns:
            Index where the sample was added.
        """
        if not self._head:
            self._head = SamplesStorageChunk()
        T = sample["time"]

        self._latestT = T
        return self._head.add(sample)

    def addBlock(self, samples: Dict[str, np.ndarray]) -> np.ndarray:
        """Add a block of numpy array samples to storage.

        Args:
            samples: Dictionary mapping field names to numpy arrays.

        Returns:
            Array of indices where samples were added.
        """
        if not all(isinstance(v, np.ndarray) for v in samples.values()):
            raise ValueError("All sample values must be numpy arrays")
        samples = {k: v for k, v in samples.items()}  # copy the dictionary
        N = len(samples["time"])

        startIndex = len(self)
        block = SamplesStorageChunk(samples=samples)
        if not self._head:
            self._head = block
        else:
            self._head.append(block)
        return np.arange(startIndex, startIndex + N)
