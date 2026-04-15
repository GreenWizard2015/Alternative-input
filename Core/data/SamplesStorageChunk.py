"""A chunk of samples in linked-list storage structure.

Implements chunked storage with automatic packing to numpy arrays
when capacity is reached, supporting efficient memory management.
"""

from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np

# Constants for chunk management
CHUNK_SIZE_LIMIT = 5000


class SamplesStorageChunk:
    """A chunk of samples in linked-list storage structure.

    Stores samples in a dictionary format with automatic packing to
    numpy arrays when capacity is reached. Forms linked list with other chunks.

    Attributes:
        _next: Pointer to next chunk (None if this is last chunk)
        _samples: Dictionary mapping field names to lists/arrays of values
    """

    def __init__(self, samples: Optional[Dict[str, Any]] = None) -> None:
        """Initialize a storage chunk.

        Args:
            samples: Dictionary of numpy arrays or None to create empty chunk.
        """
        self._next: Optional["SamplesStorageChunk"] = None
        self._samples: Dict[str, Any] = (
            defaultdict(list) if samples is None else samples
        )

    def _canAdd(self) -> bool:
        """Check if more samples can be added to this chunk.

        Chunk is full if it has next pointer or exceeds chunk size limit.

        Returns:
            True if samples can be added to this chunk, False otherwise.
        """
        if self._next:
            return False
        T = self._samples["time"]
        if not isinstance(T, list):
            return False
        if CHUNK_SIZE_LIMIT < len(T):
            return False
        return True

    def _pack(self) -> None:
        """Convert samples from lists to numpy arrays.

        Packs data into numpy arrays for memory efficiency when chunk reaches capacity.
        Converts dictionary of lists to dictionary of numpy arrays.
        """
        if isinstance(self._samples["time"], list):
            self._samples = {k: np.array(v) for k, v in self._samples.items()}

    def _ownCount(self) -> int:
        """Get count of samples in this chunk only.

        Returns:
            Number of samples stored in this chunk.
        """
        if self._samples:
            return len(self._samples["time"])
        return 0

    def count(self) -> int:
        """Get total count of samples in this chunk and all subsequent chunks.

        Returns:
            Total number of samples in linked list starting from this chunk.
        """
        cnt = self._next.count() if self._next else 0
        return self._ownCount() + cnt

    def add(self, sample: Dict[str, Any], idx: int = 0) -> int:
        """Add a sample to this chunk or delegate to next chunk.

        Args:
            sample: Dictionary of field names to values.
            idx: Current index in the sample list.

        Returns:
            Index where the sample was added in the overall storage.
        """
        if self._canAdd():
            idx += self._ownCount()
            for k, v in sample.items():
                self._samples[k].append(v)
            return idx

        if not self._next:
            self._pack()
            self._next = SamplesStorageChunk()

        return self._next.add(sample, idx + self._ownCount())

    def get(self, idx: int) -> Dict[str, Any]:
        """Retrieve a sample by index from this chunk or subsequent chunks.

        Args:
            idx: Index of sample to retrieve.

        Returns:
            Dictionary of field names to values for the sample at idx.

        Raises:
            ValueError: If index is out of bounds.
        """
        N = self._ownCount()
        if idx < N:
            return {k: v[idx] for k, v in self._samples.items()}
        if self._next is None:
            raise ValueError(f"Index {idx} out of bounds")
        return self._next.get(idx - N)

    def append(self, block: "SamplesStorageChunk") -> None:
        """Append another chunk to the end of the linked list.

        Args:
            block: Chunk to append.
        """
        if self._next and (0 < self._next._ownCount()):
            return self._next.append(block)

        self._next = block
