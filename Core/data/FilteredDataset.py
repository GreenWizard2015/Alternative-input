"""Filtered dataset managing storage with trajectory-based sample validation."""

from typing import Any, Dict, List, Set
import numpy as np
from Core.logging_config import get_logger

from Core.data.SampleFilter import SampleFilter


logger = get_logger(__name__)


class FilteredDataset:
    """Manages a dataset with samples filtered by trajectory requirements.

    Encapsulates storage and maintains a list of valid sample indices that meet
    minimum frame requirements within a time window. Handles sample addition,
    filtering, and access to both total and filtered samples.

    Attributes:
        _storage: Storage object containing all sample data
        _filter: SampleFilter instance for trajectory validation
        _samples: Numpy array of valid sample indices that meet minimum frame requirements
        _next_sample_index: Current position in sample iteration for sequential access
    """

    def __init__(
        self,
        storage: Any,
        minFrames: int,
        maxT: float = 1.0,
    ) -> None:
        """Initialize filtered dataset.

        Args:
            storage: Storage object containing sample data. Must support:
                - __getitem__(idx) to access sample at index
                - __len__() for total samples
                - add(sample) to add single sample
                - addBlock(samples) to add multiple samples
            minFrames: Minimum number of frames in a valid trajectory
            maxT: Maximum time window for selecting frames (default: 1.0 second)

        Raises:
            ValueError: If minFrames <= 0
            TypeError: If storage doesn't support required methods
        """
        if minFrames <= 0:
            raise ValueError(f"minFrames must be positive, got {minFrames}")

        self._storage = storage
        self._filter = SampleFilter(storage, minFrames, maxT)
        self._samples: np.ndarray = np.array([], dtype=np.int32)
        self._next_sample_index: int = 0

        logger.info(
            "Initialized FilteredDataset with minFrames=%d, maxT=%.2f",
            minFrames,
            maxT,
        )

    def add_sample(self, sample: Dict[str, Any]) -> int:
        """Add single sample to storage and filter if valid.

        Args:
            sample: Sample dictionary containing at minimum:
                - 'time': Timestamp for temporal ordering
                - Other fields specific to the data type

        Returns:
            Index of added sample in storage
        """
        idx = self._storage.add(sample)
        self._store_sample(idx)
        return idx

    def add_samples_block(self, samples: Any) -> None:
        """Add multiple samples to storage in batch.

        Args:
            samples: Samples in bulk format. Accepts either:
                - List of sample dictionaries: [{'time': 0.1, ...}, ...]
                - Batched dict with array values: {'time': [0.1, 0.2], ...}
        """
        indexes = self._storage.addBlock(samples)
        for idx in indexes:
            self._store_sample(idx)

    def _store_sample(self, idx: int) -> None:
        """Store sample if it meets minimum frame requirements.

        Args:
            idx: Index of sample to potentially store
        """
        if self._filter.isValid(idx):
            self._samples = np.append(self._samples, idx)

    def reset(self) -> None:
        """Shuffle samples for random ordering in next epoch."""
        np.random.shuffle(self._samples)
        self._next_sample_index = 0

    def __len__(self) -> int:
        """Get number of valid samples.

        Returns:
            Count of samples that meet minimum frame requirements
        """
        return len(self._samples)

    def __getitem__(self, position: int) -> Dict[str, Any]:
        """Get sample at position in filtered samples list.

        Args:
            position: Position in valid samples list (0-indexed)

        Returns:
            Sample dictionary at this position

        Raises:
            IndexError: If position is out of range
        """
        return self._storage[position]

    @property
    def total_samples(self) -> int:
        """Get total number of samples in storage.

        Returns:
            Total sample count (including invalid samples)
        """
        return len(self._storage)

    def valid_indices(self) -> List[int]:
        """Get sorted list of valid sample indices.

        Returns all sample indices that meet minimum frame requirements,
        sorted in ascending order.

        Returns:
            Sorted list of sample indices that meet filter requirements
        """
        return list(np.sort(self._samples))

    def used_samples(self) -> List[int]:
        """Get sorted union of all trajectory ranges for valid samples.

        For each valid sample, retrieves its trajectory range and includes all
        indices within that range. Returns the sorted union of all frames across
        all trajectories.

        Example:
            If valid samples are [5, 8, 12] with trajectory ranges
            [(2,5), (6,8), (10,12)], returns [2,3,4,5,6,7,8,10,11,12]

        Returns:
            Sorted list of all indices covered by any valid sample's trajectory
        """
        if len(self._samples) == 0:
            return []

        all_indices: Set[int] = set()
        for sample_idx in self._samples:
            min_idx = self.trajectory_start(sample_idx)
            all_indices.update(range(min_idx, sample_idx + 1))

        return list(sorted(all_indices))

    def trajectory_start(self, sample_idx: int) -> int:
        """Get trajectory start index for a sample.

        Args:
            sample_idx: Index of the sample

        Returns:
            Start index of the trajectory window
        """
        return self._filter.trajectory_start(sample_idx)

    @property
    def storage(self) -> Any:
        """Access underlying storage object.

        Returns:
            Reference to the storage instance
        """
        return self._storage

    @property
    def filter(self) -> SampleFilter:
        """Access underlying filter object.

        Returns:
            Reference to the SampleFilter instance
        """
        return self._filter

    def next_sample_index(self) -> int:
        """Get next sample index and advance iteration pointer.

        Returns the current sample index in valid samples list, then advances
        the pointer with wraparound (circular iteration).

        Returns:
            Index of the next valid sample in the filtered samples list

        Example:
            >>> dataset = FilteredDataset(storage, minFrames=5)
            >>> idx1 = dataset.next_sample_index()  # Returns _samples[0], advances to 1
            >>> idx2 = dataset.next_sample_index()  # Returns _samples[1], advances to 2
        """
        idx = self._samples[self._next_sample_index]
        self._next_sample_index += 1
        if len(self._samples) <= self._next_sample_index:
            self.reset()
        return idx

    def remove_sample(self, sample_idx: int) -> None:
        """Remove a sample from the valid samples list.

        Removes the sample index from the valid samples list. Used when sampling
        fails after retries to prevent repeatedly attempting to sample from
        problematic frames.

        Args:
            sample_idx: Index of the sample to remove from valid samples

        Example:
            >>> dataset = FilteredDataset(storage, minFrames=5)
            >>> dataset.remove_sample(5)  # Remove sample at index 5
        """
        mask = self._samples != sample_idx
        if not np.all(mask):  # sample_idx was found
            self._samples = self._samples[mask]
            # Adjust iteration pointer if needed
            if self._next_sample_index > 0:
                self._next_sample_index -= 1
