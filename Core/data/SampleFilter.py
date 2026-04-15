"""Sample filtering logic for trajectory validation."""

from typing import Any
from functools import lru_cache
from Core.logging_config import get_logger

logger = get_logger(__name__)


class SampleFilter:
    """Filter samples based on trajectory requirements.

    Encapsulates logic for filtering samples based on minimum frame count
    and maximum time window. Determines which samples have valid trajectories
    within the specified constraints.

    Attributes:
        _minFrames: Minimum number of frames required in a valid trajectory
        _maxT: Maximum time window for frame selection (seconds)
        _storage: Storage object containing all sample data with timestamps
    """

    def __init__(self, storage: Any, minFrames: int, maxT: float) -> None:
        """Initialize sample filter.

        Args:
            storage: Storage object containing sample data with time information.
                Must support __getitem__(idx) and __len__()
            minFrames: Minimum number of frames required in a trajectory.
                Must be > 0
            maxT: Maximum time window for selecting frames in seconds.
                Must be > 0

        Raises:
            ValueError: If minFrames or maxT <= 0
        """
        if minFrames <= 0:
            raise ValueError(f"minFrames must be positive, got {minFrames}")
        if maxT <= 0:
            raise ValueError(f"maxT must be positive, got {maxT}")

        self._minFrames = minFrames
        self._maxT = maxT
        self._storage = storage

        logger.debug(
            "Initialized SampleFilter with minFrames=%d, maxT=%.2f",
            minFrames,
            maxT,
        )

    def isValid(self, idx: int) -> bool:
        """Check if sample has valid trajectory.

        Determines if sample at given index has enough frames in its trajectory
        (based on maxT time window) to meet the minimum frame requirement.

        Args:
            idx: Index of sample to check

        Returns:
            True if sample has valid trajectory, False otherwise
        """
        minInd = self._getTrajectoryBefore(idx)
        frame_count = (idx - minInd) + 1
        return self._minFrames <= frame_count

    def _getTrajectoryBefore(self, mainInd: int) -> int:
        """Find the start of trajectory before given index.

        Returns the earliest sample index that is within maxT seconds before
        the sample at mainInd.

        Args:
            mainInd: Reference sample index

        Returns:
            Index of first sample in trajectory (earliest within maxT window)
        """
        mainT = self._storage[mainInd]["time"]
        minT = mainT - self._maxT

        minInd = mainInd
        for ind in range(mainInd - 1, -1, -1):
            if self._storage[ind]["time"] < minT:
                break
            minInd = ind

        return minInd

    @lru_cache(None)
    def trajectory_start(self, mainInd: int) -> int:
        """Get trajectory start index for a sample.

        Returns the index where the valid trajectory window starts
        (maxT seconds before the sample at mainInd). Results are cached.

        Args:
            mainInd: Reference sample index

        Returns:
            minInd: Start index of trajectory window where minInd <= mainInd
        """
        return self._getTrajectoryBefore(mainInd)
