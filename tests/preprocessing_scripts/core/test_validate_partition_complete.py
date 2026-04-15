"""Tests for validate_partition_complete() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_partition_complete


class TestValidatePartitionComplete:
    """Test validate_partition_complete() function."""

    def test_valid_partition(self):
        """Test valid complete disjoint partition."""
        p1 = np.array([0, 1, 2, 3])
        p2 = np.array([4, 5, 6])
        # Should not raise
        validate_partition_complete(p1, p2, total_size=7)

    def test_overlapping_partition_raises(self):
        """Test that overlapping partition raises ValueError."""
        p1 = np.array([0, 1, 2, 5])
        p2 = np.array([3, 4, 5])  # 5 is in both

        with pytest.raises(ValueError, match="overlap"):
            validate_partition_complete(p1, p2, total_size=6)

    def test_incomplete_partition_raises(self):
        """Test that incomplete partition raises ValueError."""
        p1 = np.array([0, 1, 2])
        p2 = np.array([3, 4])  # Missing 5

        with pytest.raises(ValueError, match="cover all"):
            validate_partition_complete(p1, p2, total_size=6)

    def test_over_partition_raises(self):
        """Test that over-partition (size > total) raises ValueError."""
        p1 = np.array([0, 1, 2])
        p2 = np.array([3, 4, 5])

        with pytest.raises(ValueError, match="cover all"):
            validate_partition_complete(p1, p2, total_size=5)  # Actually 6
