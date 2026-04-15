"""Tests for validate_monotonic_increasing() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_monotonic_increasing


class TestValidateMonotonicIncreasing:
    """Test validate_monotonic_increasing() function."""

    def test_strictly_increasing_valid(self):
        """Test strictly increasing array."""
        arr = np.array([0.0, 0.1, 0.2, 0.3])
        # Should not raise
        validate_monotonic_increasing(arr, strict=True)

    def test_non_decreasing_with_duplicates(self):
        """Test non-decreasing array with duplicates."""
        arr = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
        # Should not raise
        validate_monotonic_increasing(arr, strict=False)

    def test_duplicate_strict_mode_raises(self):
        """Test that duplicates fail in strict mode."""
        arr = np.array([0.0, 0.1, 0.1, 0.2])

        with pytest.raises(ValueError, match="strictly increasing"):
            validate_monotonic_increasing(arr, strict=True)

    def test_decreasing_value_raises(self):
        """Test that decreasing value raises."""
        arr = np.array([0.0, 0.1, 0.05, 0.2])

        with pytest.raises(ValueError, match="not"):
            validate_monotonic_increasing(arr)

    def test_single_element_trivially_valid(self):
        """Test single element is trivially monotonic."""
        arr = np.array([0.5])
        # Should not raise
        validate_monotonic_increasing(arr)
