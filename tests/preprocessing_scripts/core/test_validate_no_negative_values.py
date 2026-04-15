"""Tests for validate_no_negative_values() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_no_negative_values


class TestValidateNoNegativeValues:
    """Test validate_no_negative_values() function."""

    def test_all_positive_valid(self):
        """Test array with all positive values."""
        arr = np.array([1, 5, 10, 20])
        # Should not raise
        validate_no_negative_values(arr)

    def test_zero_values_valid(self):
        """Test array with zeros (not negative)."""
        arr = np.array([0, 1, 5, 10])
        # Should not raise
        validate_no_negative_values(arr)

    def test_negative_values_raise(self):
        """Test that negative values raise ValueError."""
        arr = np.array([1, 5, -3, 10])

        with pytest.raises(ValueError, match="negative"):
            validate_no_negative_values(arr)

    def test_empty_array_valid(self):
        """Test empty array is valid."""
        arr = np.array([])
        # Should not raise
        validate_no_negative_values(arr)
