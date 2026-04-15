"""Tests for validate_indices_in_range() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_indices_in_range


class TestValidateIndicesInRange:
    """Test validate_indices_in_range() function."""

    def test_valid_indices_in_range(self):
        """Test indices within valid range."""
        indices = np.array([10, 20, 30])
        # Should not raise
        validate_indices_in_range(indices, min_allowed=0, max_allowed=100)

    def test_index_below_minimum_raises(self):
        """Test that index below minimum raises ValueError."""
        indices = np.array([5, 10, 15])

        with pytest.raises(ValueError, match="minimum index"):
            validate_indices_in_range(indices, min_allowed=10)

    def test_index_above_maximum_raises(self):
        """Test that index above maximum raises ValueError."""
        indices = np.array([10, 20, 30])

        with pytest.raises(ValueError, match="maximum index"):
            validate_indices_in_range(indices, max_allowed=25)

    def test_no_maximum_constraint(self):
        """Test with no upper limit (max_allowed=None)."""
        indices = np.array([10, 100, 1000])
        # Should not raise
        validate_indices_in_range(indices, min_allowed=0, max_allowed=None)

    def test_empty_indices_valid(self):
        """Test that empty indices array is valid."""
        indices = np.array([], dtype=np.int64)
        # Should not raise
        validate_indices_in_range(indices, min_allowed=0, max_allowed=100)
